import logging
import threading
import time
from dataclasses import replace

from domain.models import QueryIntent, RetrievalPlan
from retrievers.base import Retriever
from services.rank_fusion import reciprocal_rank_fusion


logger = logging.getLogger(__name__)


class DenseRetriever(Retriever):
    """Backward-compatible dense/sparse/hybrid in-memory retriever."""

    MODES = {"dense", "sparse", "hybrid"}

    def __init__(
        self,
        repository,
        embedding_service,
        vector_index,
        top_k=10,
        min_score=0.5,
        debug=False,
        embedding_cache=None,
        document_chunker=None,
        max_chunks_per_parent=2,
        candidate_multiplier=3,
        intent_router=None,
        intent_routing_enabled=False,
        sparse_index=None,
        retrieval_mode="dense",
        hybrid_rrf_k=60,
        hybrid_candidate_k=30,
        hybrid_dense_weight=1.0,
        hybrid_sparse_weight=1.0,
        incremental_index_enabled=True,
        force_full_rebuild=False,
    ):
        if retrieval_mode not in self.MODES:
            raise ValueError("Retrieval mode harus dense, sparse, atau hybrid.")
        if max_chunks_per_parent <= 0 or candidate_multiplier <= 0:
            raise ValueError("Chunk diversification settings harus lebih besar dari nol.")
        if hybrid_rrf_k <= 0 or hybrid_candidate_k <= 0:
            raise ValueError("Hybrid RRF k dan candidate k harus lebih besar dari nol.")
        if hybrid_dense_weight < 0 or hybrid_sparse_weight < 0:
            raise ValueError("Hybrid weights tidak boleh negatif.")
        if hybrid_dense_weight == 0 and hybrid_sparse_weight == 0:
            raise ValueError("Minimal satu hybrid weight harus lebih besar dari nol.")
        if retrieval_mode in {"sparse", "hybrid"} and sparse_index is None:
            raise ValueError("Sparse index wajib tersedia untuk mode sparse/hybrid.")

        self.repository = repository
        self.embedding_service = embedding_service
        self.vector_index = vector_index
        self.sparse_index = sparse_index
        self.top_k = top_k
        self.min_score = min_score
        self.debug = debug
        self.embedding_cache = embedding_cache
        self.document_chunker = document_chunker
        self.max_chunks_per_parent = max_chunks_per_parent
        self.candidate_multiplier = candidate_multiplier
        self.intent_router = intent_router
        self.intent_routing_enabled = intent_routing_enabled
        self.retrieval_mode = retrieval_mode
        self.hybrid_rrf_k = hybrid_rrf_k
        self.hybrid_candidate_k = hybrid_candidate_k
        self.hybrid_dense_weight = hybrid_dense_weight
        self.hybrid_sparse_weight = hybrid_sparse_weight
        self.incremental_index_enabled = incremental_index_enabled
        self.force_full_rebuild = force_full_rebuild

        self.indexing_seconds = None
        self.repository_load_seconds = None
        self.extraction_seconds = None
        self.chunking_seconds = None
        self.embedding_seconds = None
        self.cache_load_seconds = None
        self.index_build_seconds = None
        self.dense_index_build_seconds = None
        self.sparse_index_build_seconds = None
        self.cache_hit = False
        self.dense_cache_hit = False
        self.sparse_cache_hit = False
        self.hybrid_embedding_one_pass = False
        self.cache_mode = None
        self.reused_vectors = 0
        self.embedded_vectors = 0
        self.removed_vectors = 0
        self.vector_reuse_rate = 0.0
        self.registry_update_plan = None
        self.query_seconds = []
        self.query_embedding_seconds = []
        self.vector_search_seconds = []
        self.sparse_search_seconds = []
        self.fusion_seconds = []
        self.post_filter_seconds = []
        self.router_seconds = []
        self.last_retrieval_plan = None
        self.last_fallback_used = None
        self.last_candidate_count = 0
        self.last_dense_candidate_count = 0
        self.last_sparse_candidate_count = 0
        self.last_dense_candidates = []
        self.last_sparse_candidates = []
        self.last_used_filter = None
        self.last_query_sparse_nonzero = None
        self._documents = []
        self._initialized = False
        self._index_lock = threading.Lock()

    @property
    def average_query_seconds(self):
        return self._average(self.query_seconds)

    @property
    def average_query_embedding_seconds(self):
        return self._average(self.query_embedding_seconds)

    @property
    def average_vector_search_seconds(self):
        return self._average(self.vector_search_seconds)

    @property
    def average_sparse_search_seconds(self):
        return self._average(self.sparse_search_seconds)

    @property
    def average_fusion_seconds(self):
        return self._average(self.fusion_seconds)

    @property
    def average_router_seconds(self):
        return self._average(self.router_seconds)

    @staticmethod
    def _average(values):
        return sum(values) / len(values) if values else 0.0

    def initialize(self):
        if self._initialized:
            return
        prebuilt_ready = (
            self.retrieval_mode == "dense" and self.vector_index.is_built
        ) or (
            self.retrieval_mode == "sparse"
            and self.sparse_index is not None
            and self.sparse_index.is_built
        ) or (
            self.retrieval_mode == "hybrid"
            and self.vector_index.is_built
            and self.sparse_index is not None
            and self.sparse_index.is_built
        )
        if prebuilt_ready:
            source_index = (
                self.vector_index
                if self.vector_index.is_built else self.sparse_index
            )
            self._documents = source_index.documents
            if self.intent_router is not None:
                self.intent_router.update_corpus_metadata(documents=self._documents)
            self._initialized = True
            return
        with self._index_lock:
            if self._initialized:
                return
            started_at = time.perf_counter()
            repository_started_at = time.perf_counter()
            parent_documents = self.repository.get_documents()
            self.repository_load_seconds = time.perf_counter() - repository_started_at
            pdf_repository = getattr(self.repository, "pdf_repository", None)
            pdf_report = getattr(pdf_repository, "report", None)
            self.extraction_seconds = getattr(pdf_report, "elapsed_seconds", 0.0)
            if not parent_documents:
                raise RuntimeError("Tidak ada dataset valid untuk membangun retrieval index.")
            chunking_started_at = time.perf_counter()
            if self.document_chunker is not None and self.document_chunker.enabled:
                documents = self.document_chunker.chunk_documents(parent_documents)
            else:
                documents = parent_documents
            self.chunking_seconds = time.perf_counter() - chunking_started_at
            self._documents = list(documents)
            record_units = getattr(self.repository, "record_retrieval_units", None)
            if callable(record_units):
                record_units(self._documents)
            prepare_registry = getattr(self.repository, "prepare_registry_update", None)
            if callable(prepare_registry):
                self.registry_update_plan = prepare_registry(self._documents)
            model_name = getattr(self.embedding_service, "model_name", None)
            configuration = dict(
                getattr(self.embedding_service, "cache_configuration", {})
            )
            if self.document_chunker is not None:
                configuration.update(self.document_chunker.cache_configuration)

            needs_dense = self.retrieval_mode in {"dense", "hybrid"}
            needs_sparse = self.retrieval_mode in {"sparse", "hybrid"}
            if needs_sparse and not all(
                hasattr(self.embedding_service, method)
                for method in ("encode_query_sparse", "encode_documents_sparse")
            ):
                raise ValueError("Embedding backend tidak mendukung native sparse output.")

            dense_vectors = None
            sparse_vectors = None
            cache_started_at = time.perf_counter()
            incremental_dense = (
                self.retrieval_mode == "dense"
                and self.incremental_index_enabled
                and self.embedding_cache is not None
            )
            if incremental_dense:
                update = self.embedding_cache.sync_dense(
                    documents,
                    self.embedding_service,
                    configuration,
                    force_rebuild=self.force_full_rebuild,
                )
                dense_vectors = update.vectors
                self.cache_mode = update.mode
                self.reused_vectors = update.reused_vectors
                self.embedded_vectors = update.embedded_vectors
                self.removed_vectors = update.removed_vectors
                self.vector_reuse_rate = update.reuse_rate
                self.embedding_seconds = update.embedding_seconds or None
                self.dense_cache_hit = update.embedded_vectors == 0
            elif self.embedding_cache is not None:
                if needs_dense:
                    dense_vectors = self.embedding_cache.load(
                        documents, model_name, configuration
                    )
                    self.dense_cache_hit = dense_vectors is not None
                if needs_sparse:
                    sparse_vectors = self.embedding_cache.load_sparse(
                        documents, model_name, configuration
                    )
                    self.sparse_cache_hit = sparse_vectors is not None
            self.cache_load_seconds = time.perf_counter() - cache_started_at

            embedding_started_at = None
            if needs_dense and needs_sparse and dense_vectors is None and sparse_vectors is None:
                if not hasattr(self.embedding_service, "encode_documents_hybrid"):
                    raise ValueError("Embedding backend tidak mendukung hybrid encoding.")
                embedding_started_at = time.perf_counter()
                hybrid = self.embedding_service.encode_documents_hybrid(
                    document.content for document in documents
                )
                dense_vectors = hybrid.dense
                sparse_vectors = list(hybrid.sparse)
                self.hybrid_embedding_one_pass = True
                if self.embedding_cache is not None:
                    self.embedding_cache.save_hybrid(
                        documents,
                        dense_vectors,
                        sparse_vectors,
                        model_name,
                        configuration,
                    )
            else:
                if needs_dense and dense_vectors is None:
                    embedding_started_at = time.perf_counter()
                    dense_vectors = self.embedding_service.encode_documents(
                        document.content for document in documents
                    )
                    if self.embedding_cache is not None:
                        self.embedding_cache.save(
                            documents, dense_vectors, model_name, configuration
                        )
                if needs_sparse and sparse_vectors is None:
                    if embedding_started_at is None:
                        embedding_started_at = time.perf_counter()
                    sparse_vectors = self.embedding_service.encode_documents_sparse(
                        document.content for document in documents
                    )
                    if self.embedding_cache is not None:
                        self.embedding_cache.save_sparse(
                            documents, sparse_vectors, model_name, configuration
                        )
            if embedding_started_at is not None:
                self.embedding_seconds = time.perf_counter() - embedding_started_at

            index_started_at = time.perf_counter()
            if needs_dense:
                dense_started_at = time.perf_counter()
                self.vector_index.build(
                    documents,
                    dense_vectors,
                    embedding_model=model_name,
                )
                self.dense_index_build_seconds = time.perf_counter() - dense_started_at
            if needs_sparse:
                sparse_started_at = time.perf_counter()
                self.sparse_index.build(
                    documents,
                    sparse_vectors,
                    embedding_model=model_name,
                )
                self.sparse_index_build_seconds = time.perf_counter() - sparse_started_at
            self.index_build_seconds = time.perf_counter() - index_started_at

            if self.intent_router is not None:
                self.intent_router.update_corpus_metadata(documents=documents)
            commit_registry = getattr(self.repository, "commit_registry_update", None)
            if callable(commit_registry):
                commit_registry()
            self.cache_hit = (
                (not needs_dense or self.dense_cache_hit)
                and (not needs_sparse or self.sparse_cache_hit)
            )
            self.indexing_seconds = time.perf_counter() - started_at
            self._initialized = True
            logger.info(
                "In-memory retrieval index selesai: mode=%s model=%s documents=%s "
                "dense_cache=%s sparse_cache=%s total=%.4f detik.",
                self.retrieval_mode,
                model_name,
                len(documents),
                self.dense_cache_hit,
                self.sparse_cache_hit,
                self.indexing_seconds,
            )

    def retrieve(self, query, top_k=None, metadata_filter=None, plan=None):
        if not query or not query.strip():
            return []
        self.initialize()
        active_plan, router_elapsed = self._make_plan(query, metadata_filter, plan)
        self.router_seconds.append(router_elapsed)
        self.last_retrieval_plan = active_plan
        effective_filter = (
            metadata_filter if metadata_filter is not None else active_plan.metadata_filter
        )
        used_filter, fallback_used = self._resolve_metadata_filter(effective_filter)
        self.last_used_filter = used_filter
        self.last_fallback_used = fallback_used

        result_limit = top_k if top_k is not None else self.top_k
        candidate_limit = (
            max(result_limit, self.hybrid_candidate_k)
            if self.retrieval_mode == "hybrid"
            else result_limit * self.candidate_multiplier
        )
        started_at = time.perf_counter()
        embedding_started_at = time.perf_counter()
        dense_query = None
        sparse_query = None
        if self.retrieval_mode == "dense":
            dense_query = self.embedding_service.encode_query(query)
        elif self.retrieval_mode == "sparse":
            sparse_query = self.embedding_service.encode_query_sparse(query)
        else:
            hybrid_query = self.embedding_service.encode_query_hybrid(query)
            dense_query = hybrid_query.dense
            sparse_query = hybrid_query.sparse
        self.last_query_sparse_nonzero = (
            len(sparse_query) if sparse_query is not None else None
        )
        self.query_embedding_seconds.append(time.perf_counter() - embedding_started_at)

        dense_results = []
        dense_started_at = time.perf_counter()
        if dense_query is not None:
            dense_results = self.vector_index.search(
                dense_query,
                top_k=candidate_limit,
                embedding_model=getattr(self.embedding_service, "model_name", None),
                metadata_filter=used_filter,
            )
            dense_results = [
                replace(
                    item,
                    dense_rank=rank,
                    dense_score=item.score,
                    final_rank=rank,
                )
                for rank, item in enumerate(dense_results, 1)
            ]
        self.vector_search_seconds.append(time.perf_counter() - dense_started_at)

        sparse_results = []
        sparse_started_at = time.perf_counter()
        if sparse_query is not None:
            sparse_results = self.sparse_index.search(
                sparse_query,
                top_k=candidate_limit,
                embedding_model=getattr(self.embedding_service, "model_name", None),
                metadata_filter=used_filter,
            )
            sparse_results = [
                replace(
                    item,
                    sparse_rank=rank,
                    sparse_score=item.score,
                    final_rank=rank,
                )
                for rank, item in enumerate(sparse_results, 1)
            ]
        self.sparse_search_seconds.append(time.perf_counter() - sparse_started_at)
        self.last_dense_candidates = dense_results
        self.last_sparse_candidates = sparse_results
        self.last_dense_candidate_count = len(dense_results)
        self.last_sparse_candidate_count = len(sparse_results)

        fusion_started_at = time.perf_counter()
        if self.retrieval_mode == "hybrid":
            candidates = reciprocal_rank_fusion(
                dense_results,
                sparse_results,
                rrf_k=self.hybrid_rrf_k,
                dense_weight=self.hybrid_dense_weight,
                sparse_weight=self.hybrid_sparse_weight,
            )
        elif self.retrieval_mode == "dense":
            candidates = dense_results
        else:
            candidates = sparse_results
        self.fusion_seconds.append(time.perf_counter() - fusion_started_at)
        self.last_candidate_count = len(candidates)

        diversify_started_at = time.perf_counter()
        results = self._diversify(candidates, result_limit)
        if self.retrieval_mode == "dense" and self.min_score is not None:
            filtered = [item for item in results if item.score >= self.min_score]
            results = filtered or results
        results = [replace(item, final_rank=rank) for rank, item in enumerate(results, 1)]
        self.post_filter_seconds.append(time.perf_counter() - diversify_started_at)
        self.query_seconds.append(time.perf_counter() - started_at)

        if self.debug:
            self._log_debug(query, active_plan, used_filter, results)
        return results

    def retrieve_plan(self, plan, top_k=None):
        return self.retrieve(plan.query, top_k=top_k, plan=plan)

    def _make_plan(self, query, metadata_filter, plan):
        started_at = time.perf_counter()
        active_plan = plan
        if (
            active_plan is None
            and metadata_filter is None
            and self.intent_routing_enabled
            and self.intent_router is not None
        ):
            active_plan = self.intent_router.route(query)
        if active_plan is None:
            active_plan = RetrievalPlan(
                query=query,
                intent=QueryIntent.GENERAL,
                metadata_filter=metadata_filter,
                reason=(
                    "explicit metadata filter"
                    if metadata_filter else "intent routing disabled"
                ),
            )
        return active_plan, time.perf_counter() - started_at

    def _resolve_metadata_filter(self, metadata_filter):
        filters = [metadata_filter]
        if metadata_filter and "chunk_role" in metadata_filter:
            filters.append(
                {key: value for key, value in metadata_filter.items() if key != "chunk_role"}
            )
        if metadata_filter:
            filters.append(None)
        seen = set()
        for attempt, candidate_filter in enumerate(filters):
            signature = tuple(sorted((candidate_filter or {}).items()))
            if signature in seen:
                continue
            seen.add(signature)
            count = sum(
                not candidate_filter
                or all(
                    document.metadata.get(key) == value
                    for key, value in candidate_filter.items()
                )
                for document in self._documents
            )
            if count:
                fallback = None if attempt == 0 else (
                    "type_only" if candidate_filter else "general"
                )
                return candidate_filter, fallback
        return metadata_filter, "no_candidates"

    def _diversify(self, candidates, result_limit):
        diversified = []
        parent_counts = {}
        for item in candidates:
            parent_id = getattr(item.document, "parent_document_id", item.document.id)
            if parent_counts.get(parent_id, 0) >= self.max_chunks_per_parent:
                continue
            diversified.append(item)
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
            if len(diversified) >= result_limit:
                break
        return diversified

    def _log_debug(self, query, plan, used_filter, results):
        logger.info(
            "RAG debug query=%r mode=%s intent=%s confidence=%s filter=%s "
            "preferred_role=%s fallback=%s dense_candidates=%s sparse_candidates=%s",
            query,
            self.retrieval_mode,
            plan.intent.value,
            plan.confidence.value,
            used_filter,
            plan.preferred_chunk_role,
            self.last_fallback_used,
            self.last_dense_candidate_count,
            self.last_sparse_candidate_count,
        )
        for item in results:
            logger.info(
                "RAG result id=%s dense_rank=%s dense_score=%s sparse_rank=%s "
                "sparse_score=%s rrf_score=%s final_rank=%s",
                item.document.id,
                item.dense_rank,
                item.dense_score,
                item.sparse_rank,
                item.sparse_score,
                item.fusion_score,
                item.final_rank,
            )
