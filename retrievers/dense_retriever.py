import logging
import threading
import time

from retrievers.base import Retriever


logger = logging.getLogger(__name__)


class DenseRetriever(Retriever):
    """Dense cosine retriever yang model-agnostic melalui EmbeddingService."""

    def __init__(
        self,
        repository,
        embedding_service,
        vector_index,
        top_k=10,
        min_score=0.5,
        debug=False,
        embedding_cache=None,
    ):
        self.repository = repository
        self.embedding_service = embedding_service
        self.vector_index = vector_index
        self.top_k = top_k
        self.min_score = min_score
        self.debug = debug
        self.embedding_cache = embedding_cache
        self.indexing_seconds = None
        self.embedding_seconds = None
        self.cache_load_seconds = None
        self.index_build_seconds = None
        self.cache_hit = False
        self.query_seconds = []
        self.query_embedding_seconds = []
        self.vector_search_seconds = []
        self._index_lock = threading.Lock()

    @property
    def average_query_seconds(self):
        if not self.query_seconds:
            return 0.0
        return sum(self.query_seconds) / len(self.query_seconds)

    @property
    def average_query_embedding_seconds(self):
        if not self.query_embedding_seconds:
            return 0.0
        return sum(self.query_embedding_seconds) / len(self.query_embedding_seconds)

    @property
    def average_vector_search_seconds(self):
        if not self.vector_search_seconds:
            return 0.0
        return sum(self.vector_search_seconds) / len(self.vector_search_seconds)

    def initialize(self):
        if self.vector_index.is_built:
            return

        with self._index_lock:
            if self.vector_index.is_built:
                return
            started_at = time.perf_counter()
            documents = self.repository.get_documents()
            if not documents:
                raise RuntimeError("Tidak ada dataset valid untuk membangun retrieval index.")
            model_name = getattr(self.embedding_service, "model_name", None)
            cache_configuration = getattr(
                self.embedding_service, "cache_configuration", {}
            )
            embeddings = None
            if self.embedding_cache is not None:
                cache_started_at = time.perf_counter()
                embeddings = self.embedding_cache.load(
                    documents, model_name, cache_configuration
                )
                self.cache_load_seconds = time.perf_counter() - cache_started_at
                self.cache_hit = embeddings is not None

            if embeddings is None:
                embedding_started_at = time.perf_counter()
                embeddings = self.embedding_service.encode_documents(
                    document.content for document in documents
                )
                self.embedding_seconds = time.perf_counter() - embedding_started_at
                if self.embedding_cache is not None:
                    self.embedding_cache.save(
                        documents,
                        embeddings,
                        getattr(self.embedding_service, "model_name", model_name),
                        getattr(self.embedding_service, "cache_configuration", {}),
                    )

            index_started_at = time.perf_counter()
            self.vector_index.build(
                documents,
                embeddings,
                embedding_model=getattr(self.embedding_service, "model_name", None),
            )
            self.index_build_seconds = time.perf_counter() - index_started_at
            self.indexing_seconds = time.perf_counter() - started_at
            logger.info(
                "In-memory index selesai: model=%s dimension=%s documents=%s "
                "cache_hit=%s total=%.4f detik.",
                self.vector_index.embedding_model,
                self.vector_index.embedding_dimension,
                len(documents),
                self.cache_hit,
                self.indexing_seconds,
            )

    def retrieve(self, query, top_k=None):
        if not query or not query.strip():
            return []

        self.initialize()
        result_limit = top_k if top_k is not None else self.top_k
        started_at = time.perf_counter()
        embedding_started_at = time.perf_counter()
        query_vector = self.embedding_service.encode_query(query)
        self.query_embedding_seconds.append(time.perf_counter() - embedding_started_at)
        search_started_at = time.perf_counter()
        candidates = self.vector_index.search(
            query_vector,
            top_k=result_limit,
            embedding_model=getattr(self.embedding_service, "model_name", None),
        )
        self.vector_search_seconds.append(time.perf_counter() - search_started_at)

        if self.min_score is None:
            results = candidates
        else:
            filtered = [item for item in candidates if item.score >= self.min_score]
            results = filtered or candidates

        elapsed = time.perf_counter() - started_at
        self.query_seconds.append(elapsed)

        if self.debug:
            logger.info("RAG debug query=%r top_k=%s", query, result_limit)
            for item in results:
                logger.info(
                    "RAG result id=%s type=%s score=%.6f",
                    item.document.id,
                    item.document.metadata.get("type"),
                    item.score,
                )

        return results
