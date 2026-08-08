import hashlib
import gzip
import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)
CACHE_VERSION = 1
INCREMENTAL_DENSE_FORMAT = "incremental-dense-v1"
SPARSE_FORMAT_VERSION = 1
FINGERPRINT_METADATA_FIELDS = (
    "type",
    "source",
    "name",
    "code",
    "role",
    "parent_document_id",
    "chunk_index",
    "section",
    "chunk_role",
    "research_interest_source",
    "source_type",
    "source_file",
    "document_title",
    "source_document_id",
    "page_start",
    "page_end",
    "content_hash",
)


def corpus_fingerprint(documents):
    """Hash deterministik dari identity dan knowledge content yang di-index."""
    digest = hashlib.sha256()
    for document in documents:
        stable_metadata = {
            field: document.metadata.get(field)
            for field in FINGERPRINT_METADATA_FIELDS
            if field in document.metadata
        }
        payload = {
            "id": document.id,
            "content": document.content,
            "metadata": stable_metadata,
        }
        digest.update(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def chunk_content_hash(document):
    """Hash representation final yang benar-benar diberikan ke embedder."""
    return hashlib.sha256(str(document.content).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DenseCacheUpdate:
    vectors: object
    mode: str
    active_vectors: int
    reused_vectors: int
    embedded_vectors: int
    removed_vectors: int
    reuse_rate: float
    embedding_seconds: float
    migrated_legacy: bool = False
    fallback_reason: str | None = None


class DocumentEmbeddingCache:
    """Persistent document-vector cache; tidak menyimpan query atau model weights."""

    def __init__(self, directory, cache_key, enabled=True):
        self.directory = Path(directory)
        self.cache_key = cache_key
        self.enabled = enabled
        self.vector_path = self.directory / f"{cache_key}_dense.npz"
        self.metadata_path = self.directory / f"{cache_key}_dense_metadata.json"
        self.sparse_path = self.directory / f"{cache_key}_sparse.json.gz"
        self.last_status = "disabled" if not enabled else "not_checked"
        self.dense_status = self.last_status
        self.sparse_status = self.last_status
        self.last_update = None

    def plan_dense_update(self, documents, model_name, configuration, force_rebuild=False):
        """Estimasi update tanpa membaca vector, memuat model, atau mengubah state."""
        document_list = list(documents)
        records, reason = self._read_incremental_records(model_name, configuration)
        if not force_rebuild and records is None and reason == "legacy_cache":
            try:
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                legacy_exact = (
                    metadata.get("cache_version") == CACHE_VERSION
                    and metadata.get("embedding_model") == model_name
                    and metadata.get("configuration") == configuration
                    and metadata.get("document_count") == len(document_list)
                    and metadata.get("corpus_fingerprint") == corpus_fingerprint(document_list)
                    and self.vector_path.exists()
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                legacy_exact = False
            if legacy_exact:
                return {
                    "cache_mode": "legacy_migration_candidate",
                    "chunks_active": len(document_list),
                    "chunks_reused": len(document_list),
                    "chunks_to_embed": 0,
                    "chunks_removed": 0,
                    "reason": "verified_v1_order_and_fingerprint",
                }
        if force_rebuild or records is None:
            return {
                "cache_mode": "full_rebuild",
                "chunks_active": len(document_list),
                "chunks_reused": 0,
                "chunks_to_embed": len(document_list),
                "chunks_removed": self._active_record_count() if records else 0,
                "reason": "explicit_rebuild" if force_rebuild else reason,
            }
        reusable, used = self._match_records(document_list, records)
        active_records = self._read_active_records(records)
        previous_active = {item["vector_index"] for item in active_records}
        removed_count = len(previous_active - used)
        return {
            "cache_mode": "full_hit" if len(reusable) == len(document_list) and removed_count == 0 else "incremental",
            "chunks_active": len(document_list),
            "chunks_reused": len(reusable),
            "chunks_to_embed": len(document_list) - len(reusable),
            "chunks_removed": removed_count,
            "reason": None,
        }

    def sync_dense(self, documents, embedding_service, configuration, force_rebuild=False):
        with self.update_lock():
            return self._sync_dense_unlocked(
                documents, embedding_service, configuration, force_rebuild=force_rebuild
            )

    def _sync_dense_unlocked(self, documents, embedding_service, configuration, force_rebuild=False):
        """Reuse vector per chunk/content lalu assemble matrix aktif secara atomik."""
        document_list = list(documents)
        if not document_list:
            raise ValueError("Cache incremental memerlukan minimal satu document.")
        model_name = getattr(embedding_service, "model_name", None)
        if not self.enabled:
            started = time.perf_counter()
            vectors = np.asarray(embedding_service.encode_documents(d.content for d in document_list))
            result = DenseCacheUpdate(vectors, "disabled", len(document_list), 0, len(document_list), 0, 0.0, time.perf_counter() - started)
            self.last_update = result
            return result

        previous_vectors = None
        previous_records = None
        previous_active_records = []
        fallback_reason = None
        migrated_legacy = False
        if not force_rebuild:
            previous_records, fallback_reason = self._read_incremental_records(model_name, configuration)
            if previous_records is not None:
                previous_vectors = self._read_incremental_vectors(previous_records)
                previous_active_records = self._read_active_records(previous_records)
                if previous_vectors is None:
                    fallback_reason = "corrupt_incremental_vectors"
                    previous_records = None
            elif fallback_reason == "legacy_cache":
                legacy = self.load(document_list, model_name, configuration)
                if legacy is not None:
                    previous_vectors = np.asarray(legacy)
                    previous_records = self._chunk_records(document_list)
                    previous_active_records = [
                        {**record, "vector_index": index}
                        for index, record in enumerate(previous_records)
                    ]
                    migrated_legacy = True
                    fallback_reason = None

        reused_by_position = {}
        used_previous = set()
        if previous_records is not None and previous_vectors is not None:
            matched, used_previous = self._match_records(document_list, previous_records)
            for current_index, previous_index in matched.items():
                reused_by_position[current_index] = previous_vectors[previous_index]

        missing_indices = [i for i in range(len(document_list)) if i not in reused_by_position]
        embedding_seconds = 0.0
        new_vectors = None
        if missing_indices:
            started = time.perf_counter()
            new_vectors = np.asarray(
                embedding_service.encode_documents(document_list[i].content for i in missing_indices)
            )
            embedding_seconds = time.perf_counter() - started
            if new_vectors.ndim != 2 or new_vectors.shape[0] != len(missing_indices):
                raise ValueError("Embedder mengembalikan matrix incremental yang invalid.")

        dimension = None
        if previous_vectors is not None and previous_vectors.ndim == 2:
            dimension = previous_vectors.shape[1]
        if new_vectors is not None:
            if dimension is not None and new_vectors.shape[1] != dimension:
                raise ValueError("Dimensi vector baru tidak cocok dengan cache reusable.")
            dimension = new_vectors.shape[1]
        if not dimension:
            raise ValueError("Dimensi embedding incremental tidak dapat ditentukan.")

        pool_records = list(previous_records or [])
        pool_rows = list(previous_vectors) if previous_vectors is not None else []
        new_pool_indexes = {}
        for offset, current_index in enumerate(missing_indices):
            new_pool_indexes[current_index] = len(pool_rows)
            pool_rows.append(new_vectors[offset])
            pool_records.append({
                "chunk_id": document_list[current_index].id,
                "content_hash": chunk_content_hash(document_list[current_index]),
            })

        rows = []
        active_records = []
        new_cursor = 0
        for index in range(len(document_list)):
            if index in reused_by_position:
                vector_index = self._matching_pool_index(document_list[index], pool_records, active_records)
                row = pool_rows[vector_index]
            else:
                vector_index = new_pool_indexes[index]
                row = pool_rows[vector_index]
                new_cursor += 1
            rows.append(row)
            active_records.append({
                "chunk_id": document_list[index].id,
                "content_hash": chunk_content_hash(document_list[index]),
                "vector_index": vector_index,
            })
        vectors = np.asarray(rows)
        if vectors.shape != (len(document_list), dimension) or not np.isfinite(vectors).all():
            raise ValueError("Candidate matrix incremental gagal validasi.")

        previous_active_indexes = {
            int(item["vector_index"]) for item in previous_active_records
            if isinstance(item.get("vector_index"), int)
        }
        current_reused_indexes = {
            int(item["vector_index"]) for item in active_records
            if int(item["vector_index"]) < len(previous_records or [])
        }
        removed = len(previous_active_indexes - current_reused_indexes)
        reused = len(reused_by_position)
        embedded = len(missing_indices)
        if previous_records is None:
            mode = "full_rebuild"
        elif embedded == 0 and removed == 0:
            mode = "full_hit"
        else:
            mode = "incremental"
        self._save_incremental(
            document_list, vectors, model_name, configuration,
            pool_records=pool_records, pool_vectors=np.asarray(pool_rows),
            active_records=active_records,
        )
        result = DenseCacheUpdate(
            vectors=vectors,
            mode=mode,
            active_vectors=len(document_list),
            reused_vectors=reused,
            embedded_vectors=embedded,
            removed_vectors=max(0, removed),
            reuse_rate=(reused / len(document_list)),
            embedding_seconds=embedding_seconds,
            migrated_legacy=migrated_legacy,
            fallback_reason=fallback_reason,
        )
        self.last_update = result
        self.last_status = self.dense_status = mode
        return result

    @contextmanager
    def update_lock(self, stale_seconds=900):
        """Lock file ringan untuk mencegah dua writer cache dense bersamaan."""
        if not self.enabled:
            yield
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        lock_path = self.directory / f"{self.cache_key}.lock"
        if lock_path.exists():
            age = time.time() - lock_path.stat().st_mtime
            if age > stale_seconds:
                lock_path.unlink()
        descriptor = None
        owns_lock = False
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            owns_lock = True
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.close(descriptor)
            descriptor = None
            yield
        except FileExistsError as exc:
            raise RuntimeError("Knowledge cache sedang diperbarui oleh process lain.") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if owns_lock and lock_path.exists():
                try:
                    lock_path.unlink()
                except OSError:
                    pass

    def _read_incremental_records(self, model_name, configuration):
        if not self.metadata_path.exists() or not self.vector_path.exists():
            return None, "cache_missing"
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None, "corrupt_metadata"
        if metadata.get("cache_format_version") != INCREMENTAL_DENSE_FORMAT:
            return None, "legacy_cache"
        if metadata.get("embedding_model") != model_name:
            return None, "embedding_model_changed"
        if metadata.get("configuration") != configuration:
            return None, "embedding_or_chunking_configuration_changed"
        records = metadata.get("chunk_records")
        if not isinstance(records, list) or not records:
            return None, "invalid_chunk_records"
        if any(not isinstance(item, dict) or not item.get("chunk_id") or not item.get("content_hash") for item in records):
            return None, "invalid_chunk_records"
        return records, None

    def _read_incremental_vectors(self, records):
        try:
            with np.load(self.vector_path, allow_pickle=False) as stored:
                vectors = np.asarray(stored["embeddings"])
            if vectors.ndim != 2 or vectors.shape[0] != len(records) or vectors.size == 0 or not np.isfinite(vectors).all():
                return None
            return vectors
        except (OSError, ValueError, KeyError):
            return None

    def _read_active_records(self, records):
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            active = metadata.get("active_chunk_records")
            if isinstance(active, list) and all(
                isinstance(item, dict) and isinstance(item.get("vector_index"), int)
                and 0 <= item["vector_index"] < len(records)
                for item in active
            ):
                return active
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        return [{**record, "vector_index": index} for index, record in enumerate(records)]

    def _active_record_count(self):
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            return len(metadata.get("active_chunk_records") or metadata.get("chunk_records") or [])
        except (OSError, UnicodeError, json.JSONDecodeError):
            return 0

    @staticmethod
    def _matching_pool_index(document, records, already_active):
        digest = chunk_content_hash(document)
        used = {item["vector_index"] for item in already_active}
        for index, record in enumerate(records):
            if index not in used and record["chunk_id"] == document.id and record["content_hash"] == digest:
                return index
        for index, record in enumerate(records):
            if index not in used and record["content_hash"] == digest:
                return index
        raise ValueError("Vector reusable tidak ditemukan saat assembly.")

    @staticmethod
    def _chunk_records(documents):
        return [{"chunk_id": item.id, "content_hash": chunk_content_hash(item)} for item in documents]

    @staticmethod
    def _match_records(documents, records):
        exact = {(item["chunk_id"], item["content_hash"]): index for index, item in enumerate(records)}
        by_content = {}
        for index, item in enumerate(records):
            by_content.setdefault(item["content_hash"], []).append(index)
        matched = {}
        used = set()
        for current_index, document in enumerate(documents):
            digest = chunk_content_hash(document)
            previous_index = exact.get((document.id, digest))
            if previous_index is None:
                candidates = [i for i in by_content.get(digest, []) if i not in used]
                previous_index = candidates[0] if candidates else None
            if previous_index is not None and previous_index not in used:
                matched[current_index] = previous_index
                used.add(previous_index)
        return matched, used

    def _save_incremental(
        self, documents, vectors, model_name, configuration,
        pool_records=None, pool_vectors=None, active_records=None,
    ):
        self.directory.mkdir(parents=True, exist_ok=True)
        pool_records = pool_records or self._chunk_records(documents)
        pool_vectors = np.asarray(pool_vectors if pool_vectors is not None else vectors)
        active_records = active_records or [
            {**record, "vector_index": index}
            for index, record in enumerate(self._chunk_records(documents))
        ]
        metadata = {
            "cache_version": CACHE_VERSION,
            "cache_format_version": INCREMENTAL_DENSE_FORMAT,
            "embedding_model": model_name,
            "embedding_dimension": int(vectors.shape[1]),
            "document_count": len(documents),
            "corpus_fingerprint": corpus_fingerprint(documents),
            "configuration": configuration,
            "chunk_records": pool_records,
            "active_chunk_records": active_records,
            "vector_pool_count": len(pool_records),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        vector_temp = metadata_temp = None
        vector_backup = metadata_backup = None
        vector_committed = metadata_committed = False
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=self.directory, delete=False) as handle:
                vector_temp = Path(handle.name)
                np.savez_compressed(handle, embeddings=pool_vectors)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory, delete=False) as handle:
                metadata_temp = Path(handle.name)
                json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
            with np.load(vector_temp, allow_pickle=False) as stored:
                candidate = np.asarray(stored["embeddings"])
            candidate_metadata = json.loads(metadata_temp.read_text(encoding="utf-8"))
            if candidate.shape != pool_vectors.shape or len(candidate_metadata["chunk_records"]) != len(pool_records) or len(candidate_metadata["active_chunk_records"]) != len(documents):
                raise ValueError("Candidate cache incremental gagal validasi sebelum commit.")

            # Kedua artifact membentuk satu unit cache. Backup lokal memungkinkan
            # rollback bila penggantian file kedua gagal setelah file pertama sukses.
            if self.vector_path.exists():
                with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as handle:
                    vector_backup = Path(handle.name)
                shutil.copyfile(self.vector_path, vector_backup)
            if self.metadata_path.exists():
                with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as handle:
                    metadata_backup = Path(handle.name)
                shutil.copyfile(self.metadata_path, metadata_backup)

            os.replace(vector_temp, self.vector_path)
            vector_committed = True
            vector_temp = None
            os.replace(metadata_temp, self.metadata_path)
            metadata_committed = True
            metadata_temp = None
        except Exception:
            # Best-effort rollback ke pasangan cache valid sebelumnya. Jika cache
            # belum pernah ada, hapus candidate yang sempat terpasang.
            if vector_backup and vector_backup.exists():
                os.replace(vector_backup, self.vector_path)
                vector_backup = None
            elif vector_committed and self.vector_path.exists():
                self.vector_path.unlink()
            if metadata_backup and metadata_backup.exists():
                os.replace(metadata_backup, self.metadata_path)
                metadata_backup = None
            elif metadata_committed and self.metadata_path.exists():
                self.metadata_path.unlink()
            raise
        finally:
            for path in (vector_temp, metadata_temp, vector_backup, metadata_backup):
                if path and path.exists():
                    path.unlink()

    def load(self, documents, model_name, configuration):
        if not self.enabled:
            self.last_status = "disabled"
            self.dense_status = self.last_status
            return None
        if not self.vector_path.exists() or not self.metadata_path.exists():
            self.last_status = "miss"
            self.dense_status = self.last_status
            logger.info("Embedding cache miss: file belum tersedia.")
            return None

        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            expected = {
                "cache_version": CACHE_VERSION,
                "embedding_model": model_name,
                "document_count": len(documents),
                "corpus_fingerprint": corpus_fingerprint(documents),
                "configuration": configuration,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    self.last_status = f"invalid_{key}"
                    self.dense_status = self.last_status
                    logger.warning("Embedding cache invalid: %s berubah.", key)
                    return None

            with np.load(self.vector_path, allow_pickle=False) as stored:
                vectors = np.asarray(stored["embeddings"])
            dimension = metadata.get("embedding_dimension")
            if (
                vectors.ndim != 2
                or vectors.shape[0] != len(documents)
                or not isinstance(dimension, int)
                or vectors.shape[1] != dimension
                or vectors.size == 0
                or not np.isfinite(vectors).all()
            ):
                self.last_status = "invalid_vectors"
                self.dense_status = self.last_status
                logger.warning("Embedding cache invalid: bentuk atau nilai vector salah.")
                return None

            self.last_status = "hit"
            self.dense_status = self.last_status
            logger.info(
                "Embedding cache hit: model=%s documents=%s dimension=%s",
                model_name,
                len(documents),
                dimension,
            )
            return vectors
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.last_status = "corrupt"
            self.dense_status = self.last_status
            logger.warning(
                "Embedding cache corrupt (%s); corpus akan di-encode ulang.",
                type(exc).__name__,
            )
            return None

    def save(self, documents, embeddings, model_name, configuration):
        if not self.enabled:
            return
        vectors = np.asarray(embeddings)
        if vectors.ndim != 2 or vectors.shape[0] != len(documents):
            raise ValueError("Embedding cache tidak dapat menyimpan bentuk vector invalid.")
        if vectors.size == 0 or not np.isfinite(vectors).all():
            raise ValueError("Embedding cache tidak dapat menyimpan vector kosong/invalid.")

        self.directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "cache_version": CACHE_VERSION,
            "embedding_model": model_name,
            "embedding_dimension": int(vectors.shape[1]),
            "dense_dimension": int(vectors.shape[1]),
            "document_count": len(documents),
            "corpus_fingerprint": corpus_fingerprint(documents),
            "configuration": configuration,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if configuration.get("chunking_enabled"):
            metadata.update(
                {
                    "chunking_enabled": True,
                    "chunk_max_tokens": configuration.get("chunk_max_tokens"),
                    "chunk_overlap_tokens": configuration.get(
                        "chunk_overlap_tokens"
                    ),
                    "chunking_version": configuration.get("chunking_version"),
                    "chunk_count": len(documents),
                }
            )

        vector_temp = None
        metadata_temp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.directory, delete=False
            ) as handle:
                vector_temp = Path(handle.name)
                np.savez_compressed(handle, embeddings=vectors)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory, delete=False
            ) as handle:
                metadata_temp = Path(handle.name)
                json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(vector_temp, self.vector_path)
            os.replace(metadata_temp, self.metadata_path)
            self.last_status = "saved"
            self.dense_status = self.last_status
            logger.info("Embedding cache disimpan: %s", self.vector_path)
        finally:
            for temporary_path in (vector_temp, metadata_temp):
                if temporary_path and temporary_path.exists():
                    temporary_path.unlink()

    def load_sparse(self, documents, model_name, configuration):
        if not self.enabled:
            self.last_status = self.sparse_status = "disabled"
            return None
        if not self.sparse_path.exists() or not self.metadata_path.exists():
            self.last_status = self.sparse_status = "sparse_miss"
            return None
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            expected = {
                "cache_version": CACHE_VERSION,
                "embedding_model": model_name,
                "document_count": len(documents),
                "corpus_fingerprint": corpus_fingerprint(documents),
                "configuration": configuration,
                "sparse_enabled": True,
                "sparse_format_version": SPARSE_FORMAT_VERSION,
                "sparse_document_count": len(documents),
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    self.last_status = self.sparse_status = f"invalid_sparse_{key}"
                    return None
            with gzip.open(self.sparse_path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("format_version") != SPARSE_FORMAT_VERSION:
                self.last_status = self.sparse_status = "invalid_sparse_format"
                return None
            raw_vectors = payload.get("vectors")
            if not isinstance(raw_vectors, list) or len(raw_vectors) != len(documents):
                self.last_status = self.sparse_status = "invalid_sparse_vectors"
                return None
            from domain.models import SparseVector

            vectors = []
            for raw_vector in raw_vectors:
                if not isinstance(raw_vector, list):
                    raise ValueError("Sparse cache vector bukan list.")
                vector = SparseVector.from_mapping(dict(raw_vector))
                if not vector.values:
                    raise ValueError("Sparse cache vector kosong.")
                vectors.append(vector)
            self.last_status = self.sparse_status = "sparse_hit"
            return vectors
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            UnicodeError,
            json.JSONDecodeError,
            gzip.BadGzipFile,
        ) as exc:
            self.last_status = self.sparse_status = "sparse_corrupt"
            logger.warning(
                "Sparse embedding cache corrupt (%s); corpus akan di-encode ulang.",
                type(exc).__name__,
            )
            return None

    def save_sparse(self, documents, sparse_vectors, model_name, configuration):
        if not self.enabled:
            return
        vectors = list(sparse_vectors)
        if len(vectors) != len(documents) or not vectors:
            raise ValueError("Jumlah sparse vector cache tidak sesuai document.")
        serialized = []
        for vector in vectors:
            values = list(getattr(vector, "values", ()))
            if not values:
                raise ValueError("Sparse cache tidak dapat menyimpan vector kosong.")
            serialized.append(values)

        self.directory.mkdir(parents=True, exist_ok=True)
        metadata = {}
        if self.metadata_path.exists():
            try:
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
        metadata.update(
            {
                "cache_version": CACHE_VERSION,
                "embedding_model": model_name,
                "document_count": len(documents),
                "corpus_fingerprint": corpus_fingerprint(documents),
                "configuration": configuration,
                "sparse_enabled": True,
                "sparse_format_version": SPARSE_FORMAT_VERSION,
                "sparse_document_count": len(documents),
                "sparse_nonzero_total": sum(len(values) for values in serialized),
                "created_at": metadata.get("created_at")
                or datetime.now(timezone.utc).isoformat(),
            }
        )
        if configuration.get("chunking_enabled"):
            metadata.update(
                {
                    "chunking_enabled": True,
                    "chunk_max_tokens": configuration.get("chunk_max_tokens"),
                    "chunk_overlap_tokens": configuration.get("chunk_overlap_tokens"),
                    "chunking_version": configuration.get("chunking_version"),
                    "chunk_count": len(documents),
                }
            )

        sparse_temp = None
        metadata_temp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.directory, delete=False
            ) as handle:
                sparse_temp = Path(handle.name)
            with gzip.open(sparse_temp, "wt", encoding="utf-8") as handle:
                json.dump(
                    {"format_version": SPARSE_FORMAT_VERSION, "vectors": serialized},
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory, delete=False
            ) as handle:
                metadata_temp = Path(handle.name)
                json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(sparse_temp, self.sparse_path)
            os.replace(metadata_temp, self.metadata_path)
            self.last_status = self.sparse_status = "sparse_saved"
        finally:
            for temporary_path in (sparse_temp, metadata_temp):
                if temporary_path and temporary_path.exists():
                    temporary_path.unlink()

    def save_hybrid(
        self, documents, dense_vectors, sparse_vectors, model_name, configuration
    ):
        self.save(documents, dense_vectors, model_name, configuration)
        self.save_sparse(documents, sparse_vectors, model_name, configuration)
