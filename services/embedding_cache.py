import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)
CACHE_VERSION = 1
FINGERPRINT_METADATA_FIELDS = ("type", "source", "name", "code", "role")


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


class DocumentEmbeddingCache:
    """Persistent document-vector cache; tidak menyimpan query atau model weights."""

    def __init__(self, directory, cache_key, enabled=True):
        self.directory = Path(directory)
        self.cache_key = cache_key
        self.enabled = enabled
        self.vector_path = self.directory / f"{cache_key}_dense.npz"
        self.metadata_path = self.directory / f"{cache_key}_dense_metadata.json"
        self.last_status = "disabled" if not enabled else "not_checked"

    def load(self, documents, model_name, configuration):
        if not self.enabled:
            self.last_status = "disabled"
            return None
        if not self.vector_path.exists() or not self.metadata_path.exists():
            self.last_status = "miss"
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
                logger.warning("Embedding cache invalid: bentuk atau nilai vector salah.")
                return None

            self.last_status = "hit"
            logger.info(
                "Embedding cache hit: model=%s documents=%s dimension=%s",
                model_name,
                len(documents),
                dimension,
            )
            return vectors
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.last_status = "corrupt"
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
            "document_count": len(documents),
            "corpus_fingerprint": corpus_fingerprint(documents),
            "configuration": configuration,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

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
            logger.info("Embedding cache disimpan: %s", self.vector_path)
        finally:
            for temporary_path in (vector_temp, metadata_temp):
                if temporary_path and temporary_path.exists():
                    temporary_path.unlink()
