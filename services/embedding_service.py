import logging
import time


logger = logging.getLogger(__name__)


class EmbeddingService:
    """Adapter sederhana untuk model embedding berbasis SentenceTransformer."""

    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self.model_load_seconds = None

    @property
    def is_loaded(self):
        return self._model is not None

    @property
    def cache_configuration(self):
        return {"backend": "baseline"}

    def load_model(self):
        """Lazy-load satu instance model dan gunakan kembali untuk request berikutnya."""
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        started_at = time.perf_counter()
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=True,
            )
        except Exception:
            logger.info("Model embedding belum ada di cache lokal; mencoba mengunduhnya.")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

        self.model_load_seconds = time.perf_counter() - started_at

        return self._model

    def encode_documents(self, texts):
        """Encode representasi teks dokumen menjadi sekumpulan vector."""
        document_texts = list(texts)
        if not document_texts:
            raise ValueError("Tidak ada teks dokumen yang dapat di-encode.")
        return self.load_model().encode(document_texts)

    def encode_query(self, query):
        """Encode satu query; dipisahkan agar preprocessing query dapat diubah nanti."""
        if not query or not query.strip():
            raise ValueError("Query tidak boleh kosong.")
        return self.load_model().encode([query])[0]
