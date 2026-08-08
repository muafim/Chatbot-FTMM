import logging

from services.bge_m3_embedding_service import BGEM3EmbeddingService
from services.embedding_service import EmbeddingService


logger = logging.getLogger(__name__)


def create_embedding_service(
    backend,
    baseline_settings,
    bge_m3_settings,
    allow_fallback=False,
):
    """Pilih tepat satu backend tanpa memuat model yang tidak dipilih."""
    normalized_backend = str(backend).strip().lower()
    if normalized_backend == "baseline":
        service = EmbeddingService(**baseline_settings)
    elif normalized_backend == "bge-m3":
        try:
            service = BGEM3EmbeddingService(**bge_m3_settings)
        except Exception:
            if not allow_fallback:
                raise
            logger.warning(
                "BGE-M3 gagal diinisialisasi; fallback explicit ke baseline digunakan."
            )
            service = EmbeddingService(**baseline_settings)
            normalized_backend = "baseline"
    else:
        raise ValueError("Embedding backend harus bernilai baseline atau bge-m3.")

    service.backend_name = normalized_backend
    logger.info(
        "Embedding backend dipilih: backend=%s model=%s",
        normalized_backend,
        service.model_name,
    )
    return service
