import os
from pathlib import Path


DEFAULT_EMBEDDING_BACKEND = "bge-m3"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_RETRIEVAL_TOP_K = 10
DEFAULT_RETRIEVAL_MIN_SCORE = 0.5
DEFAULT_BGE_M3_MODEL = "BAAI/bge-m3"
DEFAULT_BGE_M3_DEVICE = "cpu"
DEFAULT_BGE_M3_PRECISION = "bf16"
DEFAULT_BGE_M3_BATCH_SIZE = 1
DEFAULT_BGE_M3_MAX_LENGTH = 256
DEFAULT_EMBEDDING_CACHE_ENABLED = True
DEFAULT_EMBEDDING_CACHE_DIRECTORY = "cache/embeddings"


def _get_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_embedding_backend():
    backend = os.getenv("EMBEDDING_BACKEND", DEFAULT_EMBEDDING_BACKEND).strip().lower()
    if backend not in {"baseline", "bge-m3"}:
        raise ValueError("EMBEDDING_BACKEND harus bernilai baseline atau bge-m3.")
    return backend


def get_embedding_settings():
    """Baca konfigurasi embedding tanpa memuat atau menginisialisasi model."""
    return {
        "model_name": os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "device": os.getenv("EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE),
    }


def get_retrieval_settings():
    """Baca konfigurasi retrieval dan pertahankan nilai baseline Tahap 0."""
    top_k = int(os.getenv("RETRIEVAL_TOP_K", str(DEFAULT_RETRIEVAL_TOP_K)))
    min_score_value = os.getenv(
        "RETRIEVAL_MIN_SCORE", str(DEFAULT_RETRIEVAL_MIN_SCORE)
    ).strip()
    min_score = float(min_score_value) if min_score_value else None
    debug = os.getenv("RAG_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    return {"top_k": top_k, "min_score": min_score, "debug": debug}


def get_llm_settings():
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }


def get_bge_m3_settings():
    """Konfigurasi backend BGE-M3 dense."""
    return {
        "model_name": os.getenv("BGE_M3_MODEL", DEFAULT_BGE_M3_MODEL),
        "model_path": os.getenv("BGE_M3_LOCAL_PATH") or None,
        "device": os.getenv("BGE_M3_DEVICE", DEFAULT_BGE_M3_DEVICE),
        "precision": os.getenv("BGE_M3_PRECISION", DEFAULT_BGE_M3_PRECISION),
        "batch_size": int(
            os.getenv("BGE_M3_BATCH_SIZE", str(DEFAULT_BGE_M3_BATCH_SIZE))
        ),
        "max_length": int(
            os.getenv("BGE_M3_MAX_LENGTH", str(DEFAULT_BGE_M3_MAX_LENGTH))
        ),
    }


def get_embedding_cache_settings(base_directory=None):
    configured_path = Path(
        os.getenv(
            "EMBEDDING_CACHE_DIRECTORY", DEFAULT_EMBEDDING_CACHE_DIRECTORY
        )
    )
    if base_directory is not None and not configured_path.is_absolute():
        configured_path = Path(base_directory) / configured_path
    return {
        "enabled": _get_bool(
            "EMBEDDING_CACHE_ENABLED", DEFAULT_EMBEDDING_CACHE_ENABLED
        ),
        "directory": configured_path,
    }


def get_embedding_allow_fallback():
    """Fallback tidak pernah implicit; default selalu false."""
    return _get_bool("EMBEDDING_ALLOW_FALLBACK", False)
