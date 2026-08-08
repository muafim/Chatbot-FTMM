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
DEFAULT_CHUNKING_ENABLED = True
DEFAULT_CHUNK_MAX_TOKENS = 200
DEFAULT_CHUNK_OVERLAP_TOKENS = 24
DEFAULT_RETRIEVAL_MAX_CHUNKS_PER_PARENT = 2
DEFAULT_RETRIEVAL_CANDIDATE_MULTIPLIER = 10
DEFAULT_INTENT_ROUTING_ENABLED = True
DEFAULT_RETRIEVAL_MODE = "dense"
DEFAULT_HYBRID_RRF_K = 60
DEFAULT_HYBRID_CANDIDATE_K = 30
DEFAULT_HYBRID_DENSE_WEIGHT = 1.0
DEFAULT_HYBRID_SPARSE_WEIGHT = 1.0
DEFAULT_RERANKER_ENABLED = False
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_DEVICE = "cpu"
DEFAULT_RERANKER_BATCH_SIZE = 1
DEFAULT_RERANKER_MAX_LENGTH = 256
DEFAULT_RERANKER_CANDIDATE_SOURCE = "union"
DEFAULT_RERANKER_CANDIDATE_K = 10
DEFAULT_RERANKER_TOP_K = 5
DEFAULT_LLM_MAX_OUTPUT_TOKENS = 300
DEFAULT_LLM_TIMEOUT_SECONDS = 30.0
DEFAULT_LLM_MAX_RETRIES = 1
DEFAULT_CITATION_CORRECTION_RETRIES = 1
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_LLM_CACHE_ENABLED = True
DEFAULT_LLM_CACHE_DIRECTORY = "cache/answers"
DEFAULT_LLM_DAILY_CALL_LIMIT = 40
DEFAULT_LLM_MAX_CONTEXT_SOURCES = 3
DEFAULT_PDF_KNOWLEDGE_ENABLED = True
DEFAULT_PDF_SOURCE_DIRECTORY = "data/pdfs"
DEFAULT_PDF_MANIFEST_PATH = "cache/knowledge/pdf_manifest.json"
DEFAULT_PDF_MAX_FILE_SIZE_MB = 50
DEFAULT_KNOWLEDGE_REGISTRY_PATH = "cache/knowledge/registry.json"
DEFAULT_KNOWLEDGE_INCREMENTAL_INDEX = True
CHUNKING_VERSION = "stage4-lecturer-intent-v1"


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
    retrieval_mode = os.getenv(
        "RETRIEVAL_MODE", DEFAULT_RETRIEVAL_MODE
    ).strip().lower()
    if retrieval_mode not in {"dense", "sparse", "hybrid"}:
        raise ValueError("RETRIEVAL_MODE harus dense, sparse, atau hybrid.")
    return {
        "top_k": top_k,
        "min_score": min_score,
        "debug": debug,
        "max_chunks_per_parent": int(
            os.getenv(
                "RETRIEVAL_MAX_CHUNKS_PER_PARENT",
                str(DEFAULT_RETRIEVAL_MAX_CHUNKS_PER_PARENT),
            )
        ),
        "candidate_multiplier": int(
            os.getenv(
                "RETRIEVAL_CANDIDATE_MULTIPLIER",
                str(DEFAULT_RETRIEVAL_CANDIDATE_MULTIPLIER),
            )
        ),
        "intent_routing_enabled": _get_bool(
            "INTENT_ROUTING_ENABLED", DEFAULT_INTENT_ROUTING_ENABLED
        ),
        "retrieval_mode": retrieval_mode,
        "hybrid_rrf_k": int(
            os.getenv("HYBRID_RRF_K", str(DEFAULT_HYBRID_RRF_K))
        ),
        "hybrid_candidate_k": int(
            os.getenv("HYBRID_CANDIDATE_K", str(DEFAULT_HYBRID_CANDIDATE_K))
        ),
        "hybrid_dense_weight": float(
            os.getenv("HYBRID_DENSE_WEIGHT", str(DEFAULT_HYBRID_DENSE_WEIGHT))
        ),
        "hybrid_sparse_weight": float(
            os.getenv("HYBRID_SPARSE_WEIGHT", str(DEFAULT_HYBRID_SPARSE_WEIGHT))
        ),
    }


def get_chunking_settings():
    max_tokens = int(os.getenv("CHUNK_MAX_TOKENS", str(DEFAULT_CHUNK_MAX_TOKENS)))
    overlap = int(
        os.getenv("CHUNK_OVERLAP_TOKENS", str(DEFAULT_CHUNK_OVERLAP_TOKENS))
    )
    if max_tokens <= 0:
        raise ValueError("CHUNK_MAX_TOKENS harus lebih besar dari nol.")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError(
            "CHUNK_OVERLAP_TOKENS harus non-negatif dan lebih kecil dari max tokens."
        )
    return {
        "enabled": _get_bool("CHUNKING_ENABLED", DEFAULT_CHUNKING_ENABLED),
        "max_tokens": max_tokens,
        "overlap_tokens": overlap,
        "chunking_version": CHUNKING_VERSION,
        "tokenizer_name": os.getenv("BGE_M3_MODEL", DEFAULT_BGE_M3_MODEL),
        "tokenizer_path": os.getenv("BGE_M3_LOCAL_PATH") or None,
    }


def get_llm_settings():
    return {
        "provider": "openai",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        "max_output_tokens": int(
            os.getenv("LLM_MAX_OUTPUT_TOKENS", str(DEFAULT_LLM_MAX_OUTPUT_TOKENS))
        ),
        "timeout_seconds": float(
            os.getenv("LLM_TIMEOUT_SECONDS", str(DEFAULT_LLM_TIMEOUT_SECONDS))
        ),
        "max_retries": int(os.getenv("LLM_MAX_RETRIES", str(DEFAULT_LLM_MAX_RETRIES))),
        "debug": _get_bool("RAG_DEBUG", False),
    }


def get_llm_budgeting_settings(base_directory=None):
    policy = os.getenv("LLM_CALL_POLICY", "smart").strip().lower()
    if policy != "smart":
        raise ValueError("LLM_CALL_POLICY Tahap 7B harus smart.")
    cache_directory = Path(
        os.getenv("LLM_CACHE_DIRECTORY", DEFAULT_LLM_CACHE_DIRECTORY)
    )
    if base_directory is not None and not cache_directory.is_absolute():
        cache_directory = Path(base_directory) / cache_directory
    budget_path = Path(os.getenv("LLM_DAILY_COUNTER_PATH", "cache/llm_daily_budget.json"))
    if base_directory is not None and not budget_path.is_absolute():
        budget_path = Path(base_directory) / budget_path
    max_sources = int(
        os.getenv("LLM_MAX_CONTEXT_SOURCES", str(DEFAULT_LLM_MAX_CONTEXT_SOURCES))
    )
    if max_sources <= 0:
        raise ValueError("LLM_MAX_CONTEXT_SOURCES harus lebih besar dari nol.")
    return {
        "call_policy": policy,
        "cache_enabled": _get_bool("LLM_CACHE_ENABLED", DEFAULT_LLM_CACHE_ENABLED),
        "cache_directory": cache_directory,
        "daily_counter_path": budget_path,
        "daily_call_limit": int(
            os.getenv("LLM_DAILY_CALL_LIMIT", str(DEFAULT_LLM_DAILY_CALL_LIMIT))
        ),
        "max_context_sources": max_sources,
    }


def get_grounding_settings():
    retries = int(
        os.getenv(
            "CITATION_CORRECTION_RETRIES",
            str(DEFAULT_CITATION_CORRECTION_RETRIES),
        )
    )
    if retries not in {0, 1}:
        raise ValueError("CITATION_CORRECTION_RETRIES harus 0 atau 1.")
    return {"citation_correction_retries": retries}


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


def get_pdf_knowledge_settings(base_directory=None):
    """Konfigurasi ingestion PDF lokal; tidak memuat file atau model."""
    source_directory = Path(
        os.getenv("PDF_SOURCE_DIRECTORY", DEFAULT_PDF_SOURCE_DIRECTORY)
    )
    manifest_path = Path(
        os.getenv("PDF_MANIFEST_PATH", DEFAULT_PDF_MANIFEST_PATH)
    )
    if base_directory is not None:
        if not source_directory.is_absolute():
            source_directory = Path(base_directory) / source_directory
        if not manifest_path.is_absolute():
            manifest_path = Path(base_directory) / manifest_path
    max_file_size_mb = int(
        os.getenv("PDF_MAX_FILE_SIZE_MB", str(DEFAULT_PDF_MAX_FILE_SIZE_MB))
    )
    if max_file_size_mb <= 0:
        raise ValueError("PDF_MAX_FILE_SIZE_MB harus lebih besar dari nol.")
    return {
        "enabled": _get_bool("PDF_KNOWLEDGE_ENABLED", DEFAULT_PDF_KNOWLEDGE_ENABLED),
        "directory": source_directory,
        "manifest_path": manifest_path,
        "max_file_size_mb": max_file_size_mb,
    }


def get_knowledge_index_settings(base_directory=None):
    registry_path = Path(
        os.getenv("KNOWLEDGE_REGISTRY_PATH", DEFAULT_KNOWLEDGE_REGISTRY_PATH)
    )
    if base_directory is not None and not registry_path.is_absolute():
        registry_path = Path(base_directory) / registry_path
    return {
        "registry_path": registry_path,
        "incremental_index_enabled": _get_bool(
            "KNOWLEDGE_INCREMENTAL_INDEX", DEFAULT_KNOWLEDGE_INCREMENTAL_INDEX
        ),
    }


def get_embedding_allow_fallback():
    """Fallback tidak pernah implicit; default selalu false."""
    return _get_bool("EMBEDDING_ALLOW_FALLBACK", False)


def get_reranker_settings():
    """Konfigurasi eksperimen reranker; tidak mengaktifkan integrasi Flask."""
    device = os.getenv("RERANKER_DEVICE", DEFAULT_RERANKER_DEVICE).strip().lower()
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("RERANKER_DEVICE harus cpu atau device CUDA yang valid.")
    candidate_source = os.getenv(
        "RERANKER_CANDIDATE_SOURCE", DEFAULT_RERANKER_CANDIDATE_SOURCE
    ).strip().lower()
    if candidate_source not in {"dense", "hybrid", "union"}:
        raise ValueError(
            "RERANKER_CANDIDATE_SOURCE harus dense, hybrid, atau union."
        )
    settings = {
        "enabled": _get_bool("RERANKER_ENABLED", DEFAULT_RERANKER_ENABLED),
        "model_name": os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
        "device": device,
        "batch_size": int(
            os.getenv("RERANKER_BATCH_SIZE", str(DEFAULT_RERANKER_BATCH_SIZE))
        ),
        "max_length": int(
            os.getenv("RERANKER_MAX_LENGTH", str(DEFAULT_RERANKER_MAX_LENGTH))
        ),
        "candidate_source": candidate_source,
        "candidate_k": int(
            os.getenv("RERANKER_CANDIDATE_K", str(DEFAULT_RERANKER_CANDIDATE_K))
        ),
        "top_k": int(os.getenv("RERANKER_TOP_K", str(DEFAULT_RERANKER_TOP_K))),
    }
    if any(settings[key] <= 0 for key in ("batch_size", "max_length", "candidate_k", "top_k")):
        raise ValueError("Konfigurasi numerik reranker harus lebih besar dari nol.")
    return settings
