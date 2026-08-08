import logging
import os
import secrets
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, session, url_for
from flask_session import Session

from config import (
    get_bge_m3_settings,
    get_chunking_settings,
    get_embedding_allow_fallback,
    get_embedding_backend,
    get_embedding_cache_settings,
    get_embedding_settings,
    get_grounding_settings,
    get_llm_settings,
    get_llm_budgeting_settings,
    get_knowledge_index_settings,
    get_pdf_knowledge_settings,
    get_retrieval_settings,
)
from indexes.in_memory_vector_index import InMemoryVectorIndex
from indexes.in_memory_sparse_index import InMemorySparseIndex
from repositories.knowledge_repository import KnowledgeRepository
from repositories.pdf_knowledge_repository import PDFKnowledgeRepository
from retrievers.dense_retriever import DenseRetriever
from services.chat_service import ChatService
from services.context_builder import ContextBuilder
from services.document_chunker import DocumentChunker
from services.embedding_cache import DocumentEmbeddingCache, corpus_fingerprint
from services.embedding_factory import create_embedding_service
from services.llm_service import LLMService
from services.openai_llm_service import (
    GROUNDED_SCHEMA_VERSION,
    GROUNDING_PROMPT_VERSION,
    OpenAILLMProvider,
)
from services.answer_cache import GroundedAnswerCache
from services.llm_budget import DailyLLMBudgetGuard
from services.llm_telemetry import LLMTelemetry
from services.knowledge_registry import KnowledgeRegistry
from services.query_intent_router import QueryIntentRouter


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SESSION_DIR = BASE_DIR / "flask_session"
EVALUATION_DATA_PATH = DATA_DIR / "data_validasi4.csv"

load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

SESSION_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
secret_key = os.getenv("FLASK_SECRET_KEY")
if not secret_key:
    secret_key = secrets.token_hex(32)
    logger.warning(
        "FLASK_SECRET_KEY belum dikonfigurasi; menggunakan kunci sementara untuk sesi ini."
    )

app.config.update(
    SECRET_KEY=secret_key,
    SESSION_TYPE="filesystem",
    SESSION_FILE_DIR=str(SESSION_DIR),
    SESSION_PERMANENT=False,
    SESSION_USE_SIGNER=True,
)
Session(app)

# Application components. Dataset dimuat satu kali; model dan vector index tetap lazy.
pdf_knowledge_repository = PDFKnowledgeRepository(**get_pdf_knowledge_settings(BASE_DIR))
knowledge_index_settings = get_knowledge_index_settings(BASE_DIR)
knowledge_registry = KnowledgeRegistry(knowledge_index_settings["registry_path"])
knowledge_repository = KnowledgeRepository(
    DATA_DIR,
    pdf_repository=pdf_knowledge_repository,
    knowledge_registry=knowledge_registry,
)
documents = knowledge_repository.get_documents()
dataset_errors = knowledge_repository.errors + pdf_knowledge_repository.warnings

embedding_backend = get_embedding_backend()
embedding_service = create_embedding_service(
    backend=embedding_backend,
    baseline_settings=get_embedding_settings(),
    bge_m3_settings=get_bge_m3_settings(),
    allow_fallback=get_embedding_allow_fallback(),
)
document_chunker = DocumentChunker(**get_chunking_settings())
cache_settings = get_embedding_cache_settings(BASE_DIR)
retrieval_settings = get_retrieval_settings()
cache_key = embedding_backend.replace("-", "_")
if retrieval_settings["retrieval_mode"] in {"sparse", "hybrid"}:
    cache_key = f"{cache_key}_hybrid"
embedding_cache = DocumentEmbeddingCache(
    directory=cache_settings["directory"],
    cache_key=cache_key,
    enabled=cache_settings["enabled"],
)
vector_index = InMemoryVectorIndex()
sparse_index = InMemorySparseIndex()
query_intent_router = QueryIntentRouter()
retriever = DenseRetriever(
    repository=knowledge_repository,
    embedding_service=embedding_service,
    vector_index=vector_index,
    sparse_index=sparse_index,
    embedding_cache=embedding_cache,
    document_chunker=document_chunker,
    intent_router=query_intent_router,
    incremental_index_enabled=knowledge_index_settings["incremental_index_enabled"],
    **retrieval_settings,
)
context_builder = ContextBuilder()
llm_settings = get_llm_settings()
budgeting_settings = get_llm_budgeting_settings(BASE_DIR)
llm_telemetry = LLMTelemetry()
llm_budget_guard = DailyLLMBudgetGuard(
    path=budgeting_settings["daily_counter_path"],
    daily_call_limit=budgeting_settings["daily_call_limit"],
)
openai_provider = OpenAILLMProvider(
    api_key=llm_settings["api_key"],
    model_name=llm_settings["model_name"],
    max_output_tokens=llm_settings["max_output_tokens"],
    timeout_seconds=llm_settings["timeout_seconds"],
    max_retries=llm_settings["max_retries"],
    budget_guard=llm_budget_guard,
    telemetry=llm_telemetry,
    debug=llm_settings["debug"],
)
llm_service = LLMService(openai_provider)
answer_cache = GroundedAnswerCache(
    directory=budgeting_settings["cache_directory"],
    corpus_fingerprint=corpus_fingerprint(documents),
    retrieval_signature={
        "retrieval_mode": retrieval_settings["retrieval_mode"],
        "top_k": retrieval_settings["top_k"],
        "intent_routing_enabled": retrieval_settings["intent_routing_enabled"],
        **document_chunker.cache_configuration,
    },
    provider=llm_settings["provider"],
    model=llm_settings["model_name"],
    prompt_version=GROUNDING_PROMPT_VERSION,
    schema_version=GROUNDED_SCHEMA_VERSION,
    enabled=budgeting_settings["cache_enabled"],
)
chat_service = ChatService(
    retriever=retriever,
    context_builder=context_builder,
    llm_service=llm_service,
    answer_cache=answer_cache,
    telemetry=llm_telemetry,
    max_context_sources=budgeting_settings["max_context_sources"],
    debug=llm_settings["debug"],
    **get_grounding_settings(),
)


@app.route("/", methods=["GET", "POST"])
def index():
    current_year = datetime.now().year
    if "chat_history" not in session:
        session["chat_history"] = []

    form_error = None
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if not query:
            form_error = "Pertanyaan tidak boleh kosong."
        else:
            try:
                result = chat_service.answer(
                    question=query,
                    conversation_history=session.get("chat_history", []),
                )
                exchange = {
                    "user": query,
                    "bot": result.answer,
                    "answerability": result.answerability.value,
                    "grounding_status": result.grounding_status.value,
                    "answer_path": result.answer_path.value,
                    "warning": result.warning,
                    "sources": [
                        {
                            "source_id": source.source_id,
                            "chunk_id": source.chunk_id,
                            "parent_document_id": source.parent_document_id,
                            "source_type": source.source_type,
                            "title": source.title,
                            "section": source.section,
                            "url": source.url,
                            "supporting_excerpt": source.supporting_excerpt,
                            "metadata": source.metadata,
                        }
                        for source in result.sources
                    ],
                }
            except Exception as exc:
                logger.error("Chat processing gagal. Tipe error: %s.", type(exc).__name__)
                exchange = {"user": query, "bot": (
                    "Pertanyaan gagal diproses. Periksa dataset, model embedding, dan "
                    "konfigurasi aplikasi."
                ), "sources": [], "grounding_status": "error"}

            session["chat_history"].append(exchange)
            session.modified = True
            return redirect(url_for("index"))

    return render_template(
        "index.html",
        chat_history=session.get("chat_history", []),
        current_year=current_year,
        dataset_errors=dataset_errors,
        form_error=form_error,
    )


@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return redirect(url_for("index"))


@app.route("/evaluate", methods=["GET"])
def evaluate():
    if not EVALUATION_DATA_PATH.exists():
        message = "Dataset evaluasi belum tersedia. Evaluasi model dinonaktifkan sementara."
    else:
        message = (
            "Dataset evaluasi ditemukan, tetapi evaluasi versi lama belum diaktifkan pada "
            "baseline modular ini."
        )
    return render_template(
        "evaluation.html", message=message, current_year=datetime.now().year
    )


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=debug,
    )
