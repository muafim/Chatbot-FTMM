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
    get_embedding_allow_fallback,
    get_embedding_backend,
    get_embedding_cache_settings,
    get_embedding_settings,
    get_llm_settings,
    get_retrieval_settings,
)
from indexes.in_memory_vector_index import InMemoryVectorIndex
from repositories.knowledge_repository import KnowledgeRepository
from retrievers.dense_retriever import DenseRetriever
from services.chat_service import ChatService
from services.context_builder import ContextBuilder
from services.embedding_cache import DocumentEmbeddingCache
from services.embedding_factory import create_embedding_service
from services.llm_service import LLMService


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
knowledge_repository = KnowledgeRepository(DATA_DIR)
documents = knowledge_repository.get_documents()
dataset_errors = knowledge_repository.errors

embedding_backend = get_embedding_backend()
embedding_service = create_embedding_service(
    backend=embedding_backend,
    baseline_settings=get_embedding_settings(),
    bge_m3_settings=get_bge_m3_settings(),
    allow_fallback=get_embedding_allow_fallback(),
)
cache_settings = get_embedding_cache_settings(BASE_DIR)
embedding_cache = DocumentEmbeddingCache(
    directory=cache_settings["directory"],
    cache_key=embedding_backend.replace("-", "_"),
    enabled=cache_settings["enabled"],
)
vector_index = InMemoryVectorIndex()
retriever = DenseRetriever(
    repository=knowledge_repository,
    embedding_service=embedding_service,
    vector_index=vector_index,
    embedding_cache=embedding_cache,
    **get_retrieval_settings(),
)
context_builder = ContextBuilder()
llm_service = LLMService(**get_llm_settings())
chat_service = ChatService(
    retriever=retriever,
    context_builder=context_builder,
    llm_service=llm_service,
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
                answer = result.answer
            except Exception as exc:
                logger.error("Chat processing gagal. Tipe error: %s.", type(exc).__name__)
                answer = (
                    "Pertanyaan gagal diproses. Periksa dataset, model embedding, dan "
                    "konfigurasi aplikasi."
                )

            session["chat_history"].append({"user": query, "bot": answer})
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
