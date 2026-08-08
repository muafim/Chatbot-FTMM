import logging
import time

import numpy as np

from domain.models import HybridEmbedding, HybridEmbeddingBatch, SparseVector


logger = logging.getLogger(__name__)

BGE_M3_PYTORCH_FILES = [
    "config.json",
    "pytorch_model.bin",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "colbert_linear.pt",
    "sparse_linear.pt",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "modules.json",
    "1_Pooling/config.json",
]


class BGEM3EmbeddingError(RuntimeError):
    pass


class BGEM3EmbeddingService:
    """Adapter BGE-M3 dense/sparse dengan satu lazy-loaded model instance."""

    def __init__(
        self,
        model_name="BAAI/bge-m3",
        model_path=None,
        device="auto",
        precision="auto",
        batch_size=2,
        max_length=512,
    ):
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("BGE-M3 device harus salah satu dari: auto, cpu, cuda.")
        if precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError(
                "BGE-M3 precision harus salah satu dari: auto, fp32, fp16, bf16."
            )
        if batch_size <= 0:
            raise ValueError("BGE-M3 batch size harus lebih besar dari nol.")
        if max_length <= 0:
            raise ValueError("BGE-M3 max length harus lebih besar dari nol.")

        self.model_name = model_name
        self.model_path = model_path
        self.requested_device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = self._resolve_device(device)
        self.precision = self._resolve_precision(precision)
        self.use_fp16 = self.precision == "fp16"
        self.use_bf16 = self.precision == "bf16"
        self.normalize_embeddings = True
        self.embedding_dimension = None
        self._model = None
        self.model_load_seconds = None
        self.model_load_count = 0
        self.hybrid_query_encode_count = 0
        self.hybrid_document_encode_count = 0

    @property
    def is_loaded(self):
        return self._model is not None

    @property
    def cache_configuration(self):
        return {
            "backend": "bge-m3",
            "max_length": self.max_length,
            "precision": self.precision,
            "normalized": self.normalize_embeddings,
        }

    @property
    def model_identity(self):
        return {
            "embedding_model": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "precision": self.precision,
            "normalized": self.normalize_embeddings,
        }

    def _resolve_device(self, requested_device):
        import torch

        cuda_available = torch.cuda.is_available()
        if requested_device == "cuda" and not cuda_available:
            raise BGEM3EmbeddingError(
                "BGE-M3 diminta menggunakan CUDA, tetapi CUDA tidak tersedia."
            )
        if requested_device == "auto":
            return "cuda" if cuda_available else "cpu"
        return requested_device

    def _resolve_precision(self, requested_precision):
        if requested_precision == "auto":
            return "fp16" if self.device == "cuda" else "fp32"
        if requested_precision == "fp16" and self.device != "cuda":
            raise BGEM3EmbeddingError(
                "BGE-M3 FP16 hanya diizinkan ketika menggunakan CUDA."
            )
        return requested_precision

    def load_model(self):
        if self._model is not None:
            return self._model

        try:
            started_at = time.perf_counter()
            from FlagEmbedding import BGEM3FlagModel

            model_source = self.model_path
            if not model_source:
                from huggingface_hub import snapshot_download

                model_source = snapshot_download(
                    self.model_name,
                    allow_patterns=BGE_M3_PYTORCH_FILES,
                )
            self._model = BGEM3FlagModel(
                model_source,
                devices=self.device,
                use_fp16=self.use_fp16,
                use_bf16=self.use_bf16,
                normalize_embeddings=self.normalize_embeddings,
                batch_size=self.batch_size,
                query_max_length=self.max_length,
                passage_max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            logger.info(
                "BGE-M3 model %s dimuat pada %s dengan %s.",
                self.model_name,
                self.device,
                self.precision,
            )
            self.model_load_seconds = time.perf_counter() - started_at
            self.model_load_count += 1
            return self._model
        except Exception as exc:
            self._model = None
            raise self._friendly_error("gagal di-download atau dimuat", exc) from exc

    def encode_query(self, query):
        if not query or not query.strip():
            raise ValueError("Query BGE-M3 tidak boleh kosong.")
        try:
            output = self.load_model().encode_queries(
                [query],
                batch_size=1,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vectors = self._validate_dense_vectors(output, expected_count=1)
            return vectors[0]
        except (ValueError, BGEM3EmbeddingError):
            raise
        except Exception as exc:
            raise self._friendly_error("gagal meng-encode query", exc) from exc

    def encode_documents(self, texts):
        document_texts = list(texts)
        if not document_texts:
            raise ValueError("Daftar document BGE-M3 tidak boleh kosong.")
        if any(not text or not str(text).strip() for text in document_texts):
            raise ValueError("Document BGE-M3 tidak boleh berisi teks kosong.")
        try:
            output = self.load_model().encode_corpus(
                document_texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            return self._validate_dense_vectors(
                output, expected_count=len(document_texts)
            )
        except (ValueError, BGEM3EmbeddingError):
            raise
        except Exception as exc:
            raise self._friendly_error("gagal meng-encode document", exc) from exc

    def encode_query_sparse(self, query):
        if not query or not query.strip():
            raise ValueError("Query BGE-M3 tidak boleh kosong.")
        output = self._encode_native([query], is_query=True, dense=False, sparse=True)
        return self._validate_sparse_vectors(output, 1)[0]

    def encode_documents_sparse(self, texts):
        document_texts = self._validate_document_texts(texts)
        output = self._encode_native(
            document_texts, is_query=False, dense=False, sparse=True
        )
        return self._validate_sparse_vectors(output, len(document_texts))

    def encode_query_hybrid(self, query):
        if not query or not query.strip():
            raise ValueError("Query BGE-M3 tidak boleh kosong.")
        output = self._encode_native([query], is_query=True, dense=True, sparse=True)
        dense = self._validate_dense_vectors(output, 1)[0]
        sparse = self._validate_sparse_vectors(output, 1)[0]
        self.hybrid_query_encode_count += 1
        return HybridEmbedding(dense=dense, sparse=sparse)

    def encode_documents_hybrid(self, texts):
        document_texts = self._validate_document_texts(texts)
        output = self._encode_native(
            document_texts, is_query=False, dense=True, sparse=True
        )
        dense = self._validate_dense_vectors(output, len(document_texts))
        sparse = self._validate_sparse_vectors(output, len(document_texts))
        self.hybrid_document_encode_count += 1
        return HybridEmbeddingBatch(dense=dense, sparse=tuple(sparse))

    def _encode_native(self, texts, is_query, dense, sparse):
        try:
            method = (
                self.load_model().encode_queries
                if is_query else self.load_model().encode_corpus
            )
            return method(
                texts,
                batch_size=1 if is_query else self.batch_size,
                max_length=self.max_length,
                return_dense=dense,
                return_sparse=sparse,
                return_colbert_vecs=False,
            )
        except (ValueError, BGEM3EmbeddingError):
            raise
        except Exception as exc:
            subject = "query" if is_query else "document"
            raise self._friendly_error(f"gagal meng-encode {subject}", exc) from exc

    @staticmethod
    def _validate_document_texts(texts):
        document_texts = list(texts)
        if not document_texts:
            raise ValueError("Daftar document BGE-M3 tidak boleh kosong.")
        if any(not text or not str(text).strip() for text in document_texts):
            raise ValueError("Document BGE-M3 tidak boleh berisi teks kosong.")
        return document_texts

    def _validate_dense_vectors(self, output, expected_count):
        if not isinstance(output, dict) or "dense_vecs" not in output:
            raise BGEM3EmbeddingError("BGE-M3 tidak mengembalikan dense_vecs.")

        vectors = np.asarray(output["dense_vecs"])
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[0] != expected_count:
            raise BGEM3EmbeddingError(
                "Bentuk dense vector BGE-M3 tidak sesuai jumlah input."
            )
        if vectors.size == 0 or vectors.shape[1] == 0:
            raise BGEM3EmbeddingError("Dense vector BGE-M3 kosong.")
        if not np.isfinite(vectors).all():
            raise BGEM3EmbeddingError("Dense vector BGE-M3 mengandung NaN atau Inf.")
        if np.allclose(vectors, 0):
            raise BGEM3EmbeddingError("Dense vector BGE-M3 seluruhnya bernilai nol.")

        dimension = int(vectors.shape[1])
        if self.embedding_dimension is None:
            self.embedding_dimension = dimension
        elif self.embedding_dimension != dimension:
            raise BGEM3EmbeddingError(
                "Dimensi dense vector BGE-M3 berubah dalam instance yang sama."
            )
        return vectors

    @staticmethod
    def _validate_sparse_vectors(output, expected_count):
        if not isinstance(output, dict) or "lexical_weights" not in output:
            raise BGEM3EmbeddingError("BGE-M3 tidak mengembalikan lexical_weights.")
        raw_vectors = output["lexical_weights"]
        if not isinstance(raw_vectors, (list, tuple)) or len(raw_vectors) != expected_count:
            raise BGEM3EmbeddingError(
                "Jumlah sparse vector BGE-M3 tidak sesuai jumlah input."
            )
        vectors = []
        for raw_vector in raw_vectors:
            if not hasattr(raw_vector, "items"):
                raise BGEM3EmbeddingError("Sparse vector BGE-M3 bukan mapping.")
            try:
                vector = SparseVector.from_mapping(raw_vector)
            except ValueError as exc:
                raise BGEM3EmbeddingError(str(exc)) from exc
            if not vector.values:
                raise BGEM3EmbeddingError("Sparse vector BGE-M3 kosong.")
            vectors.append(vector)
        return vectors

    def _friendly_error(self, action, exc):
        error_text = str(exc).lower()
        if "out of memory" in error_text or "allocate memory" in error_text:
            message = (
                f"BGE-M3 {action} karena kehabisan memory. Turunkan batch size atau "
                "max length."
            )
        else:
            message = f"BGE-M3 {action}: {type(exc).__name__}."
        logger.error(message)
        return BGEM3EmbeddingError(message)
