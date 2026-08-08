import logging
import math
import threading
import time
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class RerankerError(RuntimeError):
    pass


class RerankerMemoryError(RerankerError):
    pass


@dataclass(frozen=True)
class RerankedCandidate:
    candidate: object
    original_rank: int
    reranker_score: float | None
    final_rank: int

    @property
    def document(self):
        return self.candidate.document

    @property
    def original_score(self):
        return self.candidate.score


class RerankerService:
    """Lazy, isolated cross-encoder service; tidak terhubung ke Flask production."""

    def __init__(
        self,
        model_name="BAAI/bge-reranker-v2-m3",
        device="cpu",
        batch_size=1,
        max_length=256,
        normalize=False,
        backend_factory=None,
    ):
        if not model_name or not model_name.strip():
            raise ValueError("Nama model reranker tidak boleh kosong.")
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError("Device reranker harus cpu atau CUDA yang valid.")
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("Batch size dan max length harus lebih besar dari nol.")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self._backend_factory = backend_factory or self._default_backend_factory
        self._backend = None
        self._load_lock = threading.Lock()
        self.model_load_count = 0
        self.model_load_seconds = None
        self.score_call_count = 0

    @property
    def is_loaded(self):
        return self._backend is not None

    @property
    def backend(self):
        return self._load_backend()

    def _default_backend_factory(self):
        from FlagEmbedding import FlagReranker

        # FlagReranker 1.4.0 mendukung use_fp16, bukan BF16 wrapper resmi.
        # FP16 sengaja false pada CPU.
        return FlagReranker(
            self.model_name,
            use_fp16=False,
            devices=self.device,
            batch_size=self.batch_size,
            max_length=self.max_length,
            normalize=self.normalize,
        )

    def _load_backend(self):
        if self._backend is not None:
            return self._backend
        with self._load_lock:
            if self._backend is not None:
                return self._backend
            started_at = time.perf_counter()
            try:
                self._backend = self._backend_factory()
            except Exception as exc:
                if "out of memory" in str(exc).lower():
                    raise RerankerMemoryError("RAM tidak cukup untuk memuat reranker.") from exc
                raise RerankerError(
                    f"Model reranker {self.model_name} gagal dimuat ({type(exc).__name__})."
                ) from exc
            self.model_load_seconds = time.perf_counter() - started_at
            self.model_load_count += 1
            logger.info(
                "Reranker experimental dimuat: model=%s device=%s fp16=false.",
                self.model_name,
                self.device,
            )
            return self._backend

    def score_pairs(self, pairs, normalize=None):
        normalized_pairs = []
        for pair in pairs:
            if len(pair) != 2 or not all(isinstance(value, str) for value in pair):
                raise ValueError("Setiap reranker pair harus berisi query dan passage string.")
            if not pair[0].strip() or not pair[1].strip():
                raise ValueError("Query dan passage reranker tidak boleh kosong.")
            normalized_pairs.append((pair[0], pair[1]))
        if not normalized_pairs:
            return []
        try:
            scores = self.backend.compute_score(
                normalized_pairs,
                batch_size=self.batch_size,
                max_length=self.max_length,
                normalize=self.normalize if normalize is None else normalize,
            )
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                raise RerankerMemoryError("RAM tidak cukup saat reranker inference.") from exc
            raise RerankerError(
                f"Reranker inference gagal ({type(exc).__name__})."
            ) from exc
        self.score_call_count += 1
        if isinstance(scores, (int, float)):
            scores = [scores]
        result = [float(score) for score in scores]
        if len(result) != len(normalized_pairs):
            raise RerankerError("Jumlah score reranker tidak cocok dengan jumlah pair.")
        if not all(math.isfinite(score) for score in result):
            raise RerankerError("Reranker menghasilkan score NaN atau Inf.")
        return result

    def rerank(self, query, candidates, top_k=5):
        if not query or not query.strip():
            raise ValueError("Query reranker tidak boleh kosong.")
        if top_k <= 0:
            raise ValueError("Top-k reranker harus lebih besar dari nol.")
        candidates = list(candidates)
        if not candidates:
            return []
        if len(candidates) == 1:
            return [RerankedCandidate(candidates[0], 1, None, 1)]
        scores = self.score_pairs(
            [(query, candidate.document.content) for candidate in candidates]
        )
        ranked = sorted(
            zip(candidates, scores, range(1, len(candidates) + 1)),
            key=lambda item: (-item[1], item[2], item[0].document.id),
        )[:top_k]
        return [
            RerankedCandidate(candidate, original_rank, score, final_rank)
            for final_rank, (candidate, score, original_rank) in enumerate(ranked, 1)
        ]
