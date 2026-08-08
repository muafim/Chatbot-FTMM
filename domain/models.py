from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Optional


class QueryIntent(str, Enum):
    GENERAL = "general"
    LECTURER = "lecturer"
    COURSE = "course"
    ACADEMIC = "academic"
    STAFF = "staff"
    FTMM = "ftmm"


class IntentConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Answerability(str, Enum):
    ANSWERABLE = "answerable"
    PARTIALLY_ANSWERABLE = "partially_answerable"
    NOT_ANSWERABLE = "not_answerable"


class GroundingStatus(str, Enum):
    GROUNDED = "grounded"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LLM_UNAVAILABLE = "llm_unavailable"
    CITATION_INVALID = "citation_invalid"
    BUDGET_LIMITED = "budget_limited"


class AnswerPath(str, Enum):
    LOCAL = "local"
    CACHE = "cache"
    LLM = "llm"
    NO_ANSWER = "no_answer"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RetrievalPlan:
    query: str
    intent: QueryIntent = QueryIntent.GENERAL
    metadata_filter: Optional[dict[str, Any]] = None
    preferred_chunk_role: Optional[str] = None
    confidence: IntentConfidence = IntentConfidence.LOW
    reason: str = "ambiguous query; dense retrieval over the full corpus"

    def __post_init__(self):
        if not self.query or not self.query.strip():
            raise ValueError("Retrieval plan query tidak boleh kosong.")


@dataclass(frozen=True)
class SparseVector:
    """Sparse lexical weights BGE-M3 dalam urutan token ID deterministic."""

    values: tuple[tuple[str, float], ...]

    @classmethod
    def from_mapping(cls, mapping):
        if mapping is None:
            raise ValueError("Sparse vector tidak boleh None.")
        normalized = []
        for token_id, weight in mapping.items():
            numeric_weight = float(weight)
            if not math.isfinite(numeric_weight):
                raise ValueError("Sparse weight mengandung NaN atau Inf.")
            if numeric_weight != 0.0:
                normalized.append((str(token_id), numeric_weight))
        return cls(tuple(sorted(normalized, key=lambda item: item[0])))

    def as_dict(self):
        return dict(self.values)

    def __len__(self):
        return len(self.values)


@dataclass(frozen=True)
class HybridEmbedding:
    dense: Any
    sparse: SparseVector


@dataclass(frozen=True)
class HybridEmbeddingBatch:
    dense: Any
    sparse: tuple[SparseVector, ...]


@dataclass(frozen=True)
class Document:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id.strip():
            raise ValueError("Document ID tidak boleh kosong.")
        if not self.content.strip():
            raise ValueError("Document content tidak boleh kosong.")


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    parent_document_id: str
    content: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id.strip() or not self.parent_document_id.strip():
            raise ValueError("Chunk ID dan parent document ID tidak boleh kosong.")
        if not self.content.strip():
            raise ValueError("Chunk content tidak boleh kosong.")
        if self.chunk_index < 0:
            raise ValueError("Chunk index tidak boleh negatif.")

    @property
    def chunk_id(self):
        return self.id


@dataclass(frozen=True)
class RetrievedDocument:
    document: Document
    score: Optional[float] = None
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    fusion_score: Optional[float] = None
    final_rank: Optional[int] = None

    @property
    def chunk_id(self):
        return self.document.id

    @property
    def parent_document_id(self):
        return getattr(self.document, "parent_document_id", self.document.id)

    @property
    def content(self):
        return self.document.content

    @property
    def metadata(self):
        return self.document.metadata


@dataclass(frozen=True)
class Citation:
    source_id: str
    chunk_id: str
    parent_document_id: str
    source_type: str
    title: str
    section: Optional[str] = None
    url: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    supporting_excerpt: Optional[str] = None


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    used_sources: list[Citation] = field(default_factory=list)
    has_sufficient_evidence: bool = False
    answerability: Answerability = Answerability.NOT_ANSWERABLE
    warning: Optional[str] = None
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    grounding_status: GroundingStatus = GroundingStatus.INSUFFICIENT_EVIDENCE
    answer_path: AnswerPath = AnswerPath.NO_ANSWER


@dataclass(frozen=True)
class ChatResult:
    answer: str
    retrieved_documents: list[RetrievedDocument] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    sources: list[Citation] = field(default_factory=list)
    answerability: Answerability = Answerability.NOT_ANSWERABLE
    grounding_status: GroundingStatus = GroundingStatus.INSUFFICIENT_EVIDENCE
    warning: Optional[str] = None
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    grounded_answer: Optional[GroundedAnswer] = None
    answer_path: AnswerPath = AnswerPath.NO_ANSWER

    @classmethod
    def from_grounded(cls, grounded_answer, retrieved_documents=None):
        return cls(
            answer=grounded_answer.answer,
            retrieved_documents=list(retrieved_documents or []),
            citations=list(grounded_answer.citations),
            sources=list(grounded_answer.used_sources),
            answerability=grounded_answer.answerability,
            grounding_status=grounded_answer.grounding_status,
            warning=grounded_answer.warning,
            retrieval_metadata=dict(grounded_answer.retrieval_metadata),
            grounded_answer=grounded_answer,
            answer_path=grounded_answer.answer_path,
        )

    @property
    def retrieval_results(self):
        return self.retrieved_documents

    @property
    def retrieval_scores(self):
        return [item.score for item in self.retrieved_documents]
