from dataclasses import dataclass, field
from typing import Any, Optional


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
class RetrievedDocument:
    document: Document
    score: Optional[float] = None


@dataclass(frozen=True)
class ChatResult:
    answer: str
    retrieved_documents: list[RetrievedDocument] = field(default_factory=list)

    @property
    def retrieval_scores(self):
        return [item.score for item in self.retrieved_documents]
