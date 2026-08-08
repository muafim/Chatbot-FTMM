import re
from dataclasses import dataclass
from enum import Enum

from domain.models import Answerability, QueryIntent


class LLMDecision(str, Enum):
    USE_LOCAL = "use_local"
    USE_LLM = "use_llm"
    NO_ANSWER = "no_answer"


@dataclass(frozen=True)
class LLMDecisionResult:
    decision: LLMDecision
    reason: str


class LLMDecisionService:
    """Deterministic smart gate. Uncertainty always falls back to LLM."""

    COMPLEX_SIGNALS = re.compile(
        r"\b(jelaskan|ringkas|bandingkan|hubungan|mengapa|kenapa|paling relevan|rekomendasi)\b",
        re.IGNORECASE,
    )

    def decide(self, question, evidence_assessment, retrieval_plan, retrieved_documents):
        if evidence_assessment.answerability == Answerability.NOT_ANSWERABLE:
            return LLMDecisionResult(LLMDecision.NO_ANSWER, evidence_assessment.reason)
        if evidence_assessment.answerability == Answerability.PARTIALLY_ANSWERABLE:
            return LLMDecisionResult(LLMDecision.USE_LLM, "partial answer requires composition")
        if self.COMPLEX_SIGNALS.search(question):
            return LLMDecisionResult(LLMDecision.USE_LLM, "complex question shape")
        intent = getattr(retrieval_plan, "intent", QueryIntent.GENERAL)
        normalized = " ".join(question.casefold().split())
        top_type = retrieved_documents[0].metadata.get("type") if retrieved_documents else None
        if intent == QueryIntent.STAFF and re.search(r"\bsiapa\b", normalized):
            return LLMDecisionResult(LLMDecision.USE_LOCAL, "simple structured staff fact")
        if intent == QueryIntent.FTMM and re.match(r"^(apa|what)\s+(visi|lokasi|kontak)\b", normalized):
            return LLMDecisionResult(LLMDecision.USE_LOCAL, "simple FTMM section fact")
        if intent == QueryIntent.COURSE and re.search(
            r"\b(berapa\s+sks|semester\s+berapa|prasyarat)\b", normalized
        ):
            return LLMDecisionResult(LLMDecision.USE_LOCAL, "simple structured course fact")
        if (intent == QueryIntent.LECTURER or top_type == "lecturer") and re.search(
            r"\b(research interest|minat penelitian|email|program studi)\b", normalized
        ):
            name = str(retrieved_documents[0].metadata.get("name", "")).casefold()
            name_tokens = [token for token in re.findall(r"\w+", name) if len(token) > 3]
            if name_tokens and any(token in normalized for token in name_tokens[:3]):
                return LLMDecisionResult(LLMDecision.USE_LOCAL, "simple structured lecturer fact")
        return LLMDecisionResult(LLMDecision.USE_LLM, "ambiguous or compositional answer")
