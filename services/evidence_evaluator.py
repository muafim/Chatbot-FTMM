import re
from dataclasses import dataclass

from domain.models import Answerability


@dataclass(frozen=True)
class EvidenceAssessment:
    answerability: Answerability
    reason: str
    missing_part: str | None = None


class EvidenceEvaluator:
    """Conservative deterministic guard before generation; no score guesswork."""

    OUT_OF_SCOPE_PATTERNS = (
        r"\b(presiden|politik|pemilu|bitcoin|crypto|kripto|harga saham)\b",
        r"\b(cuaca|weather|prakiraan hujan|suhu udara)\b",
        r"\b(resep|rendang|masakan|memasak)\b",
    )
    GENERAL_EXPLANATION_PATTERNS = (
        r"^(apa itu|jelaskan|what is)\s+(machine learning|artificial intelligence|ai|nlp|robotika)\??$",
    )
    DYNAMIC_SCHEDULE_PATTERN = re.compile(
        r"\b(jadwal|mengajar|kelas)\b.*\b(besok|hari ini|minggu ini|jam berapa)\b",
        re.IGNORECASE,
    )
    COURSE_CODE_PATTERN = re.compile(r"\b[A-Z]{2,5}\d{3,5}\b", re.IGNORECASE)

    def evaluate(self, query, retrieved_documents):
        normalized = " ".join((query or "").lower().split())
        if not retrieved_documents or not any(item.content.strip() for item in retrieved_documents):
            return EvidenceAssessment(Answerability.NOT_ANSWERABLE, "no_retrieved_evidence")
        if any(re.search(pattern, normalized) for pattern in self.OUT_OF_SCOPE_PATTERNS):
            return EvidenceAssessment(Answerability.NOT_ANSWERABLE, "out_of_scope")
        if any(re.search(pattern, normalized) for pattern in self.GENERAL_EXPLANATION_PATTERNS):
            return EvidenceAssessment(Answerability.NOT_ANSWERABLE, "general_knowledge_policy")

        corpus = "\n".join(item.content for item in retrieved_documents).lower()
        if "kedokteran" in normalized and "kedokteran" not in corpus:
            return EvidenceAssessment(Answerability.NOT_ANSWERABLE, "unsupported_program")
        if re.search(r"wakil\s+dekan\s+(iv|4)\b", normalized) and not re.search(
            r"wakil\s+dekan\s+(iv|4)\b", corpus
        ):
            return EvidenceAssessment(Answerability.NOT_ANSWERABLE, "unsupported_staff_entity")
        requested_codes = self.COURSE_CODE_PATTERN.findall(query or "")
        if requested_codes and any(code.lower() not in corpus for code in requested_codes):
            return EvidenceAssessment(Answerability.NOT_ANSWERABLE, "unsupported_course_code")
        if self.DYNAMIC_SCHEDULE_PATTERN.search(normalized):
            return EvidenceAssessment(
                Answerability.PARTIALLY_ANSWERABLE,
                "dynamic_schedule_unavailable",
                "Informasi jadwal dinamis tidak tersedia pada sumber FTMM yang dimiliki.",
            )
        return EvidenceAssessment(Answerability.ANSWERABLE, "retrieved_evidence_present")
