import re
import time
import unicodedata

from domain.models import IntentConfidence, QueryIntent, RetrievalPlan


def normalize_for_routing(value):
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


class QueryIntentRouter:
    """Router lokal deterministik; hanya menyusun search plan, bukan jawaban."""

    LECTURER_SIGNALS = (
        "dosen", "pengajar", "lecturer", "researcher", "peneliti",
    )
    COURSE_SIGNALS = (
        "mata kuliah", "matkul", "kelas", "course", "sks", "semester",
    )
    STAFF_SIGNALS = (
        "dekan", "wakil dekan", "pimpinan", "pejabat", "staf",
        "ketua departemen",
    )
    ACADEMIC_SIGNALS = (
        "surat", "pengajuan", "mengajukan", "prosedur", "pedoman",
        "akademik", "ukt", "ktm", "skripsi", "yudisium", "cuti",
    )
    RESEARCH_SIGNALS = (
        "meneliti", "peneliti", "penelitian", "research interest",
        "bidang riset", "bidang penelitian", "fokus",
    )
    PROFILE_SIGNALS = ("profil", "profile", "siapa")

    def __init__(self, available_sections=None, lecturer_names=None):
        self.available_sections = []
        self.lecturer_names = []
        self.last_latency_seconds = 0.0
        self.update_corpus_metadata(
            available_sections=available_sections,
            lecturer_names=lecturer_names,
        )

    def update_corpus_metadata(
        self, documents=None, available_sections=None, lecturer_names=None
    ):
        if documents is not None:
            available_sections = {
                item.metadata.get("section")
                for item in documents
                if item.metadata.get("type") == "ftmm"
                and item.metadata.get("section")
            }
            lecturer_names = {
                item.metadata.get("name")
                for item in documents
                if item.metadata.get("type") == "lecturer"
                and item.metadata.get("name")
            }
        if available_sections is not None:
            self.available_sections = sorted(
                {str(value) for value in available_sections if value}
            )
        if lecturer_names is not None:
            self.lecturer_names = sorted(
                {str(value) for value in lecturer_names if value}
            )

    def route(self, query):
        started_at = time.perf_counter()
        try:
            original = str(query)
            normalized = normalize_for_routing(original)
            if not normalized:
                raise ValueError("Query routing tidak boleh kosong.")

            signals = {
                QueryIntent.LECTURER: self._contains_any(
                    normalized, self.LECTURER_SIGNALS
                ),
                QueryIntent.COURSE: self._contains_any(
                    normalized, self.COURSE_SIGNALS
                ),
                QueryIntent.STAFF: self._contains_any(
                    normalized, self.STAFF_SIGNALS
                ),
                QueryIntent.ACADEMIC: self._contains_any(
                    normalized, self.ACADEMIC_SIGNALS
                ),
            }
            matched_name = self._match_lecturer_name(normalized)
            if matched_name and self._contains_any(normalized, self.PROFILE_SIGNALS):
                signals[QueryIntent.LECTURER] = True

            active = {intent for intent, matched in signals.items() if matched}
            if QueryIntent.STAFF in active:
                return self._plan(original, QueryIntent.STAFF, "explicit staff/position signal")
            if QueryIntent.ACADEMIC in active and not (
                QueryIntent.COURSE in active or QueryIntent.LECTURER in active
            ):
                return self._plan(original, QueryIntent.ACADEMIC, "explicit academic procedure signal")
            if QueryIntent.LECTURER in active and QueryIntent.COURSE in active:
                if re.search(r"\b(dosen|pengajar)\s+pengampu\b", normalized):
                    return self._lecturer_plan(original, normalized, "explicit lecturer-as-course-instructor request")
                return self._general_plan(original, "mixed lecturer and course signals")
            if QueryIntent.LECTURER in active:
                return self._lecturer_plan(original, normalized, "explicit lecturer signal")
            if QueryIntent.COURSE in active:
                return self._plan(original, QueryIntent.COURSE, "explicit course signal")

            section = self._match_section(normalized)
            if section or self._contains_phrase(normalized, "ftmm"):
                metadata_filter = {"type": QueryIntent.FTMM.value}
                if section:
                    metadata_filter["section"] = section
                return RetrievalPlan(
                    query=original,
                    intent=QueryIntent.FTMM,
                    metadata_filter=metadata_filter,
                    confidence=IntentConfidence.HIGH,
                    reason=(
                        f"matched corpus FTMM section: {section}"
                        if section else "explicit FTMM signal"
                    ),
                )
            return self._general_plan(original, "ambiguous topic-only query")
        finally:
            self.last_latency_seconds = time.perf_counter() - started_at

    def _lecturer_plan(self, query, normalized, reason):
        research_query = self._contains_any(normalized, self.RESEARCH_SIGNALS)
        metadata_filter = {"type": QueryIntent.LECTURER.value}
        preferred_role = None
        if research_query:
            preferred_role = "research_interest"
            metadata_filter["chunk_role"] = preferred_role
            reason = f"{reason}; explicit research-interest request"
        return RetrievalPlan(
            query=query,
            intent=QueryIntent.LECTURER,
            metadata_filter=metadata_filter,
            preferred_chunk_role=preferred_role,
            confidence=IntentConfidence.HIGH,
            reason=reason,
        )

    @staticmethod
    def _plan(query, intent, reason):
        return RetrievalPlan(
            query=query,
            intent=intent,
            metadata_filter={"type": intent.value},
            confidence=IntentConfidence.HIGH,
            reason=reason,
        )

    @staticmethod
    def _general_plan(query, reason):
        return RetrievalPlan(query=query, reason=reason)

    def _match_section(self, normalized_query):
        matches = []
        for section in self.available_sections:
            normalized_section = normalize_for_routing(section)
            if not normalized_section:
                continue
            significant = [
                token for token in normalized_section.split()
                if token not in {"ftmm", "fakultas", "kampus"}
            ]
            matched_tokens = sum(
                self._contains_phrase(normalized_query, token)
                for token in significant
            )
            if normalized_section in normalized_query or matched_tokens:
                matches.append(
                    (
                        matched_tokens / len(significant),
                        matched_tokens,
                        len(normalized_section),
                        section,
                    )
                )
        return max(matches)[3] if matches else None

    def _match_lecturer_name(self, normalized_query):
        for name in self.lecturer_names:
            normalized_name = normalize_for_routing(name)
            name_tokens = [
                token for token in normalized_name.split()
                if token not in {"dr", "prof", "phd", "si", "skom", "mt", "msc"}
                and len(token) > 2
            ]
            if any(self._contains_phrase(normalized_query, token) for token in name_tokens):
                return name
        return None

    @classmethod
    def _contains_any(cls, text, phrases):
        return any(cls._contains_phrase(text, phrase) for phrase in phrases)

    @staticmethod
    def _contains_phrase(text, phrase):
        return re.search(rf"(?:^|\s){re.escape(phrase)}(?:$|\s)", text) is not None
