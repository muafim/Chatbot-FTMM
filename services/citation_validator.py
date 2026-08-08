import re
from dataclasses import dataclass

from domain.models import Answerability


@dataclass(frozen=True)
class CitationValidationResult:
    is_valid: bool
    cited_source_ids: tuple[str, ...]
    invalid_source_ids: tuple[str, ...]
    duplicate_source_ids: tuple[str, ...]
    errors: tuple[str, ...]


class CitationValidator:
    MARKER_PATTERN = re.compile(r"\[S(\d+)\]")

    def validate(self, answer, declared_citations, available_sources, answerability):
        markers = [f"S{value}" for value in self.MARKER_PATTERN.findall(answer or "")]
        declared = [self._normalize(value) for value in (declared_citations or [])]
        all_references = markers + [value for value in declared if value not in markers]
        available = set(available_sources)
        invalid = self._unique(value for value in all_references if value not in available)
        duplicates = self._duplicates(markers)
        cited = self._unique(value for value in markers if value in available)
        errors = []
        if invalid:
            errors.append("citation_unavailable")
        if answerability != Answerability.NOT_ANSWERABLE and not cited:
            errors.append("factual_answer_without_inline_citation")
        if set(declared) != set(cited):
            errors.append("declared_citations_do_not_match_inline_markers")
        return CitationValidationResult(
            is_valid=not errors,
            cited_source_ids=tuple(cited),
            invalid_source_ids=tuple(invalid),
            duplicate_source_ids=tuple(duplicates),
            errors=tuple(errors),
        )

    @staticmethod
    def _normalize(value):
        return str(value).strip().upper().strip("[]")

    @staticmethod
    def _unique(values):
        return list(dict.fromkeys(values))

    @staticmethod
    def _duplicates(values):
        seen = set()
        duplicates = []
        for value in values:
            if value in seen and value not in duplicates:
                duplicates.append(value)
            seen.add(value)
        return duplicates
