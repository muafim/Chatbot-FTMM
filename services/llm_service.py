from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class LLMGeneratedAnswer:
    answer: str
    citations: tuple[str, ...]
    answerability: str
    warning: str | None = None
    provider: str | None = None
    requested_model: str | None = None
    actual_model: str | None = None
    usage: dict = field(default_factory=dict)
    latency_ms: float | None = None
    provider_calls: int = 1


class LLMProvider(Protocol):
    provider_name: str
    model_name: str

    def generate_grounded_answer(
        self,
        question,
        grounded_context,
        evidence_assessment,
        correction_instruction=None,
    ) -> LLMGeneratedAnswer:
        ...


class LLMService:
    """Provider-neutral facade used by ChatService."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    @property
    def provider_name(self):
        return self.provider.provider_name

    @property
    def model_name(self):
        return self.provider.model_name

    def generate_grounded_answer(
        self,
        question,
        grounded_context,
        evidence_assessment,
        conversation_history=None,
        correction_instruction=None,
    ):
        # Stage 7B intentionally does not forward conversation history.
        return self.provider.generate_grounded_answer(
            question=question,
            grounded_context=grounded_context,
            evidence_assessment=evidence_assessment,
            correction_instruction=correction_instruction,
        )
