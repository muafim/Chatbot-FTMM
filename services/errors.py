class RetrievalError(RuntimeError):
    pass


class InsufficientEvidence(RuntimeError):
    pass


class LLMUnavailable(RuntimeError):
    pass


class LLMGenerationError(RuntimeError):
    pass


class CitationValidationError(RuntimeError):
    pass


class LLMBudgetExceeded(LLMUnavailable):
    pass


class LLMAuthenticationError(LLMUnavailable):
    pass


class LLMRateLimitError(LLMUnavailable):
    pass


class LLMProviderError(LLMGenerationError):
    pass
