from dataclasses import asdict, dataclass
from threading import Lock


@dataclass
class TelemetrySnapshot:
    total_questions: int = 0
    local_answers: int = 0
    no_answers: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    llm_failures: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @property
    def bypass_rate(self):
        if not self.total_questions:
            return 0.0
        bypassed = self.local_answers + self.no_answers + self.cache_hits
        return bypassed / self.total_questions

    def as_dict(self):
        payload = asdict(self)
        payload["llm_bypass_rate"] = self.bypass_rate
        return payload


class LLMTelemetry:
    """Thread-safe process-local usage counters; contains no prompt or secret."""

    def __init__(self):
        self._snapshot = TelemetrySnapshot()
        self._lock = Lock()

    def increment(self, field, amount=1):
        with self._lock:
            setattr(self._snapshot, field, getattr(self._snapshot, field) + amount)

    def record_usage(self, usage):
        if not usage:
            return
        for target, source in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = usage.get(source)
            if isinstance(value, int) and value >= 0:
                self.increment(target, value)

    def snapshot(self):
        with self._lock:
            return TelemetrySnapshot(**asdict(self._snapshot))
