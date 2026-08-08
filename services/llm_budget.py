import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from threading import Lock

from services.errors import LLMBudgetExceeded


class DailyLLMBudgetGuard:
    """Persistent local-date call budget. Each provider attempt consumes one call."""

    def __init__(self, path, daily_call_limit=40, enabled=True):
        if daily_call_limit < 0:
            raise ValueError("LLM_DAILY_CALL_LIMIT tidak boleh negatif.")
        self.path = Path(path)
        self.daily_call_limit = daily_call_limit
        self.enabled = enabled
        self._lock = Lock()

    def reserve_call(self):
        if not self.enabled:
            return None
        with self._lock:
            today = datetime.now().astimezone().date().isoformat()
            state = self._read()
            calls = state.get("llm_calls", 0) if state.get("date") == today else 0
            if calls >= self.daily_call_limit:
                raise LLMBudgetExceeded("Batas penggunaan LLM harian lokal telah tercapai.")
            state = {"date": today, "llm_calls": calls + 1}
            self._write(state)
            return state["llm_calls"]

    def current(self):
        today = datetime.now().astimezone().date().isoformat()
        state = self._read()
        calls = state.get("llm_calls", 0) if state.get("date") == today else 0
        return {"date": today, "llm_calls": calls, "limit": self.daily_call_limit}

    def _read(self):
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            os.replace(temporary, self.path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink(missing_ok=True)
