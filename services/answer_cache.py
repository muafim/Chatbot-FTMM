import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from domain.models import AnswerPath, Answerability, Citation, GroundedAnswer, GroundingStatus


CACHE_VERSION = 1


def normalize_cache_query(query):
    normalized = " ".join(str(query).casefold().strip().split())
    return re.sub(r"[?!.]+$", "", normalized).rstrip()


class GroundedAnswerCache:
    def __init__(
        self,
        directory,
        corpus_fingerprint,
        retrieval_signature,
        provider,
        model,
        prompt_version,
        schema_version,
        enabled=True,
    ):
        self.directory = Path(directory)
        self.corpus_fingerprint = corpus_fingerprint
        self.retrieval_signature = retrieval_signature
        self.provider = provider
        self.model = model
        self.prompt_version = prompt_version
        self.schema_version = schema_version
        self.enabled = enabled

    def key_for(self, query):
        payload = self._identity(query)
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def load(self, query):
        if not self.enabled:
            return None
        path = self.directory / f"{self.key_for(query)}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("identity") != self._identity(query):
                return None
            answer = payload["grounded_answer"]
            citations = [Citation(**item) for item in answer.get("citations", [])]
            if not citations or answer.get("grounding_status") not in {"grounded", "partial"}:
                return None
            metadata = dict(answer.get("retrieval_metadata", {}))
            metadata.update({"cache_hit": True, "cache_key": self.key_for(query)})
            return GroundedAnswer(
                answer=answer["answer"],
                citations=citations,
                used_sources=citations,
                has_sufficient_evidence=True,
                answerability=Answerability(answer["answerability"]),
                warning=answer.get("warning"),
                retrieval_metadata=metadata,
                grounding_status=GroundingStatus(answer["grounding_status"]),
                answer_path=AnswerPath.CACHE,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def save(self, query, grounded_answer):
        if not self.enabled or not self._cacheable(grounded_answer):
            return False
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self.key_for(query)}.json"
        metadata = grounded_answer.retrieval_metadata
        payload = {
            "cache_version": CACHE_VERSION,
            "identity": self._identity(query),
            "query": query,
            "normalized_query": normalize_cache_query(query),
            "provider": metadata.get("provider", self.provider),
            "requested_model": metadata.get("requested_model", self.model),
            "actual_model": metadata.get("actual_model"),
            "source_ids": [source.source_id for source in grounded_answer.citations],
            "citation_ids": [source.source_id for source in grounded_answer.citations],
            "corpus_fingerprint": self.corpus_fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "answerability": grounded_answer.answerability.value,
            "grounded_answer": {
                "answer": grounded_answer.answer,
                "citations": [asdict(item) for item in grounded_answer.citations],
                "answerability": grounded_answer.answerability.value,
                "warning": grounded_answer.warning,
                "retrieval_metadata": metadata,
                "grounding_status": grounded_answer.grounding_status.value,
            },
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory, delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            os.replace(temporary, path)
            return True
        finally:
            if temporary and temporary.exists():
                temporary.unlink(missing_ok=True)

    def _identity(self, query):
        return {
            "cache_version": CACHE_VERSION,
            "normalized_query": normalize_cache_query(query),
            "corpus_fingerprint": self.corpus_fingerprint,
            "retrieval_signature": self.retrieval_signature,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
        }

    @staticmethod
    def _cacheable(answer):
        return (
            answer.answer_path == AnswerPath.LLM
            and answer.grounding_status in {GroundingStatus.GROUNDED, GroundingStatus.PARTIAL}
            and bool(answer.citations)
        )
