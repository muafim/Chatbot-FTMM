import re

from domain.models import AnswerPath, Answerability, GroundedAnswer, GroundingStatus, QueryIntent


class LocalAnswerComposer:
    """Narrow deterministic composer for facts explicitly present in source fields."""

    def compose(self, question, retrieval_plan, retrieved_documents, grounded_context, retrieval_metadata):
        intent = getattr(retrieval_plan, "intent", QueryIntent.GENERAL)
        if intent == QueryIntent.GENERAL and retrieved_documents:
            source_type = retrieved_documents[0].metadata.get("type")
            try:
                intent = QueryIntent(source_type)
            except (ValueError, TypeError):
                pass
        if intent == QueryIntent.STAFF:
            answer = self._staff(question, retrieved_documents, grounded_context)
        elif intent == QueryIntent.FTMM:
            answer = self._ftmm(question, retrieved_documents, grounded_context)
        elif intent == QueryIntent.COURSE:
            answer = self._course(question, retrieved_documents, grounded_context)
        elif intent == QueryIntent.LECTURER:
            answer = self._lecturer(question, retrieved_documents, grounded_context)
        else:
            answer = None
        if answer is None:
            return None
        text, source = answer
        metadata = dict(retrieval_metadata)
        metadata.update({"llm_decision": "use_local", "answer_path": "local"})
        return GroundedAnswer(
            answer=f"{text} [{source.source_id}]",
            citations=[source],
            used_sources=[source],
            has_sufficient_evidence=True,
            answerability=Answerability.ANSWERABLE,
            retrieval_metadata=metadata,
            grounding_status=GroundingStatus.GROUNDED,
            answer_path=AnswerPath.LOCAL,
        )

    def _staff(self, question, retrieved, context):
        normalized = self._normalize(question)
        for item, source in self._pairs(retrieved, context):
            role = self._value(item.metadata, "role") or self._field(item.content, "Jabatan")
            name = self._value(item.metadata, "name") or self._field(item.content, "Nama")
            if role and name and self._normalize(role) in normalized:
                role_label = role if "ftmm" in self._normalize(role) else f"{role} FTMM"
                return f"{role_label} adalah {name}.", source
        return None

    def _ftmm(self, question, retrieved, context):
        normalized = self._normalize(question)
        requested = next((name for name in ("visi", "lokasi", "kontak") if name in normalized), None)
        if not requested:
            return None
        for item, source in self._pairs(retrieved, context):
            section = self._normalize(item.metadata.get("section", ""))
            if requested in section:
                if requested == "kontak":
                    value = self._strip_semantic_header(item.content)
                    if value:
                        return f"Kontak FTMM: {value}", source
                labels = {
                    "lokasi": ("Lokasi", "Lokasi kampus"),
                    "visi": ("Visi",),
                }[requested]
                value = next((self._field(item.content, label) for label in labels if self._field(item.content, label)), None)
                value = value or self._strip_semantic_header(item.content)
                if value:
                    return f"{requested.title()} FTMM: {value}", source
        return None

    def _course(self, question, retrieved, context):
        normalized = self._normalize(question)
        for item, source in self._pairs(retrieved, context):
            metadata = item.metadata
            name = self._value(metadata, "name")
            code = self._value(metadata, "code")
            if not self._entity_mentioned(normalized, name, code):
                continue
            if "sks" in normalized:
                value = self._value(metadata, "credits") or self._field(item.content, "SKS")
                if value:
                    return f"Mata kuliah {name} ({code}) memiliki {value}.", source
            if "semester" in normalized:
                value = self._value(metadata, "semester") or self._field(item.content, "Semester")
                if value:
                    return f"Mata kuliah {name} ({code}) ditawarkan pada semester {value}.", source
            if "prasyarat" in normalized:
                value = self._value(metadata, "prerequisite") or self._field(item.content, "Prasyarat")
                if value:
                    return f"Prasyarat mata kuliah {name} ({code}) adalah {value}.", source
        return None

    def _lecturer(self, question, retrieved, context):
        normalized = self._normalize(question)
        for item, source in self._pairs(retrieved, context):
            name = self._value(item.metadata, "name")
            if not name or not self._name_mentioned(normalized, name):
                continue
            if "email" in normalized:
                value = self._value(item.metadata, "email") or self._field(item.content, "Email")
                label = "email"
            elif "program studi" in normalized:
                value = self._value(item.metadata, "program") or self._field(item.content, "Program Studi")
                label = "program studi"
            else:
                value = self._value(item.metadata, "research_interest") or self._field(item.content, "Research Interest")
                label = "research interest"
            if value and self._normalize(value) not in {"tidak ada", "nan"}:
                return f"{name} memiliki {label}: {value}.", source
        return None

    @staticmethod
    def _pairs(retrieved, context):
        for index, item in enumerate(retrieved, start=1):
            source = context.sources.get(f"S{index}")
            if source:
                yield item, source

    @staticmethod
    def _field(content, label):
        match = re.search(rf"(?im)^{re.escape(label)}\s*:\s*(.+)$", content)
        return match.group(1).strip() if match else None

    @staticmethod
    def _strip_semantic_header(content):
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        return " ".join(lines[1:] if len(lines) > 1 else lines)

    @staticmethod
    def _value(metadata, key):
        value = metadata.get(key)
        if value is None:
            return None
        text = str(value).strip()
        return text if text and text.casefold() not in {"nan", "none", "tidak ada"} else None

    @staticmethod
    def _normalize(value):
        return " ".join(str(value).casefold().split())

    def _entity_mentioned(self, query, name, code):
        return bool((name and self._normalize(name) in query) or (code and self._normalize(code) in query))

    def _name_mentioned(self, query, name):
        tokens = [token for token in re.findall(r"[\w’]+", self._normalize(name)) if len(token) > 3]
        return any(token in query for token in tokens[:3])
