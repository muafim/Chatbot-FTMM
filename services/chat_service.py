import logging
from time import perf_counter

from domain.models import (
    AnswerPath,
    Answerability,
    ChatResult,
    GroundedAnswer,
    GroundingStatus,
)
from services.citation_validator import CitationValidator
from services.errors import LLMBudgetExceeded, LLMGenerationError, LLMUnavailable, RetrievalError
from services.evidence_evaluator import EvidenceEvaluator
from services.llm_decision_service import LLMDecision, LLMDecisionService
from services.local_answer_composer import LocalAnswerComposer


logger = logging.getLogger(__name__)


class ChatService:
    """Provider-neutral grounded RAG with deterministic LLM budgeting."""

    NO_ANSWER_MESSAGE = (
        "Saya belum menemukan informasi yang cukup pada sumber FTMM yang tersedia "
        "untuk menjawab pertanyaan tersebut. Coba gunakan istilah yang lebih spesifik."
    )
    OUT_OF_SCOPE_MESSAGE = (
        "Pertanyaan tersebut berada di luar cakupan knowledge base FTMM. "
        "Saya berfokus pada informasi akademik dan kelembagaan FTMM."
    )
    GENERAL_POLICY_MESSAGE = (
        "Saya berfokus pada informasi yang tersedia di knowledge base FTMM dan tidak "
        "memberikan penjelasan pengetahuan umum tanpa sumber FTMM."
    )
    LLM_UNAVAILABLE_MESSAGE = (
        "Sumber FTMM berhasil ditemukan, tetapi layanan penyusunan jawaban AI belum tersedia."
    )
    BUDGET_LIMIT_MESSAGE = (
        "Layanan penyusunan jawaban AI sedang mencapai batas penggunaan. "
        "Informasi sumber FTMM yang ditemukan tetap ditampilkan di bawah ini."
    )

    def __init__(
        self,
        retriever,
        context_builder,
        llm_service,
        evidence_evaluator=None,
        citation_validator=None,
        decision_service=None,
        local_composer=None,
        answer_cache=None,
        telemetry=None,
        citation_correction_retries=1,
        max_context_sources=3,
        debug=False,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm_service = llm_service
        self.evidence_evaluator = evidence_evaluator or EvidenceEvaluator()
        self.citation_validator = citation_validator or CitationValidator()
        self.decision_service = decision_service or LLMDecisionService()
        self.local_composer = local_composer or LocalAnswerComposer()
        self.answer_cache = answer_cache
        self.telemetry = telemetry
        self.citation_correction_retries = min(max(citation_correction_retries, 0), 1)
        self.max_context_sources = max_context_sources
        self.debug = debug

    def answer(self, question, conversation_history=None):
        total_started = perf_counter()
        self._count("total_questions")

        retrieval_started = perf_counter()
        try:
            retrieved = self.retriever.retrieve(question)
        except Exception as exc:
            logger.error("Retrieval gagal. Tipe error: %s", type(exc).__name__)
            raise RetrievalError("Retrieval sumber FTMM gagal.") from exc
        retrieval_ms = self._elapsed_ms(retrieval_started)

        evidence_started = perf_counter()
        assessment = self.evidence_evaluator.evaluate(question, retrieved)
        evidence_ms = self._elapsed_ms(evidence_started)
        metadata = self._retrieval_metadata(retrieved, assessment, retrieval_ms, evidence_ms)

        plan = getattr(self.retriever, "last_retrieval_plan", None)
        gate_started = perf_counter()
        decision = self.decision_service.decide(question, assessment, plan, retrieved)
        metadata["timing_ms"]["smart_gate"] = self._elapsed_ms(gate_started)
        metadata.update({"llm_decision": decision.decision.value, "llm_decision_reason": decision.reason})

        if decision.decision == LLMDecision.NO_ANSWER:
            self._count("no_answers")
            metadata["answer_path"] = AnswerPath.NO_ANSWER.value
            metadata["timing_ms"]["total"] = self._elapsed_ms(total_started)
            self._log_trace(question, assessment, metadata, (), True)
            return self._result(
                answer=self._no_answer_for(assessment.reason),
                retrieved=retrieved,
                answerability=Answerability.NOT_ANSWERABLE,
                grounding_status=GroundingStatus.INSUFFICIENT_EVIDENCE,
                answer_path=AnswerPath.NO_ANSWER,
                retrieval_metadata=metadata,
            )

        selected = list(retrieved[: self.max_context_sources])
        context_started = perf_counter()
        context = self.context_builder.build_grounded_context(selected)
        metadata["timing_ms"]["context"] = self._elapsed_ms(context_started)
        metadata["context_chunk_ids"] = [item.chunk_id for item in selected]

        if decision.decision == LLMDecision.USE_LOCAL:
            local_started = perf_counter()
            local = self.local_composer.compose(question, plan, selected, context, metadata)
            metadata["timing_ms"]["local_or_cache"] = self._elapsed_ms(local_started)
            if local and self._grounded_answer_is_valid(local, context):
                self._count("local_answers")
                local.retrieval_metadata["timing_ms"]["total"] = self._elapsed_ms(total_started)
                self._log_trace(question, assessment, local.retrieval_metadata, [c.source_id for c in local.citations], True)
                return ChatResult.from_grounded(local, retrieved)
            metadata.update({
                "llm_decision": LLMDecision.USE_LLM.value,
                "llm_decision_reason": "local composer could not prove exact structured fact",
            })

        cache_started = perf_counter()
        cached = self.answer_cache.load(question) if self.answer_cache else None
        metadata["timing_ms"]["local_or_cache"] = self._elapsed_ms(cache_started)
        if cached and self._cached_answer_matches_context(cached, context):
            self._count("cache_hits")
            cached_metadata = dict(cached.retrieval_metadata)
            cached_metadata.update(metadata)
            cached_metadata.update({"cache_hit": True, "answer_path": AnswerPath.CACHE.value})
            cached_metadata["timing_ms"]["total"] = self._elapsed_ms(total_started)
            cached = self._copy_grounded(cached, retrieval_metadata=cached_metadata, answer_path=AnswerPath.CACHE)
            self._log_trace(question, assessment, cached_metadata, [c.source_id for c in cached.citations], True)
            return ChatResult.from_grounded(cached, retrieved)

        if not hasattr(self.llm_service, "generate_grounded_answer"):
            answer = self.llm_service.generate_answer(
                question=question,
                context=self.context_builder.build_context(selected),
                conversation_history=conversation_history,
            )
            return ChatResult(answer=answer, retrieved_documents=retrieved)

        return self._generate(
            question=question,
            retrieved=retrieved,
            assessment=assessment,
            context=context,
            metadata=metadata,
            total_started=total_started,
        )

    def _generate(self, question, retrieved, assessment, context, metadata, total_started):
        generation_ms = validation_ms = 0.0
        correction = None
        generated = validation = None
        effective_answerability = assessment.answerability
        provider_calls_total = 0
        usage_total = {}
        attempts = self.citation_correction_retries + 1
        try:
            for attempt in range(attempts):
                started = perf_counter()
                generated = self.llm_service.generate_grounded_answer(
                    question=question,
                    grounded_context=context,
                    evidence_assessment=assessment,
                    correction_instruction=correction,
                )
                generation_ms += self._elapsed_ms(started)
                provider_calls_total += generated.provider_calls
                for key, value in generated.usage.items():
                    if isinstance(value, (int, float)):
                        usage_total[key] = usage_total.get(key, 0) + value
                effective_answerability = self._reconcile_answerability(
                    assessment.answerability, generated.answerability
                )
                started = perf_counter()
                validation = self.citation_validator.validate(
                    generated.answer,
                    generated.citations,
                    context.sources,
                    effective_answerability,
                )
                validation_ms += self._elapsed_ms(started)
                if validation.is_valid:
                    break
                correction = (
                    "Previous output failed citation validation. Regenerate once using "
                    f"only {', '.join(context.available_source_ids)} and make the JSON citation list "
                    "exactly match inline markers."
                )
        except LLMBudgetExceeded:
            return self._generation_unavailable(
                retrieved, assessment, context, metadata, total_started, budget_limited=True
            )
        except (LLMUnavailable, LLMGenerationError):
            return self._generation_unavailable(
                retrieved, assessment, context, metadata, total_started, budget_limited=False
            )

        metadata["timing_ms"].update({
            "llm": round(generation_ms, 3),
            "validation": round(validation_ms, 3),
            "total": self._elapsed_ms(total_started),
        })
        metadata.update({
            "provider": generated.provider,
            "requested_model": generated.requested_model,
            "actual_model": generated.actual_model,
            "usage": usage_total,
            "provider_calls": provider_calls_total,
            "citation_retry_count": attempt,
            "cache_hit": False,
            "answer_path": AnswerPath.LLM.value,
        })
        if not validation.is_valid:
            self._count("no_answers")
            return self._result(
                self.NO_ANSWER_MESSAGE,
                retrieved,
                Answerability.NOT_ANSWERABLE,
                GroundingStatus.CITATION_INVALID,
                answer_path=AnswerPath.NO_ANSWER,
                warning="Jawaban provider ditahan karena citation tetap invalid.",
                retrieval_metadata=metadata,
            )

        sources = [context.sources[source_id] for source_id in validation.cited_source_ids]
        if effective_answerability == Answerability.NOT_ANSWERABLE:
            self._count("no_answers")
            answer = self.NO_ANSWER_MESSAGE
            sources = []
            status = GroundingStatus.INSUFFICIENT_EVIDENCE
            path = AnswerPath.NO_ANSWER
        else:
            answer = generated.answer
            status = GroundingStatus.PARTIAL if effective_answerability == Answerability.PARTIALLY_ANSWERABLE else GroundingStatus.GROUNDED
            path = AnswerPath.LLM
        grounded = GroundedAnswer(
            answer=answer,
            citations=sources,
            used_sources=sources,
            has_sufficient_evidence=bool(sources),
            answerability=effective_answerability,
            warning=assessment.missing_part or generated.warning,
            retrieval_metadata=metadata,
            grounding_status=status,
            answer_path=path,
        )
        if path == AnswerPath.LLM and self.answer_cache:
            self.answer_cache.save(question, grounded)
        self._log_trace(question, assessment, metadata, validation.cited_source_ids, True)
        return ChatResult.from_grounded(grounded, retrieved)

    def _generation_unavailable(self, retrieved, assessment, context, metadata, total_started, budget_limited):
        metadata["timing_ms"]["total"] = self._elapsed_ms(total_started)
        metadata["answer_path"] = AnswerPath.UNAVAILABLE.value
        metadata["provider_failure"] = True
        sources = list(context.sources.values())
        return self._result(
            self.BUDGET_LIMIT_MESSAGE if budget_limited else self.LLM_UNAVAILABLE_MESSAGE,
            retrieved,
            assessment.answerability,
            GroundingStatus.BUDGET_LIMITED if budget_limited else GroundingStatus.LLM_UNAVAILABLE,
            used_sources=sources,
            answer_path=AnswerPath.UNAVAILABLE,
            warning="Generation dibatasi; source cards berasal dari retrieval dan bukan citation jawaban.",
            retrieval_metadata=metadata,
        )

    def _cached_answer_matches_context(self, answer, context):
        current_parents = {source.parent_document_id for source in context.sources.values()}
        if any(source.parent_document_id not in current_parents for source in answer.citations):
            return False
        cached_sources = {source.source_id: source for source in answer.citations}
        validation = self.citation_validator.validate(
            answer.answer,
            [item.source_id for item in answer.citations],
            cached_sources,
            answer.answerability,
        )
        return validation.is_valid

    def _grounded_answer_is_valid(self, answer, context):
        validation = self.citation_validator.validate(
            answer.answer,
            [item.source_id for item in answer.citations],
            context.sources,
            answer.answerability,
        )
        return validation.is_valid

    @staticmethod
    def _copy_grounded(answer, retrieval_metadata, answer_path):
        return GroundedAnswer(
            answer=answer.answer,
            citations=list(answer.citations),
            used_sources=list(answer.used_sources),
            has_sufficient_evidence=answer.has_sufficient_evidence,
            answerability=answer.answerability,
            warning=answer.warning,
            retrieval_metadata=retrieval_metadata,
            grounding_status=answer.grounding_status,
            answer_path=answer_path,
        )

    @staticmethod
    def _result(
        answer,
        retrieved,
        answerability,
        grounding_status,
        citations=None,
        used_sources=None,
        answer_path=AnswerPath.NO_ANSWER,
        warning=None,
        retrieval_metadata=None,
    ):
        grounded = GroundedAnswer(
            answer=answer,
            citations=list(citations or []),
            used_sources=list(used_sources or citations or []),
            has_sufficient_evidence=bool(citations),
            answerability=answerability,
            warning=warning,
            retrieval_metadata=dict(retrieval_metadata or {}),
            grounding_status=grounding_status,
            answer_path=answer_path,
        )
        return ChatResult.from_grounded(grounded, retrieved)

    def _retrieval_metadata(self, retrieved, assessment, retrieval_ms, evidence_ms):
        plan = getattr(self.retriever, "last_retrieval_plan", None)
        return {
            "retrieved_count": len(retrieved),
            "chunk_ids": [item.chunk_id for item in retrieved],
            "intent": getattr(getattr(plan, "intent", None), "value", "unknown"),
            "evidence_reason": assessment.reason,
            "timing_ms": {
                "retrieval": round(retrieval_ms, 3),
                "evidence": round(evidence_ms, 3),
                "smart_gate": 0.0,
                "context": 0.0,
                "local_or_cache": 0.0,
                "llm": 0.0,
                "validation": 0.0,
            },
        }

    def _no_answer_for(self, reason):
        if reason == "out_of_scope":
            return self.OUT_OF_SCOPE_MESSAGE
        if reason == "general_knowledge_policy":
            return self.GENERAL_POLICY_MESSAGE
        return self.NO_ANSWER_MESSAGE

    def _count(self, field):
        if self.telemetry:
            self.telemetry.increment(field)

    def _log_trace(self, question, assessment, metadata, citations, valid):
        if not self.debug:
            return
        logger.info(
            "RAG query=%r intent=%s decision=%s path=%s cache_hit=%s model=%s actual_model=%s usage=%s citations=%s valid=%s timing_ms=%s",
            question,
            metadata.get("intent"),
            metadata.get("llm_decision"),
            metadata.get("answer_path"),
            metadata.get("cache_hit", False),
            metadata.get("requested_model"),
            metadata.get("actual_model"),
            metadata.get("usage", {}),
            list(citations),
            valid,
            metadata.get("timing_ms"),
        )

    @staticmethod
    def _elapsed_ms(started):
        return round((perf_counter() - started) * 1000, 3)

    @staticmethod
    def _reconcile_answerability(deterministic, generated):
        try:
            generated_value = Answerability(generated)
        except ValueError:
            return deterministic
        if generated_value == Answerability.NOT_ANSWERABLE:
            return Answerability.NOT_ANSWERABLE
        if generated_value == Answerability.PARTIALLY_ANSWERABLE and deterministic == Answerability.ANSWERABLE:
            return Answerability.PARTIALLY_ANSWERABLE
        return deterministic
