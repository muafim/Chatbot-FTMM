from domain.models import ChatResult


class ChatService:
    """Orkestrasi retrieval, context construction, dan answer generation."""

    def __init__(self, retriever, context_builder, llm_service):
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm_service = llm_service

    def answer(self, question, conversation_history=None):
        retrieved_documents = self.retriever.retrieve(question)
        if not retrieved_documents:
            return ChatResult(
                answer="Tidak ditemukan informasi yang cukup relevan pada dataset lokal.",
                retrieved_documents=[],
            )

        context = self.context_builder.build_context(retrieved_documents)
        answer = self.llm_service.generate_answer(
            question=question,
            context=context,
            conversation_history=conversation_history,
        )
        return ChatResult(answer=answer, retrieved_documents=retrieved_documents)
