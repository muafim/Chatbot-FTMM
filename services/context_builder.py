class ContextBuilder:
    """Ubah hasil retrieval menjadi context internal dengan source identifier."""

    def build_context(self, retrieved_documents):
        sections = []
        for source_number, item in enumerate(retrieved_documents, start=1):
            document = item.document
            sections.append(
                "\n".join(
                    [
                        f"[SOURCE {source_number}]",
                        f"Document ID: {document.id}",
                        f"Type: {document.metadata.get('type', 'unknown')}",
                        f"Source: {document.metadata.get('source', 'unknown')}",
                        document.content,
                    ]
                )
            )
        return "\n\n".join(sections)
