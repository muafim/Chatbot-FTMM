from collections import defaultdict

from domain.models import RetrievedDocument, SparseVector


class InMemorySparseIndex:
    """Inverted index untuk native lexical weights BGE-M3."""

    def __init__(self):
        self._documents = []
        self._vectors = []
        self._inverted = {}
        self.embedding_model = None

    @property
    def is_built(self):
        return bool(self._documents)

    @property
    def document_count(self):
        return len(self._documents)

    @property
    def documents(self):
        return list(self._documents)

    @property
    def posting_count(self):
        return sum(len(postings) for postings in self._inverted.values())

    @property
    def unique_token_count(self):
        return len(self._inverted)

    def build(self, documents, sparse_vectors, embedding_model=None):
        document_list = list(documents)
        vector_list = list(sparse_vectors)
        if not document_list:
            raise ValueError("Sparse index tidak dapat dibuat tanpa Document.")
        if len(document_list) != len(vector_list):
            raise ValueError("Jumlah Document dan sparse vector tidak sama.")

        inverted = defaultdict(list)
        validated = []
        for document_index, vector in enumerate(vector_list):
            if not isinstance(vector, SparseVector) or not vector.values:
                raise ValueError("Sparse index menerima vector kosong atau invalid.")
            validated.append(vector)
            for token_id, weight in vector.values:
                inverted[token_id].append((document_index, weight))

        self._documents = document_list
        self._vectors = validated
        self._inverted = dict(inverted)
        self.embedding_model = embedding_model

    def matching_document_count(self, metadata_filter=None):
        return sum(
            self._matches(document, metadata_filter) for document in self._documents
        )

    def search(
        self, query_vector, top_k=5, embedding_model=None, metadata_filter=None
    ):
        if not self.is_built:
            raise RuntimeError("In-memory sparse index belum dibuat.")
        if top_k <= 0:
            raise ValueError("top_k harus lebih besar dari nol.")
        if not isinstance(query_vector, SparseVector):
            raise ValueError("Query sparse harus berupa SparseVector.")
        if (
            self.embedding_model
            and embedding_model
            and self.embedding_model != embedding_model
        ):
            raise ValueError(
                "Sparse embedding model mismatch: "
                f"index={self.embedding_model} query={embedding_model}."
            )
        if not query_vector.values:
            return []

        allowed = {
            index
            for index, document in enumerate(self._documents)
            if self._matches(document, metadata_filter)
        }
        if not allowed:
            return []

        scores = defaultdict(float)
        for token_id, query_weight in query_vector.values:
            for document_index, document_weight in self._inverted.get(token_id, ()):
                if document_index in allowed:
                    scores[document_index] += query_weight * document_weight

        ranked = sorted(
            (
                (index, score)
                for index, score in scores.items()
                if score > 0.0
            ),
            key=lambda item: (-item[1], self._documents[item[0]].id),
        )[:top_k]
        return [
            RetrievedDocument(document=self._documents[index], score=float(score))
            for index, score in ranked
        ]

    @staticmethod
    def _matches(document, metadata_filter):
        return not metadata_filter or all(
            document.metadata.get(key) == value
            for key, value in metadata_filter.items()
        )
