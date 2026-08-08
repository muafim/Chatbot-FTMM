import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from domain.models import RetrievedDocument


class InMemoryVectorIndex:
    """Simpan Document dan dense vector di memory tanpa asumsi dimensionality."""

    def __init__(self):
        self._documents = []
        self._embeddings = None
        self.embedding_model = None
        self.embedding_dimension = None

    @property
    def is_built(self):
        return self._embeddings is not None

    def build(self, documents, embeddings, embedding_model=None):
        document_list = list(documents)
        embedding_matrix = np.asarray(embeddings)
        if embedding_matrix.ndim != 2:
            raise ValueError("Document embeddings harus berupa matrix dua dimensi.")
        if len(document_list) != embedding_matrix.shape[0]:
            raise ValueError("Jumlah Document dan document vector tidak sama.")
        if not document_list:
            raise ValueError("Index tidak dapat dibuat tanpa Document.")

        self._documents = document_list
        self._embeddings = embedding_matrix
        self.embedding_model = embedding_model
        self.embedding_dimension = int(embedding_matrix.shape[1])

    def search(self, query_vector, top_k=5, embedding_model=None):
        if not self.is_built:
            raise RuntimeError("In-memory vector index belum dibuat.")
        if top_k <= 0:
            raise ValueError("top_k harus lebih besar dari nol.")

        query_matrix = np.asarray(query_vector).reshape(1, -1)
        if query_matrix.shape[1] != self._embeddings.shape[1]:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"index={self._embeddings.shape[1]} query={query_matrix.shape[1]}."
            )
        if (
            self.embedding_model
            and embedding_model
            and self.embedding_model != embedding_model
        ):
            raise ValueError(
                "Embedding model mismatch: "
                f"index={self.embedding_model} query={embedding_model}."
            )

        similarities = cosine_similarity(query_matrix, self._embeddings)[0]
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            RetrievedDocument(
                document=self._documents[int(index)],
                score=float(similarities[int(index)]),
            )
            for index in ranked_indices
        ]
