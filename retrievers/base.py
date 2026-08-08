from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query, top_k=None):
        """Return ranked RetrievedDocument objects untuk sebuah query."""
        raise NotImplementedError
