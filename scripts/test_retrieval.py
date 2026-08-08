"""Smoke test retrieval lokal tanpa memanggil LLM/OpenAI."""

from app import knowledge_repository, retriever


QUERIES = [
    "machine learning",
    "dosen natural language processing",
    "dekan ftmm",
    "mata kuliah semester 1",
    "surat mahasiswa aktif",
]


def main():
    for query in QUERIES:
        results = retriever.retrieve(query, top_k=3)
        print(f"\nQuery: {query}")
        print("Top results:")
        for rank, item in enumerate(results, start=1):
            metadata = item.document.metadata
            print(f"{rank}.")
            print(f"   Document ID: {item.document.id}")
            print(f"   Type: {metadata.get('type')}")
            print(f"   Score: {item.score:.6f}")
            print(f"   Name/title: {metadata.get('name', '-')}")

    print("\nPerformance:")
    print(f"Document count: {len(knowledge_repository.get_documents())}")
    print(f"Dataset load time: {knowledge_repository.load_seconds:.6f} seconds")
    print(f"Document indexing time: {retriever.indexing_seconds:.6f} seconds")
    print(f"Average retrieval time: {retriever.average_query_seconds:.6f} seconds")


if __name__ == "__main__":
    main()
