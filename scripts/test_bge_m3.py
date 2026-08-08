"""Isolated BGE-M3 dense embedding smoke test; tidak menggunakan Flask atau LLM."""

import gc
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.metrics.pairwise import cosine_similarity

from config import get_bge_m3_settings, get_embedding_settings
from indexes.in_memory_vector_index import InMemoryVectorIndex
from repositories.knowledge_repository import KnowledgeRepository
from services.bge_m3_embedding_service import BGEM3EmbeddingService
from services.embedding_service import EmbeddingService


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEMANTIC_CASES = [
    {
        "name": "Indonesian NLP",
        "query": "dosen yang meneliti pemrosesan bahasa alami",
        "candidates": [
            "Penelitian dosen ini berfokus pada natural language processing dan machine learning.",
            "Penelitian dosen ini berfokus pada teknologi material dan energi.",
            "Informasi jadwal kegiatan akademik mahasiswa.",
        ],
        "expected": 0,
    },
    {
        "name": "Indonesian AI course",
        "query": "mata kuliah yang mempelajari kecerdasan buatan",
        "candidates": [
            "Mata kuliah ini membahas artificial intelligence, machine learning, dan penerapannya.",
            "Mata kuliah ini membahas struktur beton dan teknik konstruksi.",
            "Informasi mengenai surat keterangan mahasiswa aktif.",
        ],
        "expected": 0,
    },
    {
        "name": "Indonesian active student letter",
        "query": "bagaimana cara mengajukan surat mahasiswa aktif",
        "candidates": [
            "Prosedur pengajuan Surat Keterangan Mahasiswa Aktif bagi mahasiswa FTMM.",
            "Profil dosen Program Studi Teknologi Sains Data.",
            "Deskripsi mata kuliah pembelajaran mesin.",
        ],
        "expected": 0,
    },
    {
        "name": "Semantic variation NLP",
        "query": "pengajar yang fokus NLP",
        "candidates": [
            "Research interest: natural language processing.",
            "Research interest: material technology and renewable energy.",
        ],
        "expected": 0,
    },
    {
        "name": "Semantic variation AI",
        "query": "kelas tentang AI",
        "candidates": [
            "Mata kuliah Kecerdasan Artifisial.",
            "Prosedur administrasi surat mahasiswa.",
        ],
        "expected": 0,
    },
    {
        "name": "Cross-lingual HCI",
        "query": "dosen yang meneliti interaksi manusia dan komputer",
        "candidates": [
            "Research interests include Human-Computer Interaction.",
            "Research interests include power systems and material engineering.",
        ],
        "expected": 0,
    },
    {
        "name": "Cross-lingual NLP",
        "query": "pemrosesan bahasa alami",
        "candidates": [
            "Natural Language Processing.",
            "Structural Engineering and Construction.",
        ],
        "expected": 0,
    },
]

FTMM_QUERIES = [
    "dosen yang meneliti natural language processing",
    "mata kuliah yang berkaitan dengan machine learning",
    "siapa dekan FTMM",
    "surat mahasiswa aktif",
    "mata kuliah semester 1",
]


def _rank_baseline(service, case):
    query_vector = service.encode_query(case["query"])
    document_vectors = service.encode_documents(case["candidates"])
    scores = cosine_similarity([query_vector], document_vectors)[0]
    return np.argsort(scores)[::-1], scores


def _rank_bge(service, case):
    query_vector = service.encode_query(case["query"])
    document_vectors = service.encode_documents(case["candidates"])
    scores = document_vectors @ query_vector
    return np.argsort(scores)[::-1], scores


def _select_ftmm_sample(documents):
    priorities = {
        "course": ["machine learning", "natural language", "semester: 1"],
        "lecturer": ["natural language", "human-computer interaction", "machine learning"],
        "academic": ["surat keterangan mahasiswa aktif", "surat", "panduan"],
        "staff": ["jabatan: dekan", "wakil dekan"],
        "ftmm": ["tentang fakultas"],
    }
    limits = {"course": 10, "lecturer": 6, "academic": 4, "staff": 4, "ftmm": 1}
    selected = []
    for document_type, limit in limits.items():
        typed = [d for d in documents if d.metadata["type"] == document_type]
        ordered = []
        for keyword in priorities[document_type]:
            ordered.extend(
                d for d in typed if keyword in d.content.lower() and d not in ordered
            )
        ordered.extend(d for d in typed if d not in ordered)
        selected.extend(ordered[:limit])
    return selected


def main():
    process = psutil.Process()
    memory_before = process.memory_info().rss

    print("=== BASELINE COMPARISON ===")
    baseline = EmbeddingService(**get_embedding_settings())
    baseline_results = {}
    for case in SEMANTIC_CASES:
        order, scores = _rank_baseline(baseline, case)
        baseline_results[case["name"]] = int(order[0])
        print(f"\n{case['name']} | Query: {case['query']}")
        print("BASELINE:")
        for rank, index in enumerate(order, start=1):
            print(f"  {rank}. candidate={int(index)+1} score={scores[index]:.6f}")

    baseline._model = None
    del baseline
    gc.collect()

    print("\n=== BGE-M3 VALIDATION ===")
    bge = BGEM3EmbeddingService(**get_bge_m3_settings())
    load_started = time.perf_counter()
    bge.load_model()
    load_seconds = time.perf_counter() - load_started
    memory_after_load = process.memory_info().rss

    first_query_started = time.perf_counter()
    first_query = bge.encode_query(SEMANTIC_CASES[0]["query"])
    first_query_seconds = time.perf_counter() - first_query_started
    if first_query.shape != (1024,):
        raise AssertionError(f"BGE-M3 dimension bukan 1024: {first_query.shape}")
    if not np.isfinite(first_query).all() or np.allclose(first_query, 0):
        raise AssertionError("BGE-M3 query vector tidak valid.")

    bge_results = {}
    all_expected_pass = True
    observed_norms = []
    for case in SEMANTIC_CASES:
        order, scores = _rank_bge(bge, case)
        bge_results[case["name"]] = int(order[0])
        passed = int(order[0]) == case["expected"]
        all_expected_pass = all_expected_pass and passed
        print(f"\n{case['name']} | Query: {case['query']}")
        print("BGE-M3:")
        for rank, index in enumerate(order, start=1):
            print(f"  {rank}. candidate={int(index)+1} score={scores[index]:.6f}")
        print(f"  expected_candidate=1 pass={passed}")
        vectors = bge.encode_documents(case["candidates"])
        observed_norms.extend(np.linalg.norm(vectors, axis=1).tolist())

    if not all_expected_pass:
        raise AssertionError("Satu atau lebih semantic test BGE-M3 gagal ranking #1.")

    print("\n=== BASELINE VS BGE-M3 SUMMARY ===")
    for case in SEMANTIC_CASES:
        expected = case["expected"]
        baseline_top = baseline_results[case["name"]]
        bge_top = bge_results[case["name"]]
        print(
            f"{case['name']}: baseline_top={baseline_top+1} "
            f"baseline_correct={baseline_top==expected} bge_top={bge_top+1} "
            f"bge_correct={bge_top==expected}"
        )

    ten_texts = [f"Dokumen benchmark singkat nomor {index}." for index in range(10)]
    fifty_texts = [f"Dokumen benchmark singkat nomor {index}." for index in range(50)]
    ten_started = time.perf_counter()
    ten_vectors = bge.encode_documents(ten_texts)
    ten_seconds = time.perf_counter() - ten_started
    fifty_started = time.perf_counter()
    fifty_vectors = bge.encode_documents(fifty_texts)
    fifty_seconds = time.perf_counter() - fifty_started

    repository = KnowledgeRepository(PROJECT_ROOT / "data")
    sample_documents = _select_ftmm_sample(repository.get_documents())
    sample_started = time.perf_counter()
    sample_vectors = bge.encode_documents(d.content for d in sample_documents)
    sample_index = InMemoryVectorIndex()
    sample_index.build(
        sample_documents,
        sample_vectors,
        embedding_model=bge.model_name,
    )
    sample_indexing_seconds = time.perf_counter() - sample_started

    retrieval_times = []
    print("\n=== FTMM SAMPLE RETRIEVAL ===")
    print(f"Sample document count: {len(sample_documents)}")
    for query in FTMM_QUERIES:
        query_vector = bge.encode_query(query)
        retrieval_started = time.perf_counter()
        results = sample_index.search(
            query_vector,
            top_k=5,
            embedding_model=bge.model_name,
        )
        retrieval_times.append(time.perf_counter() - retrieval_started)
        print(f"\nQuery: {query}")
        for rank, item in enumerate(results, start=1):
            content_excerpt = item.document.content.replace("\n", " ")[:140]
            print(
                f"  {rank}. id={item.document.id} "
                f"type={item.document.metadata['type']} "
                f"name={item.document.metadata.get('name', '-')} "
                f"score={item.score:.6f} excerpt={content_excerpt}"
            )

    warm_times = []
    for query in ["FTMM", "mata kuliah", "dosen NLP"]:
        started = time.perf_counter()
        bge.encode_query(query)
        warm_times.append(time.perf_counter() - started)

    print("\n=== VECTOR VALIDATION ===")
    print(f"query_shape={first_query.shape}")
    print(f"ten_document_shape={ten_vectors.shape}")
    print(f"fifty_document_shape={fifty_vectors.shape}")
    print(f"finite={np.isfinite(fifty_vectors).all()}")
    print(f"all_zero={np.allclose(fifty_vectors, 0)}")
    print(f"norm_min={min(observed_norms):.6f} norm_max={max(observed_norms):.6f}")
    print(f"model_identity={bge.model_identity}")

    print("\n=== PERFORMANCE ===")
    print(f"model_load_seconds={load_seconds:.6f}")
    print(f"cold_load_plus_first_query_seconds={load_seconds + first_query_seconds:.6f}")
    print(f"first_query_after_load_seconds={first_query_seconds:.6f}")
    print(f"warm_query_average_seconds={sum(warm_times)/len(warm_times):.6f}")
    print(f"ten_document_encoding_seconds={ten_seconds:.6f}")
    print(f"fifty_document_encoding_seconds={fifty_seconds:.6f}")
    print(f"sample_ftmm_indexing_seconds={sample_indexing_seconds:.6f}")
    print(f"retrieval_average_seconds={sum(retrieval_times)/len(retrieval_times):.6f}")
    print(f"process_rss_before_gib={memory_before/1024**3:.3f}")
    print(f"process_rss_after_load_gib={memory_after_load/1024**3:.3f}")


if __name__ == "__main__":
    main()
