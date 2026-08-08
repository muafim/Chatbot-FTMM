from dataclasses import replace


def reciprocal_rank_fusion(
    dense_results,
    sparse_results,
    rrf_k=60,
    dense_weight=1.0,
    sparse_weight=1.0,
):
    """Gabungkan ranking; RRF score hanya sinyal urutan, bukan confidence."""
    if rrf_k <= 0:
        raise ValueError("RRF k harus lebih besar dari nol.")
    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError("RRF weights tidak boleh negatif.")
    if dense_weight == 0 and sparse_weight == 0:
        raise ValueError("Minimal satu RRF weight harus lebih besar dari nol.")

    merged = {}
    for rank, item in enumerate(dense_results, 1):
        entry = merged.setdefault(
            item.document.id,
            {"document": item.document, "fusion": 0.0},
        )
        entry.update(dense_rank=rank, dense_score=item.score)
        entry["fusion"] += dense_weight / (rrf_k + rank)
    for rank, item in enumerate(sparse_results, 1):
        entry = merged.setdefault(
            item.document.id,
            {"document": item.document, "fusion": 0.0},
        )
        entry.update(sparse_rank=rank, sparse_score=item.score)
        entry["fusion"] += sparse_weight / (rrf_k + rank)

    ranked = sorted(
        merged.values(),
        key=lambda entry: (
            -entry["fusion"],
            min(entry.get("dense_rank", 10**9), entry.get("sparse_rank", 10**9)),
            entry["document"].id,
        ),
    )
    return [
        replace(
            _as_result(entry),
            final_rank=rank,
        )
        for rank, entry in enumerate(ranked, 1)
    ]


def _as_result(entry):
    from domain.models import RetrievedDocument

    return RetrievedDocument(
        document=entry["document"],
        score=entry["fusion"],
        dense_rank=entry.get("dense_rank"),
        dense_score=entry.get("dense_score"),
        sparse_rank=entry.get("sparse_rank"),
        sparse_score=entry.get("sparse_score"),
        fusion_score=entry["fusion"],
    )
