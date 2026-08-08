from dataclasses import replace


def union_candidates(dense_candidates, sparse_candidates):
    """Dense-first deterministic union; retrieval rank tidak memengaruhi reranker score."""
    merged = {}
    order = []
    for source, candidates in (
        ("dense", dense_candidates),
        ("sparse", sparse_candidates),
    ):
        for rank, item in enumerate(candidates, 1):
            document_id = item.document.id
            if document_id not in merged:
                merged[document_id] = item
                order.append(document_id)
            current = merged[document_id]
            if source == "dense":
                merged[document_id] = replace(
                    current,
                    dense_rank=current.dense_rank or rank,
                    dense_score=(
                        current.dense_score
                        if current.dense_score is not None else current.score
                    ),
                )
            else:
                merged[document_id] = replace(
                    current,
                    sparse_rank=current.sparse_rank or rank,
                    sparse_score=(
                        current.sparse_score
                        if current.sparse_score is not None else item.score
                    ),
                )
    return [merged[document_id] for document_id in order]


def candidate_slice(strategy, dense, sparse, hybrid, depth):
    if depth <= 0:
        raise ValueError("Candidate depth harus lebih besar dari nol.")
    if strategy == "dense":
        return list(dense[:depth])
    if strategy == "hybrid":
        return list(hybrid[:depth])
    if strategy == "union":
        return union_candidates(dense[:depth], sparse[:depth])
    raise ValueError("Candidate strategy harus dense, hybrid, atau union.")
