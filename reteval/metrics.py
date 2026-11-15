"""Per-query ranking metrics (graded nDCG, recall, reciprocal rank)."""

from __future__ import annotations

import numpy as np

K = 10


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = K) -> float:
    """nDCG@k with linear gains (the trec_eval / BEIR convention)."""
    gains = [rel.get(d, 0) for d in ranked[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int = K) -> float:
    return sum(1 for d in ranked[:k] if d in rel) / len(rel)


def rr_at_k(ranked: list[str], rel: dict[str, int], k: int = K) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def top_k(scores: np.ndarray, k: int = K) -> np.ndarray:
    """Indices of the k highest scores per row, best first. scores: (n_queries, n_docs)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)
