"""Retrieval metrics for a benchmark with exactly one relevant document per query.

Everything works from the 1-based rank of the gold document, so every metric is a
simple function of one integer per query.
"""

from __future__ import annotations

import numpy as np


def gold_ranks(scores: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """1-based rank of the gold document for each query (higher score = better).

    Ties are broken pessimistically: the gold document is ranked below every
    document that has an equal score. This stops a retriever that scores every
    document 0 (for example, a query with no matching terms) from getting credit
    for a lucky position.
    """
    rows = np.arange(scores.shape[0])
    gold_scores = scores[rows, gold][:, None]
    return (scores >= gold_scores).sum(axis=1).astype(np.int64)


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float((ranks <= k).mean())


def mrr_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.where(ranks <= k, 1.0 / ranks, 0.0).mean())


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    """nDCG@k with one relevant document: the ideal DCG is 1, so nDCG = 1/log2(rank+1)."""
    return float(np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0).mean())


def summarize(ranks: np.ndarray, ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    out = {f"Recall@{k}": recall_at_k(ranks, k) for k in ks}
    out["MRR@10"] = mrr_at_k(ranks, 10)
    out["nDCG@10"] = ndcg_at_k(ranks, 10)
    return out
