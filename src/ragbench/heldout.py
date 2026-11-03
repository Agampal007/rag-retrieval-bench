"""Choose a fusion weight on one set of questions and report it on another.

The fusion sweep in the main README picks its best weight on the same questions it reports, which
flatters the best row. Here the questions are split by Wikipedia article, the weight is chosen on
one half and every number reported comes from the other half. Splitting by article rather than by
question keeps near-duplicate questions about the same paragraph on one side of the split.
"""

from __future__ import annotations

import numpy as np

from .data import Corpus, Queries
from .metrics import gold_ranks, summarize
from .retrievers import zscore_rows


def split_by_article(corpus: Corpus, queries: Queries, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (tune, test) query indices with each article's questions entirely on one side."""
    titles = np.asarray(corpus.titles)[queries.gold]
    articles = sorted(set(titles.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(articles)
    tune_articles = set(articles[: len(articles) // 2])
    in_tune = np.array([t in tune_articles for t in titles])
    return np.flatnonzero(in_tune), np.flatnonzero(~in_tune)


def fusion_ranks(
    first, second, docs: list[str], queries: Queries, weights: tuple[float, ...], batch_size: int = 500
) -> dict[float, np.ndarray]:
    """Gold ranks of z-score fusion `z(first) + w * z(second)` for every weight `w`.

    Each retriever scores every batch once and all weights reuse those scores, which matches
    `ScoreFusion([first, second], weights=[1, w])` without re-scoring per weight.
    """
    first.fit(docs)
    second.fit(docs)
    ranks = {w: np.empty(len(queries), dtype=np.int64) for w in weights}
    for start in range(0, len(queries), batch_size):
        stop = start + batch_size
        texts = queries.texts[start:stop]
        z_first = zscore_rows(first.score(texts))
        z_second = zscore_rows(second.score(texts))
        for w in weights:
            ranks[w][start:stop] = gold_ranks(z_first + w * z_second, queries.gold[start:stop])
    return ranks


def choose_weight(ranks_by_weight: dict[float, np.ndarray], indices: np.ndarray, metric: str = "MRR@10") -> float:
    """The weight with the best `metric` on `indices`; ties go to the smaller weight."""
    return max(sorted(ranks_by_weight), key=lambda w: (summarize(ranks_by_weight[w][indices])[metric], -w))
