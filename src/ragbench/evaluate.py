"""Run retrievers over a benchmark and collect metrics."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .data import Corpus, Queries
from .metrics import gold_ranks, summarize


def rank_gold(retriever, query_texts: list[str], gold: np.ndarray, batch_size: int = 500) -> tuple[np.ndarray, float]:
    """Return (gold ranks, milliseconds per query spent scoring)."""
    ranks = np.empty(len(gold), dtype=np.int64)
    started = time.perf_counter()
    for start in range(0, len(gold), batch_size):
        stop = start + batch_size
        scores = retriever.score(query_texts[start:stop])
        ranks[start:stop] = gold_ranks(scores, gold[start:stop])
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return ranks, elapsed_ms / len(gold)


def run_benchmark(retrievers: list, corpus: Corpus, queries: Queries) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    all_ranks: dict[str, np.ndarray] = {}
    for retriever in retrievers:
        retriever.fit(corpus.texts)
        ranks, ms_per_query = rank_gold(retriever, queries.texts, queries.gold)
        all_ranks[retriever.name] = ranks
        row = {"Retriever": retriever.name, **summarize(ranks), "ms/query": ms_per_query}
        rows.append(row)
    return pd.DataFrame(rows), all_ranks


def by_question_type(all_ranks: dict[str, np.ndarray], queries: Queries, k: int = 1) -> pd.DataFrame:
    qtypes = np.asarray(queries.qtypes)
    rows = []
    for qtype in sorted(set(queries.qtypes), key=lambda t: -(qtypes == t).sum()):
        mask = qtypes == qtype
        row = {"Question type": qtype, "n": int(mask.sum())}
        for name, ranks in all_ranks.items():
            row[name] = float((ranks[mask] <= k).mean())
        rows.append(row)
    return pd.DataFrame(rows)
