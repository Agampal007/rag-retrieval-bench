"""BM25 versus a dense retriever, and BM25 + dense fusion, for any embedding function."""

from __future__ import annotations

import pandas as pd

from .data import Corpus, Queries
from .evaluate import run_benchmark
from .retrievers import BM25Retriever, DenseRetriever, RRFHybrid, ScoreFusion

DENSE_WEIGHTS = (0.25, 0.5, 1.0, 2.0)


def run_dense_experiment(
    encode,
    corpus: Corpus,
    queries: Queries,
    name: str = "Dense",
    weights: tuple[float, ...] = DENSE_WEIGHTS,
) -> pd.DataFrame:
    """Compare BM25, a dense retriever and their fusions.

    `encode` maps a list of strings to an (n, d) array. The dense weight is relative to a BM25
    weight of 1 in the score-fusion rows.
    """
    bm25 = BM25Retriever()
    dense = DenseRetriever(encode, name=name)
    retrievers = [
        bm25,
        dense,
        RRFHybrid([bm25, dense], name=f"BM25 + {name} (RRF)"),
    ]
    for weight in weights:
        retrievers.append(
            ScoreFusion([bm25, dense], weights=[1.0, weight], name=f"BM25 + {name} (score fusion, dense x{weight:g})")
        )
    results, _ = run_benchmark(retrievers, corpus, queries)
    return results
