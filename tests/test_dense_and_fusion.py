import sys

import numpy as np
import pandas as pd
import pytest

from ragbench.data import Corpus, Queries
from ragbench.dense_experiment import run_dense_experiment
from ragbench.retrievers import (
    BM25Retriever,
    DenseRetriever,
    ScoreFusion,
    SentenceTransformerRetriever,
    zscore_rows,
)

DOCS = [
    "The mitochondria is the powerhouse of the cell and produces ATP.",
    "Paris is the capital of France and sits on the river Seine.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Photosynthesis in plants converts sunlight into chemical energy.",
]
VOCAB = sorted({w for d in DOCS for w in d.lower().replace(".", "").split()})


class CountingEncoder:
    """Deterministic bag-of-words encoder that records every text it is asked to embed."""

    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, texts):
        self.seen.extend(texts)
        out = np.zeros((len(texts), len(VOCAB)))
        for i, text in enumerate(texts):
            for word in text.lower().replace("?", "").replace(".", "").split():
                if word in VOCAB:
                    out[i, VOCAB.index(word)] += 1.0
        return out


def test_dense_retriever_ranks_the_matching_document_first():
    dense = DenseRetriever(CountingEncoder())
    dense.fit(DOCS)
    scores = dense.score(["Which ocean is the deepest?", "What is the capital of France?"])
    assert scores.argmax(axis=1).tolist() == [2, 1]
    assert np.all(np.abs(scores) <= 1.0 + 1e-9)


def test_dense_retriever_encodes_each_text_once():
    encoder = CountingEncoder()
    dense = DenseRetriever(encoder)
    dense.fit(DOCS)
    dense.fit(DOCS)
    dense.score(["capital of France"])
    dense.score(["capital of France"])
    assert len(encoder.seen) == len(DOCS) + 1


def test_zscore_rows_standardises_each_query():
    scores = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 40.0]])
    z = zscore_rows(scores)
    assert z.mean(axis=1) == pytest.approx([0.0, 0.0])
    assert z.std(axis=1) == pytest.approx([1.0, 1.0])


def test_score_fusion_with_zero_weight_matches_first_retriever():
    bm25 = BM25Retriever()
    dense = DenseRetriever(CountingEncoder())
    fusion = ScoreFusion([bm25, dense], weights=[1.0, 0.0])
    fusion.fit(DOCS)
    queries = ["Which ocean is the deepest?", "powerhouse of the cell"]
    assert np.allclose(fusion.score(queries), zscore_rows(bm25.score(queries)))


def test_score_fusion_rejects_wrong_number_of_weights():
    with pytest.raises(ValueError):
        ScoreFusion([BM25Retriever(), BM25Retriever()], weights=[1.0])


def test_dense_experiment_returns_a_row_per_retriever():
    corpus = Corpus(titles=["t"] * len(DOCS), texts=DOCS)
    queries = Queries(
        ids=["a", "b"],
        texts=["Which ocean is the deepest?", "What is the capital of France?"],
        gold=np.array([2, 1]),
        answers=["", ""],
        qtypes=["which", "what"],
    )
    results = run_dense_experiment(CountingEncoder(), corpus, queries, name="Toy", weights=(0.5, 1.0))
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 3 + 2  # BM25, dense, RRF, plus one score-fusion row per weight
    assert results.loc[results["Retriever"] == "BM25", "Recall@1"].item() == 1.0


def test_sentence_transformer_retriever_explains_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="pip install sentence-transformers"):
        SentenceTransformerRetriever()
