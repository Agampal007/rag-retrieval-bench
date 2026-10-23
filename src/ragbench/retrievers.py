"""First-stage retrievers behind one tiny interface.

Every retriever has:
    name: str
    fit(docs: list[str]) -> None
    score(queries: list[str]) -> np.ndarray   # shape (n_queries, n_docs), higher is better

A dense embedding model can be added by implementing the same two methods.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

_TOKEN_PATTERN = r"(?u)\b\w+\b"


def ordinal_ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every document for every query (ties broken by position)."""
    n_queries, n_docs = scores.shape
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    positions = np.broadcast_to(np.arange(1, n_docs + 1), (n_queries, n_docs))
    np.put_along_axis(ranks, order, positions, axis=1)
    return ranks


class BM25Retriever:
    """Okapi BM25 with Lucene-style IDF, computed with sparse matrix algebra."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, stopwords: bool = True, name: str | None = None):
        self.k1 = k1
        self.b = b
        self.stopwords = stopwords
        self.name = name or "BM25"

    def fit(self, docs: list[str]) -> None:
        self._vec = CountVectorizer(
            lowercase=True,
            token_pattern=_TOKEN_PATTERN,
            stop_words="english" if self.stopwords else None,
        )
        tf = self._vec.fit_transform(docs).astype(np.float64).tocoo()
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = doc_len.mean()
        df = np.bincount(tf.col, minlength=tf.shape[1])
        self._idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        norm = self.k1 * (1.0 - self.b + self.b * doc_len[tf.row] / avg_len)
        weights = tf.data * (self.k1 + 1.0) / (tf.data + norm)
        self._weights = sparse.csr_matrix((weights, (tf.row, tf.col)), shape=tf.shape)

    def score(self, queries: list[str]) -> np.ndarray:
        q = self._vec.transform(queries).astype(np.float64)
        q = q @ sparse.diags(self._idf)
        return (q @ self._weights.T).toarray()


class TfidfRetriever:
    """Cosine similarity over sublinear TF-IDF vectors."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 1), stopwords: bool = True, name: str | None = None):
        self.ngram_range = ngram_range
        self.stopwords = stopwords
        self.name = name or "TF-IDF"

    def fit(self, docs: list[str]) -> None:
        self._vec = TfidfVectorizer(
            lowercase=True,
            token_pattern=_TOKEN_PATTERN,
            stop_words="english" if self.stopwords else None,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
        )
        self._docs = self._vec.fit_transform(docs)

    def score(self, queries: list[str]) -> np.ndarray:
        q = self._vec.transform(queries)
        return (q @ self._docs.T).toarray()


class LSARetriever:
    """Latent semantic analysis: TF-IDF -> truncated SVD -> cosine in the latent space.

    A cheap, model-free stand-in for a dense retriever. It can match a query to a
    paragraph that shares no exact words with it, but only through co-occurrence
    statistics learned from the corpus itself.
    """

    def __init__(self, n_components: int = 256, stopwords: bool = True, name: str | None = None, seed: int = 0):
        self.n_components = n_components
        self.stopwords = stopwords
        self.seed = seed
        self.name = name or f"LSA ({n_components}d)"

    def fit(self, docs: list[str]) -> None:
        self._vec = TfidfVectorizer(
            lowercase=True,
            token_pattern=_TOKEN_PATTERN,
            stop_words="english" if self.stopwords else None,
            sublinear_tf=True,
        )
        tfidf = self._vec.fit_transform(docs)
        n_components = min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        self._docs = normalize(self._svd.fit_transform(tfidf))

    def score(self, queries: list[str]) -> np.ndarray:
        q = normalize(self._svd.transform(self._vec.transform(queries)))
        return q @ self._docs.T


class RRFHybrid:
    """Reciprocal Rank Fusion: score(d) = sum_r w_r / (k + rank_r(d)). Weights default to 1."""

    def __init__(self, retrievers: list, k: int = 60, weights: list[float] | None = None, name: str | None = None):
        self.retrievers = retrievers
        self.k = k
        self.weights = list(weights) if weights is not None else [1.0] * len(retrievers)
        if len(self.weights) != len(retrievers):
            raise ValueError("weights must have one entry per retriever")
        self.name = name or "Hybrid RRF (" + " + ".join(r.name for r in retrievers) + ")"

    def fit(self, docs: list[str]) -> None:
        for retriever in self.retrievers:
            retriever.fit(docs)

    def score(self, queries: list[str]) -> np.ndarray:
        total = None
        for retriever, weight in zip(self.retrievers, self.weights):
            contribution = weight / (self.k + ordinal_ranks(retriever.score(queries)))
            total = contribution if total is None else total + contribution
        return total
