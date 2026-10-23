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


def zscore_rows(scores: np.ndarray) -> np.ndarray:
    """Standardise each query's scores across documents (mean 0, std 1)."""
    centred = scores - scores.mean(axis=1, keepdims=True)
    return centred / (scores.std(axis=1, keepdims=True) + 1e-12)


class DenseRetriever:
    """Cosine similarity between embeddings from any `encode(list[str]) -> np.ndarray` callable.

    Embeddings are memoised by text, so refitting or re-scoring the same documents and
    queries (for example, in a fusion-weight sweep) does not call the encoder again.
    """

    def __init__(self, encode, name: str = "Dense"):
        self.encode = encode
        self.name = name
        self._cache: dict[str, np.ndarray] = {}

    def _embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for text, vector in zip(missing, normalize(np.asarray(self.encode(missing), dtype=np.float64))):
                self._cache[text] = vector
        return np.stack([self._cache[t] for t in texts])

    def fit(self, docs: list[str]) -> None:
        self._docs = self._embed(docs)

    def score(self, queries: list[str]) -> np.ndarray:
        return self._embed(queries) @ self._docs.T


class SentenceTransformerRetriever(DenseRetriever):
    """DenseRetriever backed by a sentence-transformers model (pip install sentence-transformers)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "SentenceTransformerRetriever needs the optional dependency: pip install sentence-transformers"
            ) from error
        model = SentenceTransformer(model_name)

        def encode(texts: list[str]) -> np.ndarray:
            return model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)

        super().__init__(encode, name=model_name.split("/")[-1])


class ScoreFusion:
    """Weighted sum of per-query z-scored retriever scores.

    Unlike rank fusion, this keeps how far apart documents are in each retriever's own scores.
    """

    def __init__(self, retrievers: list, weights: list[float] | None = None, name: str | None = None):
        self.retrievers = retrievers
        self.weights = list(weights) if weights is not None else [1.0] * len(retrievers)
        if len(self.weights) != len(retrievers):
            raise ValueError("weights must have one entry per retriever")
        self.name = name or "Score fusion (" + " + ".join(r.name for r in retrievers) + ")"

    def fit(self, docs: list[str]) -> None:
        for retriever in self.retrievers:
            retriever.fit(docs)

    def score(self, queries: list[str]) -> np.ndarray:
        total = None
        for retriever, weight in zip(self.retrievers, self.weights):
            contribution = weight * zscore_rows(retriever.score(queries))
            total = contribution if total is None else total + contribution
        return total


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
