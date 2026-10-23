import numpy as np
import pytest

from ragbench.metrics import gold_ranks
from ragbench.retrievers import BM25Retriever, LSARetriever, RRFHybrid, TfidfRetriever, ordinal_ranks

DOCS = [
    "The mitochondria is the powerhouse of the cell and produces ATP.",
    "Paris is the capital of France and sits on the river Seine.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Photosynthesis in plants converts sunlight into chemical energy.",
    "The Eiffel Tower in Paris was completed in 1889.",
]


def test_bm25_matches_hand_computed_score():
    # N=2, df("banana")=1, avgdl=2.5; doc0 has tf=1, dl=2.
    bm25 = BM25Retriever(k1=1.5, b=0.75, stopwords=False)
    bm25.fit(["apple banana", "apple cherry cherry"])
    idf = np.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    tf_part = 1 * (1.5 + 1) / (1 + 1.5 * (1 - 0.75 + 0.75 * 2 / 2.5))
    scores = bm25.score(["banana"])
    assert scores[0, 0] == pytest.approx(idf * tf_part)
    assert scores[0, 1] == 0.0


@pytest.mark.parametrize("retriever", [BM25Retriever(), TfidfRetriever(), TfidfRetriever(ngram_range=(1, 2))])
def test_lexical_retrievers_find_obvious_match(retriever):
    retriever.fit(DOCS)
    scores = retriever.score(["What produces ATP in the cell?", "Which ocean is the deepest?"])
    assert scores.shape == (2, len(DOCS))
    assert scores.argmax(axis=1).tolist() == [0, 2]


def test_no_overlap_query_is_not_rewarded():
    bm25 = BM25Retriever()
    bm25.fit(DOCS)
    scores = bm25.score(["zzzz qqqq"])
    assert (scores == 0).all()
    assert gold_ranks(scores, np.array([0]))[0] == len(DOCS)


def test_lsa_runs_and_returns_cosine_scores():
    lsa = LSARetriever(n_components=3)
    lsa.fit(DOCS)
    scores = lsa.score(["capital of France"])
    assert scores.shape == (1, len(DOCS))
    assert np.all(np.abs(scores) <= 1.0 + 1e-9)


def test_ordinal_ranks_are_a_permutation():
    scores = np.array([[0.2, 0.9, 0.2, 0.5]])
    ranks = ordinal_ranks(scores)
    assert sorted(ranks[0].tolist()) == [1, 2, 3, 4]
    assert ranks[0, 1] == 1 and ranks[0, 3] == 2


def test_rrf_combines_two_rankings():
    class Fixed:
        def __init__(self, name, matrix):
            self.name, self._m = name, matrix

        def fit(self, docs):
            pass

        def score(self, queries):
            return self._m

    a = Fixed("a", np.array([[3.0, 2.0, 1.0]]))  # ranks docs 0 > 1 > 2
    b = Fixed("b", np.array([[1.0, 3.0, 2.0]]))  # ranks docs 1 > 2 > 0
    hybrid = RRFHybrid([a, b], k=60)
    scores = hybrid.score(["q"])
    expected = np.array([1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62])
    assert scores[0] == pytest.approx(expected)
    assert scores.argmax() == 1
