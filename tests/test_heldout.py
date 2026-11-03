import numpy as np

from ragbench.data import Corpus, Queries
from ragbench.evaluate import run_benchmark
from ragbench.heldout import choose_weight, fusion_ranks, split_by_article
from ragbench.retrievers import BM25Retriever, DenseRetriever, ScoreFusion

from test_dense_and_fusion import DOCS, CountingEncoder

QUESTIONS = [
    "Which ocean is the deepest?",
    "What is the capital of France?",
    "What produces ATP in the cell?",
    "What converts sunlight into energy?",
    "Which river runs through Paris?",
]
GOLD = np.array([2, 1, 0, 3, 1])


def _benchmark(titles):
    corpus = Corpus(titles=titles, texts=DOCS)
    queries = Queries(
        ids=[f"q{i}" for i in range(len(QUESTIONS))],
        texts=QUESTIONS,
        gold=GOLD,
        answers=[""] * len(QUESTIONS),
        qtypes=["what"] * len(QUESTIONS),
    )
    return corpus, queries


def test_split_keeps_each_article_on_one_side_and_covers_every_question():
    corpus, queries = _benchmark(titles=["Biology", "France", "Oceans", "Biology"])
    tune, test = split_by_article(corpus, queries, seed=3)

    assert sorted(np.concatenate([tune, test]).tolist()) == list(range(len(QUESTIONS)))
    tune_titles = {corpus.titles[g] for g in GOLD[tune]}
    test_titles = {corpus.titles[g] for g in GOLD[test]}
    assert tune_titles and test_titles and not tune_titles & test_titles


def test_split_is_deterministic_for_a_seed():
    corpus, queries = _benchmark(titles=["Biology", "France", "Oceans", "Biology"])
    first = split_by_article(corpus, queries, seed=7)
    second = split_by_article(corpus, queries, seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_fusion_ranks_match_score_fusion_for_every_weight():
    corpus, queries = _benchmark(titles=["t"] * len(DOCS))
    weights = (0.0, 0.5, 2.0)
    ranks = fusion_ranks(
        BM25Retriever(), DenseRetriever(CountingEncoder()), corpus.texts, queries, weights, batch_size=2
    )
    for w in weights:
        fusion = ScoreFusion([BM25Retriever(), DenseRetriever(CountingEncoder())], weights=[1.0, w], name="f")
        _, expected = run_benchmark([fusion], corpus, queries)
        assert np.array_equal(ranks[w], expected["f"]), w


def test_choose_weight_picks_the_best_metric_and_breaks_ties_low():
    ranks = {
        0.0: np.array([2, 2, 1, 1]),
        1.0: np.array([1, 1, 1, 1]),
        2.0: np.array([1, 1, 1, 1]),
        3.0: np.array([1, 1, 5, 5]),
    }
    assert choose_weight(ranks, np.arange(4)) == 1.0
    assert choose_weight(ranks, np.array([0, 1])) == 1.0
    assert choose_weight(ranks, np.array([2, 3])) == 0.0
