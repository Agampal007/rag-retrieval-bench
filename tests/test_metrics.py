import numpy as np
import pytest

from ragbench.metrics import gold_ranks, mrr_at_k, ndcg_at_k, recall_at_k, summarize


def test_gold_ranks_basic():
    scores = np.array([[0.1, 0.9, 0.5], [0.3, 0.2, 0.8]])
    assert gold_ranks(scores, np.array([2, 0])).tolist() == [2, 2]
    assert gold_ranks(scores, np.array([1, 2])).tolist() == [1, 1]


def test_gold_ranks_ties_are_pessimistic():
    # Every document scores 0: the gold document must be ranked last, not first.
    scores = np.zeros((2, 4))
    assert gold_ranks(scores, np.array([0, 3])).tolist() == [4, 4]


def test_recall_mrr_ndcg_known_values():
    ranks = np.array([1, 2, 4, 20])
    assert recall_at_k(ranks, 1) == pytest.approx(0.25)
    assert recall_at_k(ranks, 5) == pytest.approx(0.75)
    assert mrr_at_k(ranks, 10) == pytest.approx((1 + 0.5 + 0.25 + 0) / 4)
    expected_ndcg = (1 + 1 / np.log2(3) + 1 / np.log2(5) + 0) / 4
    assert ndcg_at_k(ranks, 10) == pytest.approx(expected_ndcg)


def test_summarize_keys():
    out = summarize(np.array([1, 3, 12]))
    assert set(out) == {"Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"}
    assert out["Recall@10"] == pytest.approx(2 / 3)
