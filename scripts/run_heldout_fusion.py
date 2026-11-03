"""Choose the BM25 + dense fusion weight on half the articles and evaluate it on the other half.

    pip install -e ".[dense]"
    python scripts/run_heldout_fusion.py

Writes results/heldout_fusion.md and .csv. Uses the same model and data as scripts/run_dense.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ragbench.data import load_squad
from ragbench.heldout import choose_weight, fusion_ranks, split_by_article
from ragbench.metrics import gold_ranks, summarize
from ragbench.retrievers import BM25Retriever, SentenceTransformerRetriever
from ragbench.run import df_to_markdown

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0)
METRIC = "MRR@10"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--data", default="data/dev-v1.1.json", type=Path)
    parser.add_argument("--out", default="results", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    corpus, queries = load_squad(args.data)
    tune, test = split_by_article(corpus, queries, seed=args.seed)
    dense = SentenceTransformerRetriever(args.model)
    ranks = fusion_ranks(BM25Retriever(), dense, corpus.texts, queries, WEIGHTS)

    # Dense alone: the limit of z(BM25) + w * z(dense) as w grows, scored directly.
    dense_ranks = np.concatenate(
        [
            gold_ranks(dense.score(queries.texts[s : s + 500]), queries.gold[s : s + 500])
            for s in range(0, len(queries), 500)
        ]
    )

    chosen = choose_weight(ranks, tune, METRIC)
    oracle = choose_weight(ranks, test, METRIC)

    sweep = pd.DataFrame(
        [
            {
                "Dense weight": w,
                f"Tune {METRIC}": summarize(ranks[w][tune])[METRIC],
                f"Test {METRIC}": summarize(ranks[w][test])[METRIC],
            }
            for w in WEIGHTS
        ]
    )
    rows = [
        ("BM25 alone", ranks[0.0]),
        (dense.name + " alone", dense_ranks),
        ("Fusion, equal weights (x1)", ranks[1.0]),
        (f"Fusion, weight chosen on tune split (x{chosen:g})", ranks[chosen]),
        (f"Fusion, best weight on test split (x{oracle:g}, optimistic)", ranks[oracle]),
    ]
    test_table = pd.DataFrame([{"Retriever": name, **summarize(r[test])} for name, r in rows])

    n_tune_articles = len({corpus.titles[g] for g in queries.gold[tune]})
    n_test_articles = len({corpus.titles[g] for g in queries.gold[test]})
    report = "\n".join(
        [
            f"Tune split: {len(tune)} questions from {n_tune_articles} articles. "
            f"Test split: {len(test)} questions from {n_test_articles} articles. "
            f"Weight chosen by {METRIC} on the tune split: x{chosen:g}.",
            "",
            "## Test split",
            "",
            df_to_markdown(test_table),
            "",
            "## Sweep",
            "",
            df_to_markdown(sweep),
        ]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    test_table.to_csv(args.out / "heldout_fusion.csv", index=False)
    (args.out / "heldout_fusion.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
