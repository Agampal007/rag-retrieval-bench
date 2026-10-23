"""Command line entry point: python -m ragbench.run --data data/dev-v1.1.json --out results"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_squad
from .evaluate import by_question_type, run_benchmark
from .metrics import mrr_at_k
from .evaluate import rank_gold
from .retrievers import BM25Retriever, LSARetriever, RRFHybrid, ScoreFusion, TfidfRetriever

K1_GRID = (0.5, 0.9, 1.2, 1.5, 2.0)
B_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def default_retrievers() -> list:
    return [
        TfidfRetriever(name="TF-IDF"),
        TfidfRetriever(ngram_range=(1, 2), name="TF-IDF + bigrams"),
        BM25Retriever(name="BM25"),
        LSARetriever(n_components=256, name="LSA"),
        RRFHybrid([BM25Retriever(), LSARetriever(n_components=256)], name="BM25 + LSA (RRF)"),
    ]


def df_to_markdown(df: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    def fmt(value) -> str:
        return float_format.format(value) if isinstance(value, (float, np.floating)) else str(value)

    header = "| " + " | ".join(df.columns) + " |"
    divider = "|" + "|".join(" --- " for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *body])


def bm25_grid(corpus, queries) -> pd.DataFrame:
    rows = {}
    for k1 in K1_GRID:
        row = {}
        for b in B_GRID:
            retriever = BM25Retriever(k1=k1, b=b)
            retriever.fit(corpus.texts)
            ranks, _ = rank_gold(retriever, queries.texts, queries.gold)
            row[b] = mrr_at_k(ranks, 10)
        rows[k1] = row
    grid = pd.DataFrame(rows).T
    grid.index.name = "k1"
    grid.columns.name = "b"
    return grid


FUSION_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)


def fusion_sweep(corpus, queries) -> pd.DataFrame:
    """BM25 plus LSA at growing LSA weight, fused by ranks (RRF) and by z-scored scores."""
    bm25 = BM25Retriever()
    lsa = LSARetriever(n_components=256)
    rrf = RRFHybrid([bm25, lsa])
    score_fusion = ScoreFusion([bm25, lsa])
    rrf.fit(corpus.texts)  # fits the shared BM25 and LSA objects once
    rows = []
    for method, hybrid in (("RRF (ranks)", rrf), ("Score fusion (z-scores)", score_fusion)):
        for weight in FUSION_WEIGHTS:
            hybrid.weights = [1.0, weight]
            ranks, _ = rank_gold(hybrid, queries.texts, queries.gold)
            rows.append(
                {
                    "Fusion": method,
                    "LSA weight": weight,
                    "Recall@1": float((ranks <= 1).mean()),
                    "MRR@10": mrr_at_k(ranks, 10),
                }
            )
    return pd.DataFrame(rows)


def bm25_failures(ranks: np.ndarray, corpus, queries, n: int = 30, seed: int = 0) -> pd.DataFrame:
    missed = np.flatnonzero(ranks > 10)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(missed, size=min(n, len(missed)), replace=False)
    chosen.sort()
    return pd.DataFrame(
        {
            "question": [queries.texts[i] for i in chosen],
            "answer": [queries.answers[i] for i in chosen],
            "question_type": [queries.qtypes[i] for i in chosen],
            "gold_rank": ranks[chosen],
            "article": [corpus.titles[queries.gold[i]] for i in chosen],
            "gold_paragraph_start": [corpus.texts[queries.gold[i]][:160] for i in chosen],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dev-v1.1.json", type=Path)
    parser.add_argument("--out", default="results", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N queries (quick check)")
    parser.add_argument("--skip-grid", action="store_true", help="skip the BM25 k1/b sensitivity grid")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    corpus, queries = load_squad(args.data)
    if args.limit:
        queries = queries.subset(np.arange(min(args.limit, len(queries))))
    print(f"{len(corpus)} paragraphs, {len(queries)} queries")

    results, all_ranks = run_benchmark(default_retrievers(), corpus, queries)
    results.to_csv(args.out / "main_results.csv", index=False)
    (args.out / "main_results.md").write_text(df_to_markdown(results) + "\n", encoding="utf-8")
    print(df_to_markdown(results))

    by_type = by_question_type(all_ranks, queries, k=1)
    by_type.to_csv(args.out / "recall1_by_question_type.csv", index=False)
    (args.out / "recall1_by_question_type.md").write_text(df_to_markdown(by_type) + "\n", encoding="utf-8")
    print()
    print(df_to_markdown(by_type))

    failures = bm25_failures(all_ranks["BM25"], corpus, queries)
    failures.to_csv(args.out / "bm25_failures_sample.csv", index=False)

    if not args.skip_grid:
        grid = bm25_grid(corpus, queries)
        grid.to_csv(args.out / "bm25_grid_mrr10.csv")
        print()
        print(grid.round(3).to_string())

    sweep = fusion_sweep(corpus, queries)
    sweep.to_csv(args.out / "fusion_sweep.csv", index=False)
    (args.out / "fusion_sweep.md").write_text(df_to_markdown(sweep) + "\n", encoding="utf-8")
    print()
    print(df_to_markdown(sweep))

    # Ranks are saved so plots and further analysis never need to re-run retrieval.
    np.savez_compressed(args.out / "ranks.npz", **{name: ranks for name, ranks in all_ranks.items()})


if __name__ == "__main__":
    main()
