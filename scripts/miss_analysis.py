"""Why does BM25 miss? Split its Recall@1 failures into 'wrong paragraph, right article' and 'wrong article'.

Also counts questions whose gold paragraph shares no indexed term with the question
(pure vocabulary mismatch), which no lexical retriever can fix.

Usage: PYTHONPATH=src python scripts/miss_analysis.py --data data/dev-v1.1.json --out results/bm25_miss_analysis.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ragbench.data import load_squad
from ragbench.metrics import gold_ranks
from ragbench.retrievers import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dev-v1.1.json", type=Path)
    parser.add_argument("--out", default="results/bm25_miss_analysis.md", type=Path)
    args = parser.parse_args()

    corpus, queries = load_squad(args.data)
    bm25 = BM25Retriever()
    bm25.fit(corpus.texts)

    titles = np.asarray(corpus.titles)
    top1 = np.empty(len(queries), dtype=np.int64)
    ranks = np.empty(len(queries), dtype=np.int64)
    no_overlap = np.zeros(len(queries), dtype=bool)
    for start in range(0, len(queries), 500):
        stop = start + 500
        scores = bm25.score(queries.texts[start:stop])
        gold = queries.gold[start:stop]
        top1[start:stop] = scores.argmax(axis=1)
        ranks[start:stop] = gold_ranks(scores, gold)
        no_overlap[start:stop] = scores[np.arange(len(gold)), gold] == 0.0

    miss = ranks > 1
    same_article = titles[top1] == titles[queries.gold]
    n, n_miss = len(queries), int(miss.sum())
    lines = [
        f"BM25 Recall@1 misses: {n_miss} of {n} questions ({n_miss / n:.1%}).",
        "",
        f"- Top-1 paragraph is from the correct article (wrong paragraph): {int((miss & same_article).sum())} "
        f"({(miss & same_article).sum() / n_miss:.1%} of misses)",
        f"- Top-1 paragraph is from a different article: {int((miss & ~same_article).sum())} "
        f"({(miss & ~same_article).sum() / n_miss:.1%} of misses)",
        f"- Gold paragraph shares no indexed term with the question (BM25 score 0): {int(no_overlap.sum())} "
        f"({no_overlap.sum() / n_miss:.1%} of misses)",
        f"- Gold paragraph outside the top 10: {int((ranks > 10).sum())} questions ({(ranks > 10).mean():.1%} of all)",
    ]
    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
