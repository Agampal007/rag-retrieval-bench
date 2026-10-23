"""Run the dense-retrieval experiment with a sentence-transformers model.

    pip install -e ".[dense]"
    python scripts/run_dense.py --model sentence-transformers/all-MiniLM-L6-v2

Writes results/dense_<model>.csv and .md. Needs a one-off model download from Hugging Face.
The first run encodes 2,067 paragraphs and 10,570 questions, which takes a few minutes on CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ragbench.data import load_squad
from ragbench.dense_experiment import run_dense_experiment
from ragbench.retrievers import SentenceTransformerRetriever
from ragbench.run import df_to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--data", default="data/dev-v1.1.json", type=Path)
    parser.add_argument("--out", default="results", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N questions (quick check)")
    args = parser.parse_args()

    corpus, queries = load_squad(args.data)
    if args.limit:
        import numpy as np

        queries = queries.subset(np.arange(min(args.limit, len(queries))))

    embedder = SentenceTransformerRetriever(args.model)
    results = run_dense_experiment(embedder.encode, corpus, queries, name=embedder.name)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = embedder.name.replace("/", "_")
    results.to_csv(args.out / f"dense_{stem}.csv", index=False)
    (args.out / f"dense_{stem}.md").write_text(df_to_markdown(results) + "\n", encoding="utf-8")
    print(df_to_markdown(results))


if __name__ == "__main__":
    main()
