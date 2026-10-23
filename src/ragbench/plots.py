"""Render the README figures from saved results: python -m ragbench.plots --results results"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e7e6e2"
# Categorical slots, in fixed order, assigned by retriever (not by rank).
SERIES = {
    "TF-IDF": "#2a78d6",
    "TF-IDF + bigrams": "#eb6834",
    "BM25": "#1baf7a",
    "LSA": "#eda100",
    "BM25 + LSA (RRF)": "#e87ba4",
}
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def recall_curves(ranks_path: Path, out: Path, max_k: int = 20) -> None:
    ranks = np.load(ranks_path)
    ks = np.arange(1, max_k + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    _style(ax)
    for name, color in SERIES.items():
        recall = [(ranks[name] <= k).mean() for k in ks]
        ax.plot(ks, recall, color=color, linewidth=2, label=name)
    ax.set_xlim(1, max_k)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks([1, 3, 5, 10, 15, 20])
    ax.set_xlabel("k (number of paragraphs retrieved)", color=INK_2)
    ax.set_ylabel("Recall@k", color=INK_2)
    ax.set_title("Recall@k on SQuAD v1.1 dev (10,570 questions, 2,067 paragraphs)", color=INK, loc="left", fontsize=11)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def bm25_grid(grid_path: Path, out: Path) -> None:
    grid = pd.read_csv(grid_path, index_col=0)
    values = grid.to_numpy()
    cmap = LinearSegmentedColormap.from_list("blue_seq", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(5.6, 4.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    image = ax.imshow(values, cmap=cmap, aspect="auto", vmin=values.min(), vmax=values.max())
    ax.set_xticks(range(grid.shape[1]), [f"{float(c):g}" for c in grid.columns], color=INK_2)
    ax.set_yticks(range(grid.shape[0]), [f"{float(i):g}" for i in grid.index], color=INK_2)
    ax.set_xlabel("b (length normalisation)", color=INK_2)
    ax.set_ylabel("k1 (term-frequency saturation)", color=INK_2)
    ax.set_title("BM25 MRR@10 by k1 and b", color=INK, loc="left", fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    midpoint = (values.min() + values.max()) / 2
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            ax.text(c, r, f"{values[r, c]:.3f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if values[r, c] > midpoint else INK)
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_2, length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", type=Path)
    args = parser.parse_args()
    recall_curves(args.results / "ranks.npz", args.results / "recall_at_k.png")
    bm25_grid(args.results / "bm25_grid_mrr10.csv", args.results / "bm25_grid.png")
    print("wrote", args.results / "recall_at_k.png", "and", args.results / "bm25_grid.png")


if __name__ == "__main__":
    main()
