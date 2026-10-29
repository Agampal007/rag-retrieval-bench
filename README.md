# rag-retrieval-bench

[![CI](https://github.com/Agampal007/rag-retrieval-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Agampal007/rag-retrieval-bench/actions/workflows/ci.yml)

A small, reproducible benchmark for the **retrieval step of a RAG pipeline**. It measures how often a retriever puts the paragraph containing the answer in the top-k results, using the 10,570 questions and 2,067 paragraphs of the SQuAD v1.1 dev set.

It compares five retrievers behind one two-method interface (`fit`, `score`), all implemented from scratch on NumPy/SciPy/scikit-learn, with no model downloads and no API keys:

| Retriever | What it is |
| --- | --- |
| TF-IDF | Cosine similarity over sublinear TF-IDF vectors |
| TF-IDF + bigrams | Same, with unigrams and bigrams |
| BM25 | Okapi BM25 (Lucene-style IDF), written with sparse matrix algebra |
| LSA | TF-IDF, then 256-dimensional truncated SVD: a model-free stand-in for dense retrieval |
| BM25 + LSA (RRF) | Reciprocal Rank Fusion of BM25 and LSA |

A second experiment adds a neural embedding model (`all-MiniLM-L6-v2`) and two fusion methods; see [Dense retrieval and fusion](#dense-retrieval-and-fusion).

## Results

Single relevant paragraph per question, 10,570 questions, ranking over all 2,067 paragraphs.

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF | 0.707 | 0.854 | 0.891 | 0.931 | 0.788 | 0.823 |
| TF-IDF + bigrams | 0.725 | 0.859 | 0.893 | 0.932 | 0.798 | 0.831 |
| **BM25** | **0.748** | **0.874** | **0.906** | **0.938** | **0.817** | **0.847** |
| LSA | 0.507 | 0.695 | 0.766 | 0.851 | 0.617 | 0.673 |
| BM25 + LSA (RRF) | 0.574 | 0.792 | 0.847 | 0.914 | 0.695 | 0.748 |

![Recall@k curves for the five retrievers](results/recall_at_k.png)

### What the numbers say

- **BM25 is the strongest retriever here**, ahead of TF-IDF by 4.1 points of Recall@1 and 2.9 points of MRR@10. Bigrams help TF-IDF a little (+1.8 points Recall@1) but do not close the gap.
- **Fusion with a weaker retriever hurt, and how you fuse matters.** Adding LSA to BM25 through RRF lowered Recall@1 from 0.748 to 0.574, and every non-zero LSA weight I tried scored below BM25 alone. Fusing z-scored scores instead degrades much more gently (0.746 at LSA weight 0.1, 0.718 at equal weights) but still never beats BM25 alone (`results/fusion_sweep.md`). Rank-based fusion throws away how far apart documents are in each retriever's scores, so a clearly weaker ranker mostly adds noise. A neural embedding model behaved very differently from LSA; see [Dense retrieval and fusion](#dense-retrieval-and-fusion).
- **Most misses are the right article, wrong paragraph.** Of BM25's 2,665 Recall@1 misses, 75.8% have a top-1 paragraph from the correct article, and only 3.6% have a gold paragraph with no indexed term in common with the question (`results/bm25_miss_analysis.md`). The hard part of this task is choosing between neighbouring paragraphs on the same topic, not finding the topic.
- **Question type barely matters for lexical retrieval.** BM25 Recall@1 stays between 0.709 (`where`) and 0.753 (`when`, `how`) across question types with 400+ examples (`results/recall1_by_question_type.md`).
- **BM25 parameters matter less than the choice of BM25.** Across a 5x5 grid, MRR@10 ranges from 0.779 to 0.829 (default k1=1.5, b=0.75 gives 0.817). Switching length normalisation off (b=0) is consistently the worst setting.

![BM25 MRR@10 across k1 and b](results/bm25_grid.png)

## Dense retrieval and fusion

I added a neural embedding model, `all-MiniLM-L6-v2` (sentence-transformers, run locally), and fused it with BM25 in two ways: by ranks (RRF) and by z-scored scores at several dense weights. Same 10,570 questions and 2,067 paragraphs. Full output, including timings, is in `results/dense_all-MiniLM-L6-v2.md`.

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.748 | 0.874 | 0.906 | 0.938 | 0.817 | 0.847 |
| all-MiniLM-L6-v2 | 0.625 | 0.808 | 0.864 | 0.917 | 0.727 | 0.773 |
| BM25 + MiniLM, RRF (ranks) | 0.699 | 0.885 | 0.926 | 0.960 | 0.798 | 0.838 |
| BM25 + MiniLM, score fusion (dense x0.25) | 0.763 | 0.888 | 0.917 | 0.949 | 0.831 | 0.860 |
| BM25 + MiniLM, score fusion (dense x0.5) | 0.771 | 0.897 | 0.926 | 0.955 | 0.840 | 0.868 |
| BM25 + MiniLM, score fusion (dense x1, equal weights) | 0.784 | 0.908 | 0.938 | 0.963 | 0.852 | 0.879 |
| BM25 + MiniLM, score fusion (dense x2) | 0.794 | 0.920 | 0.948 | 0.971 | 0.861 | 0.888 |

- **On its own, MiniLM is weaker than BM25 here** (Recall@1 0.625 against 0.748). SQuAD questions share many words with their paragraph, which favours lexical matching.
- **Score fusion beat BM25 at every weight I tried**, and kept improving as the dense weight grew. With equal weights and no tuning, Recall@1 rose from 0.748 to 0.784, MRR@10 from 0.817 to 0.852 and Recall@10 from 0.938 to 0.963. At dense x2, MRR@10 reached 0.861 and Recall@10 0.971.
- **Rank fusion (RRF) was mixed.** It raised deeper recall (Recall@5 0.926 against 0.906, Recall@10 0.960 against 0.938) but lowered Recall@1 (0.699 against 0.748) and MRR@10 (0.798 against 0.817). RRF with LSA lowered every metric, including deeper recall, so the neural model is a better partner even under rank fusion.
- **LSA did not help in fusion at any weight; the neural model helped at every weight.** Both are weaker than BM25 alone, so being weaker is not what decides it. This experiment does not show what does; how different the two retrievers' mistakes are is a plausible cause I did not test.

To reproduce:

```bash
pip install -e ".[dense]"
python scripts/download_data.py  # once, if data/ is missing
python scripts/run_dense.py     # downloads all-MiniLM-L6-v2 (about 90 MB)
```

## Read this before quoting the numbers

- **The corpus is small and the questions are easy for lexical search.** SQuAD annotators wrote each question while looking at its paragraph, so questions share many words with their answer paragraph. These scores are not comparable to BEIR or MS MARCO, and they say little about how BM25 would do on a large, noisy corpus.
- **LSA is not a neural dense retriever.** It is a cheap latent-semantic baseline. The neural model is evaluated separately in the next section.
- **The dense experiment uses one model on one dataset.** The fusion weight sweep stopped at a dense weight of 2 while results were still improving, so the best weight was not found, and weights were not chosen on held-out data. The `ms/query` column in `results/dense_all-MiniLM-L6-v2.md` (measured on my own PC) is not comparable across rows: after the dense row, query embeddings are cached, so the fusion rows leave out encoding time.
- **The k1/b grid and the fusion-weight sweeps are evaluated on the same questions as the headline table.** Treat the best cells as optimistic. The headline BM25 row uses the default parameters, not the tuned ones.
- **Ties are scored pessimistically.** If the gold paragraph ties with other paragraphs (for example, all-zero scores), it is ranked below all of them.
- **One gold paragraph per question.** A different paragraph that also answers the question is counted as a miss.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/download_data.py      # SQuAD v1.1 dev, checksum-verified
python -m ragbench.run               # tables, CSVs and ranks in ./results
python -m ragbench.plots             # figures in ./results
python scripts/miss_analysis.py      # BM25 failure breakdown
pytest                               # 21 unit tests
```

`python -m ragbench.run --limit 500 --skip-grid` gives a quick sanity run. All numbers are deterministic (fixed seeds, fixed dataset checksum).

## Method

- **Documents**: every SQuAD paragraph (2,067). **Queries**: every question (10,570). **Gold**: the paragraph the question was written about.
- **Tokenisation**: lowercase, `\b\w+\b`, English stop words removed.
- **BM25**: `idf = ln(1 + (N - df + 0.5) / (df + 0.5))`, term weight `tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avglen))`. Verified against a hand-computed score in `tests/test_retrievers.py`.
- **RRF**: `score(d) = sum_r w_r / (60 + rank_r(d))`.
- **Score fusion**: `score(d) = sum_r w_r * z_r(d)`, where `z_r` standardises retriever r's scores across documents for each query.
- **Dense retriever**: cosine similarity between embeddings from any `encode(list[str]) -> array` function. Embeddings are cached by text, so sweeps do not re-encode.
- **Metrics**: with one relevant document per query, everything is a function of its 1-based rank: Recall@k = share of queries with rank <= k, MRR@10 = mean of 1/rank (0 beyond 10), nDCG@10 = mean of 1/log2(rank + 1) (0 beyond 10).

## Layout

```
src/ragbench/
  data.py        SQuAD loader, question-type tagging
  retrievers.py  TF-IDF, BM25, LSA, dense, RRF and score fusion behind fit()/score()
  dense_experiment.py  BM25 vs dense vs fusion for any embedding function
  metrics.py     rank-based Recall@k, MRR@k, nDCG@k
  evaluate.py    benchmark loop, per-question-type breakdown
  run.py         CLI: all tables and CSVs
  plots.py       figures
scripts/         dataset download, BM25 miss analysis, sentence-transformers experiment
tests/           21 unit tests (metrics, retrievers, fusion, dense, data loader)
results/         committed outputs from the run described above
```

## Roadmap

- Choose the fusion weight on a held-out split (the sweep was still improving at dense x2) and try stronger embedding models.
- Add a cross-encoder reranker on the top 20 and measure the lift on the "right article, wrong paragraph" misses.
- Chunk-size ablation: sentence windows aggregated to paragraph scores.
- Run the same harness on a larger corpus (for example a BEIR dataset) to see whether the ranking of retrievers holds.

## License

MIT
