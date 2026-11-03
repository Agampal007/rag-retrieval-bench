Tune split: 5364 questions from 24 articles. Test split: 5206 questions from 24 articles. Weight chosen by MRR@10 on the tune split: x2.

## Test split

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 alone | 0.740 | 0.876 | 0.910 | 0.939 | 0.815 | 0.845 |
| all-MiniLM-L6-v2 alone | 0.618 | 0.805 | 0.864 | 0.918 | 0.723 | 0.770 |
| Fusion, equal weights (x1) | 0.775 | 0.910 | 0.939 | 0.963 | 0.847 | 0.876 |
| Fusion, weight chosen on tune split (x2) | 0.785 | 0.920 | 0.948 | 0.971 | 0.857 | 0.885 |
| Fusion, best weight on test split (x3, optimistic) | 0.784 | 0.921 | 0.952 | 0.973 | 0.857 | 0.886 |

## Sweep

| Dense weight | Tune MRR@10 | Test MRR@10 |
| --- | --- | --- |
| 0.000 | 0.819 | 0.815 |
| 0.250 | 0.833 | 0.829 |
| 0.500 | 0.843 | 0.836 |
| 0.750 | 0.851 | 0.843 |
| 1.000 | 0.856 | 0.847 |
| 1.500 | 0.863 | 0.854 |
| 2.000 | 0.865 | 0.857 |
| 3.000 | 0.865 | 0.857 |
| 4.000 | 0.860 | 0.855 |
| 6.000 | 0.850 | 0.847 |
| 8.000 | 0.838 | 0.835 |
| 16.000 | 0.804 | 0.804 |
