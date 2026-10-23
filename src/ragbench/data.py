"""Turn SQuAD v1.1 into a paragraph-retrieval benchmark.

Each SQuAD paragraph becomes one document. Each question is a query whose single
relevant document is the paragraph its answer was annotated in.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_WH_WORDS = ("what", "who", "whom", "whose", "when", "where", "why", "how", "which")
_WH_MERGE = {"whom": "who", "whose": "who"}
_WORD = re.compile(r"[a-z]+")


def question_type(question: str) -> str:
    """Return the first wh-word in the question, or "other".

    "In what year did ..." -> "what"; "Whose ..." -> "who".
    """
    for token in _WORD.findall(question.lower()):
        if token in _WH_WORDS:
            return _WH_MERGE.get(token, token)
    return "other"


@dataclass(frozen=True)
class Corpus:
    titles: list[str]
    texts: list[str]

    def __len__(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class Queries:
    ids: list[str]
    texts: list[str]
    gold: np.ndarray  # index into Corpus of the single relevant paragraph
    answers: list[str]
    qtypes: list[str]

    def __len__(self) -> int:
        return len(self.texts)

    def subset(self, indices: np.ndarray) -> "Queries":
        idx = np.asarray(indices)
        return Queries(
            ids=[self.ids[i] for i in idx],
            texts=[self.texts[i] for i in idx],
            gold=self.gold[idx],
            answers=[self.answers[i] for i in idx],
            qtypes=[self.qtypes[i] for i in idx],
        )


def load_squad(path: str | Path) -> tuple[Corpus, Queries]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    titles: list[str] = []
    texts: list[str] = []
    ids: list[str] = []
    questions: list[str] = []
    gold: list[int] = []
    answers: list[str] = []
    qtypes: list[str] = []

    for article in raw["data"]:
        for paragraph in article["paragraphs"]:
            doc_index = len(texts)
            texts.append(paragraph["context"])
            titles.append(article["title"])
            for qa in paragraph["qas"]:
                ids.append(qa["id"])
                questions.append(qa["question"])
                gold.append(doc_index)
                answers.append(qa["answers"][0]["text"] if qa["answers"] else "")
                qtypes.append(question_type(qa["question"]))

    corpus = Corpus(titles=titles, texts=texts)
    queries = Queries(
        ids=ids,
        texts=questions,
        gold=np.asarray(gold, dtype=np.int64),
        answers=answers,
        qtypes=qtypes,
    )
    return corpus, queries
