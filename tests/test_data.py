import json

from ragbench.data import load_squad, question_type


def test_question_type():
    assert question_type("What is the capital of France?") == "what"
    assert question_type("In what year did it happen?") == "what"
    assert question_type("Whose idea was it?") == "who"
    assert question_type("How many people attended?") == "how"
    assert question_type("Name the largest ocean.") == "other"


def test_load_squad_maps_questions_to_their_paragraph(tmp_path):
    fixture = {
        "version": "1.1",
        "data": [
            {
                "title": "A",
                "paragraphs": [
                    {
                        "context": "First paragraph.",
                        "qas": [
                            {"id": "q1", "question": "Who wrote it?", "answers": [{"text": "Ann", "answer_start": 0}]},
                            {"id": "q2", "question": "When was it?", "answers": [{"text": "1999", "answer_start": 0}]},
                        ],
                    },
                    {
                        "context": "Second paragraph.",
                        "qas": [{"id": "q3", "question": "Where is it?", "answers": [{"text": "Rome", "answer_start": 0}]}],
                    },
                ],
            }
        ],
    }
    path = tmp_path / "squad.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    corpus, queries = load_squad(path)
    assert len(corpus) == 2 and len(queries) == 3
    assert queries.gold.tolist() == [0, 0, 1]
    assert queries.qtypes == ["who", "when", "where"]
    assert queries.answers == ["Ann", "1999", "Rome"]
    assert corpus.titles == ["A", "A"]
