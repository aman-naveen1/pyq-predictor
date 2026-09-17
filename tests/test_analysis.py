from pyq_predictor.analysis import question_similarity
from pyq_predictor.prediction import recurrence_scores
from pyq_predictor.preprocessing import prepare_questions


def test_similarity_returns_related_question_first():
    questions = prepare_questions([
        {"text": "Explain ACID properties in transactions", "year": 2025},
        {"text": "Explain normalization and functional dependency", "year": 2024},
    ])
    result = question_similarity(questions, "What are ACID transaction properties?", top_n=1)
    assert result[0]["year"] == 2025


def test_recurrence_scores_are_sorted():
    questions = [
        {"text": "Normalization", "clean_text": "normalization", "topic": "normalization", "year": 2023},
        {"text": "Normalization", "clean_text": "normalization", "topic": "normalization", "year": 2024},
        {"text": "ACID", "clean_text": "acid", "topic": "acid", "year": 2024},
    ]
    result = recurrence_scores(questions)
    assert result[0]["topic"] == "normalization"
    assert result[0]["recurrence_score"] >= result[1]["recurrence_score"]
