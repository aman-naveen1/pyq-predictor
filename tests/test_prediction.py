from pyq_predictor.prediction import recurrence_scores


def test_empty_input():
    assert recurrence_scores([]) == []


def test_score_is_bounded():
    questions = [
        {"topic": "SQL", "year": 2024},
        {"topic": "SQL", "year": 2025},
        {"topic": "ER", "year": 2025},
    ]
    for row in recurrence_scores(questions):
        assert 0 <= row["recurrence_score"] <= 100
