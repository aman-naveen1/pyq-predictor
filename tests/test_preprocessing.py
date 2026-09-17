from pyq_predictor.preprocessing import clean_text, prepare_questions


def test_clean_text_normalizes_noise():
    assert clean_text("Q1. Explain 2NF | 3NF") == "explain 2nf 3nf"


def test_prepare_questions_keeps_original():
    result = prepare_questions([{"text": "Explain ACID properties"}])
    assert result[0]["text"] == "Explain ACID properties"
    assert result[0]["clean_text"] == "explain acid properties"
