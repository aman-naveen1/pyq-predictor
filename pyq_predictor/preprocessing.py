"""NLP preprocessing utilities for noisy question-paper text."""

from __future__ import annotations

import re
from typing import Sequence


def clean_text(text: str) -> str:
    """Normalize common OCR noise and return lowercase text."""
    text = text.lower()
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"\bq(?:uestion)?\s*\d{1,2}\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_questions(questions: Sequence[dict]) -> list[dict]:
    """Add cleaned text while preserving the original question text."""
    prepared = []
    for question in questions:
        item = dict(question)
        item["clean_text"] = clean_text(str(question.get("text", "")))
        if item["clean_text"]:
            prepared.append(item)
    return prepared


def build_tfidf(texts: Sequence[str], max_features: int = 3000):
    """Build a TF-IDF matrix and fitted vectorizer."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer
