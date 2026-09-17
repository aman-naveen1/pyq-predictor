"""Exploratory and unsupervised analysis of PYQs."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

from .preprocessing import build_tfidf


def question_similarity(questions: Sequence[dict], query: str, top_n: int = 5) -> list[dict]:
    """Return historical questions most similar to a supplied question."""
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [q.get("clean_text", q.get("text", "")) for q in questions]
    if not texts or not query.strip():
        return []
    matrix, vectorizer = build_tfidf(texts)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    order = np.argsort(scores)[::-1][:top_n]
    return [{**questions[i], "similarity": float(scores[i])} for i in order]


def cluster_questions(questions: Sequence[dict], n_clusters: int = 5) -> list[dict]:
    """Cluster questions with K-Means and return topic labels plus keywords."""
    from sklearn.cluster import KMeans

    if len(questions) < 2:
        return []
    texts = [q.get("clean_text", q.get("text", "")) for q in questions]
    matrix, vectorizer = build_tfidf(texts)
    k = max(1, min(n_clusters, len(questions)))
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)
    terms = np.array(vectorizer.get_feature_names_out())
    topics = []
    for cluster_id in range(k):
        center = model.cluster_centers_[cluster_id]
        top_terms = terms[np.argsort(center)[-6:][::-1]].tolist()
        topics.append({"cluster": cluster_id + 1, "label": ", ".join(top_terms[:3]), "keywords": top_terms})
    enriched = []
    for q, label in zip(questions, labels):
        enriched.append({**q, "cluster": int(label) + 1, "topic": topics[int(label)]["label"]})
    return enriched


def topic_summary(questions: Sequence[dict]) -> list[dict]:
    """Aggregate cluster/topic recurrence across distinct papers."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        topic = q.get("topic") or q.get("clean_text", "")[:50] or "Unknown"
        grouped[topic].append(q)

    rows = []
    years = [q.get("year") for q in questions if q.get("year")]
    max_year = max(years) if years else None
    for topic, items in grouped.items():
        paper_years = sorted({q.get("year") for q in items if q.get("year")})
        frequency = len(items)
        paper_count = len(paper_years) if paper_years else frequency
        recency = 1.0
        if max_year and paper_years:
            recency = 1 / (1 + (max_year - max(paper_years)))
        score = 100 * (0.55 * min(paper_count / max(1, len({q.get('year') for q in questions if q.get('year')})), 1) + 0.45 * recency)
        rows.append({
            "topic": topic,
            "frequency": frequency,
            "papers": paper_count,
            "years": ", ".join(map(str, paper_years)) if paper_years else "Unknown",
            "recurrence_score": round(float(score), 2),
        })
    return sorted(rows, key=lambda x: x["recurrence_score"], reverse=True)
