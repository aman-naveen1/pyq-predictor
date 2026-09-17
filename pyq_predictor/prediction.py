"""Transparent, history-based recurrence scoring."""

from __future__ import annotations

import math
from collections import defaultdict


def recurrence_scores(questions: list[dict]) -> list[dict]:
    """Rank question topics using frequency and recency.

    This is an interpretable heuristic, not a claim of guaranteed exam prediction.
    """
    if not questions:
        return []

    years = sorted({q.get("year") for q in questions if q.get("year")})
    max_year = max(years) if years else None
    by_topic = defaultdict(list)
    for q in questions:
        topic = q.get("topic") or q.get("clean_text", "")[:60] or "Unknown"
        by_topic[topic].append(q)

    result = []
    total_papers = max(1, len(years))
    for topic, items in by_topic.items():
        topic_years = {q.get("year") for q in items if q.get("year")}
        frequency = len(items)
        coverage = len(topic_years) / total_papers if years else min(frequency / 10, 1)
        last_seen = max(topic_years) if topic_years else None
        gap = (max_year - last_seen) if max_year and last_seen else None
        recency = math.exp(-0.35 * gap) if gap is not None else 0.5
        score = 100 * (0.60 * coverage + 0.40 * recency)
        category = "High recurrence" if score >= 70 else "Moderate recurrence" if score >= 45 else "Lower recurrence"
        result.append({
            "topic": topic,
            "frequency": frequency,
            "paper_coverage": round(coverage * 100, 1),
            "last_seen": last_seen,
            "recurrence_score": round(score, 1),
            "category": category,
        })
    return sorted(result, key=lambda x: x["recurrence_score"], reverse=True)
