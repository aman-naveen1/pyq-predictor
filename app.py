"""Streamlit interface for PYQ Predictor."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pyq_predictor.analysis import cluster_questions, question_similarity
from pyq_predictor.extraction import extract_pdf_text, extract_year, segment_questions
from pyq_predictor.prediction import recurrence_scores
from pyq_predictor.preprocessing import prepare_questions

st.set_page_config(page_title="PYQ Predictor", page_icon="📚", layout="wide")

st.title("📚 PYQ Predictor")
st.caption("OCR + NLP analytics for historical question papers")

with st.sidebar:
    st.header("Upload papers")
    files = st.file_uploader("Upload PDF question papers", type=["pdf"], accept_multiple_files=True)
    use_ocr = st.checkbox("Enable OCR fallback", value=True)
    n_clusters = st.slider("Number of topic clusters", 2, 10, 5)

if not files:
    st.info("Upload two or more previous-year papers to begin. For scanned papers, keep OCR enabled and install Tesseract locally.")
    st.stop()

all_questions = []
with st.spinner("Extracting and analysing papers..."):
    for uploaded in files:
        text = extract_pdf_text(uploaded, ocr=use_ocr)
        questions = segment_questions(text)
        year = extract_year(uploaded.name)
        for q in questions:
            q["year"] = year
            q["source"] = uploaded.name
        all_questions.extend(questions)

all_questions = prepare_questions(all_questions)

if len(all_questions) < 2:
    st.error("Not enough questions were extracted. Try text-based PDFs or enable OCR for scanned papers.")
    st.stop()

clustered = cluster_questions(all_questions, n_clusters=n_clusters)
scores = recurrence_scores(clustered)

c1, c2, c3 = st.columns(3)
c1.metric("Papers", len(files))
c2.metric("Questions extracted", len(all_questions))
c3.metric("Topics detected", len({q.get('topic') for q in clustered}))

st.subheader("🔮 Recurrence analysis")
st.write("Scores summarize historical frequency, paper coverage and recency. They are not guarantees about a future exam.")
score_df = pd.DataFrame(scores)
if not score_df.empty:
    st.dataframe(score_df, use_container_width=True, hide_index=True)
    st.bar_chart(score_df.set_index("topic")["recurrence_score"])

st.subheader("🔎 Find similar previous questions")
query = st.text_area("Enter a question", placeholder="e.g. Explain ACID properties in DBMS")
if query.strip():
    matches = question_similarity(clustered, query, top_n=5)
    for i, match in enumerate(matches, 1):
        year = match.get("year") or "Year unknown"
        st.markdown(f"**{i}. {year} — {match.get('similarity', 0) * 100:.1f}% similarity**")
        st.write(match.get("text", ""))

st.subheader("🧠 Detected topic clusters")
cluster_df = pd.DataFrame([
    {"Cluster": q.get("cluster"), "Topic": q.get("topic"), "Year": q.get("year"), "Question": q.get("text")}
    for q in clustered
])
st.dataframe(cluster_df, use_container_width=True, hide_index=True)

with st.expander("How it works"):
    st.markdown("""
    1. **Extraction:** text is read directly from PDFs; OCR is used when a page has no extractable text.
    2. **Cleaning:** common OCR artifacts and question labels are normalized.
    3. **TF-IDF:** questions are converted into numerical text vectors.
    4. **Similarity:** cosine similarity finds historically related questions.
    5. **Clustering:** K-Means groups questions with similar vocabulary.
    6. **Recurrence scoring:** frequency, cross-paper coverage and recency are combined into an interpretable score.
    """)
