# PYQ Predictor 📚

A data-science project that extracts questions from previous-year question papers using OCR/NLP and ranks recurring topics and question patterns.

> **Important:** This project predicts *recurrence patterns*, not the exact future exam paper. Results are historical-data analysis and should not be treated as guaranteed predictions.

## Features

- Extract text from PDF question papers
- OCR fallback for scanned/image-based PDFs using Tesseract
- Clean and normalize noisy OCR text
- Detect question boundaries and marks where possible
- TF-IDF + cosine similarity for related questions
- Lightweight topic clustering using TF-IDF + K-Means
- Recency-weighted topic recurrence scoring
- Topic frequency and trend analysis
- Similar-question search
- Streamlit dashboard
- Unit tests and GitHub Actions CI

## Pipeline

```text
PDF / scanned paper
        ↓
Text extraction + OCR
        ↓
Question segmentation
        ↓
Text cleaning
        ↓
TF-IDF representation
        ↓
Similarity + topic clustering
        ↓
Frequency + recency analysis
        ↓
Recurrence score
        ↓
Interactive dashboard
```

## Setup

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

For OCR on scanned PDFs, install the Tesseract OCR engine separately and make sure `tesseract` is on PATH. The application will still work with text-based PDFs without Tesseract.

## Data format

You can upload one or more PDFs. The filename should ideally contain the year, for example:

```text
DBMS_2022.pdf
DBMS_2023.pdf
DBMS_2024.pdf
DBMS_2025.pdf
```

The year is extracted from the filename when possible.

## Methodology

The recurrence score combines normalized historical frequency, recency, and the number of distinct papers containing a topic. The weighting is intentionally transparent so that the result can be explained in a college viva rather than presented as a black-box claim.

Question similarity uses cosine similarity over TF-IDF vectors. Topic discovery uses K-Means clustering when enough questions are available.

## Project structure

```text
pyq-predictor/
├── app.py
├── pyq_predictor/
│   ├── __init__.py
│   ├── extraction.py
│   ├── preprocessing.py
│   ├── analysis.py
│   └── prediction.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_analysis.py
│   └── test_prediction.py
├── requirements.txt
├── .gitignore
└── .github/workflows/ci.yml
```

## Limitations

OCR quality depends on scan quality and document layout. The system does not know a university's confidential paper-setting process and cannot guarantee that a predicted topic will appear.

For privacy and security, do not upload documents containing sensitive personal information.

## Resume bullet

**PYQ Predictor — OCR/NLP Data Science Project:** Built a question-paper analytics system using OCR, TF-IDF, cosine similarity, K-Means clustering, and recency-weighted recurrence analysis to identify repeated concepts and similar questions from historical exam papers.