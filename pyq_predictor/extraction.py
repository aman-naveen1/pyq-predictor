"""PDF text extraction with an optional OCR fallback."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import BinaryIO


def extract_pdf_text(file_obj: BinaryIO | bytes | str | Path, ocr: bool = True) -> str:
    """Extract text from a PDF, falling back to OCR for scanned pages.

    ``file_obj`` may be a path, bytes, or an object accepted by PyMuPDF.
    OCR requires the optional pytesseract and Pillow dependencies plus the
    Tesseract executable installed on the operating system.
    """
    import fitz  # PyMuPDF

    if isinstance(file_obj, (str, Path)):
        document = fitz.open(str(file_obj))
    elif isinstance(file_obj, bytes):
        document = fitz.open(stream=file_obj, filetype="pdf")
    else:
        data = file_obj.read()
        document = fitz.open(stream=data, filetype="pdf")

    pages: list[str] = []
    for page in document:
        text = page.get_text("text").strip()
        if text:
            pages.append(text)
            continue

        if not ocr:
            pages.append("")
            continue

        try:
            from PIL import Image
            import pytesseract

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            pages.append(pytesseract.image_to_string(image))
        except Exception as exc:
            pages.append(f"[OCR unavailable: {exc}]")

    document.close()
    return "\n\n".join(pages)


def extract_year(filename: str) -> int | None:
    """Return a plausible four-digit year found in a filename."""
    matches = re.findall(r"(?:19|20)\d{2}", filename)
    return int(matches[-1]) if matches else None


def segment_questions(text: str) -> list[dict]:
    """Split a paper into question-like blocks using common exam numbering."""
    cleaned = text.replace("\r", "\n")
    pattern = re.compile(
        r"(?im)(?=^\s*(?:Q(?:uestion)?\s*)?\d{1,2}\s*[.)\-:]\s+)"
    )
    chunks = [c.strip() for c in pattern.split(cleaned) if c.strip()]

    questions: list[dict] = []
    for index, chunk in enumerate(chunks, start=1):
        match = re.match(r"^\s*(?:Q(?:uestion)?\s*)?(\d{1,2})\s*[.)\-:]\s+", chunk, re.I)
        number = int(match.group(1)) if match else index
        body = chunk[match.end():].strip() if match else chunk
        marks_match = re.search(r"(?:\(|\[)?\s*(\d{1,3})\s*marks?\s*(?:\)|\])?\s*$", body, re.I)
        marks = int(marks_match.group(1)) if marks_match else None
        if marks_match:
            body = body[: marks_match.start()].strip()
        if len(body) >= 12:
            questions.append({"number": number, "text": body, "marks": marks})
    return questions
