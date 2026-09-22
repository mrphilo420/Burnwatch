#!/usr/bin/env python3
"""Document text extraction for the screening upload path.

Accepts only compatible files: PDF, DOCX, TXT, MD. Everything else is
rejected before extraction. Extraction is intentionally conservative:
text-only, no layout interpretation.
"""

import io
import os

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_BYTES = 20 * 1024 * 1024


def extension_of(name):
    return os.path.splitext(name)[1].lower()


def validate(name, data):
    """Return (ok, error). Enforces the whitelist and size cap."""
    ext = extension_of(name)
    if ext not in ALLOWED_EXTENSIONS:
        return False, (f"Unsupported file type '{ext or '(none)'}'. "
                       f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}.")
    if len(data) > MAX_FILE_BYTES:
        return False, f"File too large (max {MAX_FILE_BYTES // (1024 * 1024)} MB)."
    if len(data) == 0:
        return False, "File is empty."
    return True, None


def extract_text(name, data):
    """Extract plain text from an allowed file. Never raises for content
    issues; returns whatever text is recoverable."""
    ext = extension_of(name)
    if ext == ".pdf":
        return _extract_pdf(data)
    if ext == ".docx":
        return _extract_docx(data)
    return _extract_plain(data)


def _extract_pdf(data):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def _extract_docx(data):
    import docx
    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


def _extract_plain(data):
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")