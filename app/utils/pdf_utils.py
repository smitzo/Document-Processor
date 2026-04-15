"""
PDF Utilities
=============
Helpers for rendering PDF pages to base64 PNGs and extracting text.
Uses PyMuPDF (fitz).
"""

from __future__ import annotations
import base64
import io
import logging
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def get_page_count(pdf_bytes: bytes) -> int:
    """Return the total number of pages in the PDF."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count


def render_pages_to_base64(pdf_bytes: bytes, dpi: int = 120) -> dict[int, str]:
    """
    Render every page of a PDF to a base64-encoded PNG string.

    Parameters
    ----------
    pdf_bytes : bytes  — raw PDF content
    dpi       : int    — rendering resolution (higher = better quality, larger size)

    Returns
    -------
    dict mapping 0-indexed page number → base64 PNG string
    """
    result: dict[int, str] = {}
    zoom = dpi / 72.0  # fitz uses 72 dpi as baseline
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            png_bytes = pixmap.tobytes("png")
            result[page_num] = base64.b64encode(png_bytes).decode("utf-8")
            logger.debug("Rendered page %d (%d bytes PNG)", page_num, len(png_bytes))

    return result


def extract_page_texts(pdf_bytes: bytes) -> dict[int, str]:
    """
    Extract plain text from every page of a PDF.

    Parameters
    ----------
    pdf_bytes : bytes

    Returns
    -------
    dict mapping 0-indexed page number → extracted text string
    """
    result: dict[int, str] = {}

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text("text")
            result[page_num] = text.strip()

    return result
