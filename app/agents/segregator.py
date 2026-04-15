"""
Segregator Agent
================
LangGraph node: START → [Segregator Agent] → ...

Responsibilities
----------------
1. Receive the full PDF (bytes) + per-page text from the pipeline state.
2. Use the LLM (with vision) to classify every page into one of 9 document types.
3. Populate `state.segregator_output` with a per-page classification and a
   document_page_map (doc_type → [page_indices]).

The agent sends *all* pages to the LLM in a single vision call to maximise
context for cross-page decisions (e.g., a multi-page itemized bill).
"""

from __future__ import annotations
import json
import logging
from typing import Any

from app.core.schemas import (
    ClaimState,
    PageClassification,
    SegregatorOutput,
    DOCUMENT_TYPES,
)
from app.utils.llm_client import call_llm_json, build_vision_message
from app.utils.pdf_utils import render_pages_to_base64, extract_page_texts, get_page_count

logger = logging.getLogger(__name__)
# In segregator.py, add a limit check
MAX_PAGES_PER_CALL = 20
CHUNK_SIZE = 8

# if total_pages > MAX_PAGES_PER_CALL:
#     logger.warning(f"[Segregator] PDF has {total_pages} pages, which may exceed token limits")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SEGREGATOR_SYSTEM = """You are an expert medical document analyst specialising in insurance claim processing.

Your task is to analyse each page of a multi-page PDF and classify it into EXACTLY ONE of these document types:
- claim_forms            : Medical claim submission forms, insurance claim forms
- cheque_or_bank_details : Cheques, bank account statements, payment details, SWIFT/IFSC info
- identity_document      : Government IDs, driving licences, passports, Aadhaar cards
- itemized_bill          : Detailed hospital/clinic bills listing individual charges
- discharge_summary      : Hospital discharge summaries, admission/discharge records
- prescription           : Doctor prescriptions, medication lists issued by a physician
- investigation_report   : Lab reports, pathology reports, radiology reports, blood tests
- cash_receipt           : Payment receipts, cash memos, proof of payment
- other                  : Anything that does not fit the above types

Rules:
1. Classify EVERY page — do not skip any.
2. Use the page text AND visual layout to make your decision.
3. Return ONLY valid JSON, no markdown fences, no commentary.

Response format (strict JSON array, one entry per page, 0-indexed):
[
  {
    "page_number": 0,
    "document_type": "<one of the 9 types>",
    "confidence": "high|medium|low",
    "description": "<one sentence describing what this page contains>"
  },
  ...
]"""

# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def segregator_agent(state: ClaimState) -> ClaimState:
    """LangGraph node: classify every PDF page by document type."""
    logger.info("[Segregator] Starting page classification for claim %s", state.claim_id)

    pdf_bytes = state.pdf_bytes
    total_pages = get_page_count(pdf_bytes)
    state.total_pages = total_pages
    logger.info("[Segregator] PDF page count: %d", total_pages)

    # Extract text for all pages
    page_texts = extract_page_texts(pdf_bytes)
    state.page_texts = page_texts
    logger.info("[Segregator] Extracted text from %d pages", len(page_texts))

    # Render all pages to images for vision model
    logger.info("[Segregator] Rendering %d pages to images…", total_pages)
    page_images = render_pages_to_base64(pdf_bytes, dpi=120)
    state.page_images = page_images
    logger.info("[Segregator] Rendered %d page images", len(page_images))

    errors = list(state.errors or [])

    # Build prompt with text context to assist vision
    text_context_parts = []
    for i in range(total_pages):
        txt = page_texts.get(i, "").strip()
        snippet = txt[:600] if txt else "(no extractable text — likely image-based)"
        text_context_parts.append(f"=== Page {i} text snippet ===\n{snippet}")
    text_context = "\n\n".join(text_context_parts)

    prompt = (
        f"This PDF has {total_pages} page(s) for claim ID: {state.claim_id}.\n\n"
        f"I am providing you with:\n"
        f"1. Images of every page (in order, 0-indexed)\n"
        f"2. Extracted text snippets for context\n\n"
        f"Text snippets:\n{text_context}\n\n"
        f"Please classify EVERY page (0 to {total_pages - 1}) into one of the 9 document types.\n"
        f"Return a JSON array with exactly {total_pages} elements."
    )

    # Vision call — send all page images + prompt
    images = [page_images[k] for k in sorted(page_images.keys())]
    messages = build_vision_message(prompt, images)

    try:
        result = call_llm_json(SEGREGATOR_SYSTEM, messages, max_tokens=4096)
    except Exception as exc:
        logger.exception("[Segregator] LLM call failed")
        errors.append(f"Segregator LLM error: {exc}")
        # Fallback: classify everything as 'other'
        result = [
            {"page_number": i, "document_type": "other", "confidence": "low", "description": "Fallback classification"}
            for i in range(total_pages)
        ]

    # Parse and validate the response
    pages: list[PageClassification] = []
    doc_page_map: dict[str, list[int]] = {dt: [] for dt in DOCUMENT_TYPES}

    for entry in result:
        try:
            page_num = int(entry.get("page_number", 0))
            doc_type = entry.get("document_type", "other")
            if doc_type not in DOCUMENT_TYPES:
                logger.warning("[Segregator] Unknown doc type '%s' on page %d, setting to 'other'", doc_type, page_num)
                doc_type = "other"
            pc = PageClassification(
                page_number=page_num,
                document_type=doc_type,
                confidence=entry.get("confidence", "medium"),
                description=entry.get("description", ""),
            )
            pages.append(pc)
            doc_page_map[doc_type].append(page_num)
        except Exception as exc:
            logger.warning("[Segregator] Failed to parse entry %s: %s", entry, exc)

    # Remove empty doc types
    doc_page_map = {k: v for k, v in doc_page_map.items() if v}

    state.segregator_output = SegregatorOutput(pages=pages, document_page_map=doc_page_map)

    logger.info(
        "[Segregator] Classification complete | classified_pages=%d/%d | document_types=%s",
        len(pages),
        total_pages,
        list(doc_page_map.keys()),
    )
    return {
        "total_pages": total_pages,
        "page_texts": page_texts,
        "page_images": page_images,
        "segregator_output": state.segregator_output,
        "errors": errors,
    }
