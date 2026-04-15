"""Segregator node for page-level document classification."""

from __future__ import annotations

import logging
import re
import time

from app.core.schemas import (
    ClaimState,
    PageClassification,
    SegregatorOutput,
    DOCUMENT_TYPES,
)
from app.utils.llm_client import build_vision_message, call_llm_json, call_llm_json_text_only
from app.utils.pdf_utils import render_pages_to_base64, extract_page_texts, get_page_count

logger = logging.getLogger(__name__)
SEGREGATOR_CHUNK_SIZE = 4
SEGREGATOR_TEXT_ONLY_MIN_CHARS = 120
SEGREGATOR_TEXT_SNIPPET_LIMIT = 900
SEGREGATOR_MAX_RETRIES = 2

HEURISTIC_PATTERNS: list[tuple[str, tuple[str, ...], str]] = [
    ("claim_forms", ("claim form", "claim reference", "date filed", "amount claimed", "insurance company"), "Claim or insurance form"),
    ("cheque_or_bank_details", ("bank account", "ifsc", "swift", "cheque", "routing number", "account number"), "Bank or cheque details"),
    ("identity_document", ("date of birth", "id number", "identification card", "government of", "aadhaar", "passport"), "Identity document"),
    ("itemized_bill", ("itemized bill", "invoice", "room charges", "pharmacy", "subtotal", "total amount", "bill number"), "Itemized medical bill"),
    ("discharge_summary", ("discharge summary", "admission date", "discharge date", "condition at discharge", "hospital course"), "Hospital discharge summary"),
    ("prescription", ("prescription", "rx", "dosage", "take one", "tablet", "capsule"), "Medical prescription"),
    ("investigation_report", ("lab report", "laboratory report", "cbc", "metabolic panel", "lipid panel", "thyroid", "pathology"), "Investigation report"),
    ("cash_receipt", ("cash receipt", "receipt no", "paid amount", "received with thanks", "payment received"), "Cash receipt"),
]

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
4. Keep each description under 12 words.

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

def segregator_agent(state: ClaimState) -> ClaimState:
    """LangGraph node: classify every PDF page by document type."""
    logger.info("[Segregator] Starting page classification for claim %s", state.claim_id)

    pdf_bytes = state.pdf_bytes
    total_pages = get_page_count(pdf_bytes)
    state.total_pages = total_pages
    logger.info("[Segregator] PDF page count: %d", total_pages)

    page_texts = extract_page_texts(pdf_bytes)
    state.page_texts = page_texts
    logger.info("[Segregator] Extracted text from %d pages", len(page_texts))

    logger.info("[Segregator] Rendering %d pages to images", total_pages)
    page_images = render_pages_to_base64(pdf_bytes, dpi=120)
    state.page_images = page_images
    logger.info("[Segregator] Rendered %d page images", len(page_images))

    errors = list(state.errors or [])

    result: list[dict] = []
    unresolved_pages: list[int] = []

    for page in range(total_pages):
        heuristic = _heuristic_classify_page(page, page_texts.get(page, ""))
        if heuristic is not None:
            result.append(heuristic)
        else:
            unresolved_pages.append(page)

    logger.info(
        "[Segregator] Heuristic classification resolved %d/%d pages",
        len(result),
        total_pages,
    )

    for chunk in _chunk_pages(unresolved_pages, SEGREGATOR_CHUNK_SIZE):
        try:
            result.extend(_classify_chunk(state.claim_id, chunk, page_texts, page_images))
        except Exception as exc:
            logger.exception("[Segregator] Chunk classification failed for pages %s", chunk)
            errors.append(f"Segregator chunk error for pages {chunk}: {exc}")
            result.extend(_classify_pages_individually(state.claim_id, chunk, page_texts, page_images, errors))

    result = sorted(result, key=lambda item: item["page_number"])

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


def _classify_chunk(
    claim_id: str,
    chunk: list[int],
    page_texts: dict[int, str],
    page_images: dict[int, str],
) -> list[dict]:
    logger.info("[Segregator] Classifying chunk pages %s", chunk)
    prompt = (
        f"Claim ID: {claim_id}\n"
        f"Pages in this chunk: {chunk}\n\n"
        "Classify each page into exactly one document type and preserve the original page_number values.\n\n"
        f"Page text:\n{_build_text_context(chunk, page_texts)}\n\n"
        f"Return a JSON array with exactly {len(chunk)} elements."
    )

    last_error: Exception | None = None
    for attempt in range(1, SEGREGATOR_MAX_RETRIES + 1):
        try:
            if _should_use_text_only(chunk, page_texts):
                chunk_result = call_llm_json_text_only(SEGREGATOR_SYSTEM, prompt, max_tokens=1536)
                return _normalize_chunk_result(chunk, chunk_result)

            images = [page_images[p] for p in chunk if p in page_images]
            messages = build_vision_message(prompt, images)
            chunk_result = call_llm_json(SEGREGATOR_SYSTEM, messages, max_tokens=1536)
            return _normalize_chunk_result(chunk, chunk_result)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "[Segregator] Chunk attempt %d/%d failed for pages %s: %s",
                attempt,
                SEGREGATOR_MAX_RETRIES,
                chunk,
                exc,
            )
            if attempt < SEGREGATOR_MAX_RETRIES:
                time.sleep(1.5 * attempt)

    raise last_error if last_error is not None else RuntimeError("Unknown segregator chunk failure")


def _classify_pages_individually(
    claim_id: str,
    pages: list[int],
    page_texts: dict[int, str],
    page_images: dict[int, str],
    errors: list[str],
) -> list[dict]:
    results: list[dict] = []
    for page in pages:
        try:
            page_result = _classify_chunk(claim_id, [page], page_texts, page_images)
            if page_result:
                results.extend(page_result)
                continue
            raise RuntimeError("Empty page classification result")
        except Exception as exc:
            logger.warning("[Segregator] Page fallback failed for page %d: %s", page, exc)
            errors.append(f"Segregator page fallback error for page {page}: {exc}")
            results.append({
                "page_number": page,
                "document_type": "other",
                "confidence": "low",
                "description": "Fallback classification",
            })
    return results


def _normalize_chunk_result(chunk: list[int], chunk_result: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    if not isinstance(chunk_result, list):
        raise ValueError(f"Expected list result for chunk {chunk}, got {type(chunk_result).__name__}")

    for index, page in enumerate(chunk):
        entry = chunk_result[index] if index < len(chunk_result) and isinstance(chunk_result[index], dict) else {}
        doc_type = entry.get("document_type", "other")
        if doc_type not in DOCUMENT_TYPES:
            doc_type = "other"
        normalized.append({
            "page_number": page,
            "document_type": doc_type,
            "confidence": entry.get("confidence", "medium"),
            "description": entry.get("description", "LLM classification"),
        })

    return normalized


def _chunk_pages(pages: list[int], size: int) -> list[list[int]]:
    return [pages[index:index + size] for index in range(0, len(pages), size)]


def _build_text_context(chunk: list[int], page_texts: dict[int, str]) -> str:
    parts = []
    for page in chunk:
        text = (page_texts.get(page) or "").strip()
        snippet = text[:SEGREGATOR_TEXT_SNIPPET_LIMIT] if text else "(no extractable text)"
        parts.append(f"Page {page}:\n{snippet}")
    return "\n\n".join(parts)


def _should_use_text_only(chunk: list[int], page_texts: dict[int, str]) -> bool:
    return all(len((page_texts.get(page) or "").strip()) >= SEGREGATOR_TEXT_ONLY_MIN_CHARS for page in chunk)


def _heuristic_classify_page(page_number: int, page_text: str) -> dict | None:
    text = re.sub(r"\s+", " ", (page_text or "").strip()).lower()
    if len(text) < 80:
        return None

    matches: list[tuple[str, int, str]] = []
    for doc_type, keywords, description in HEURISTIC_PATTERNS:
        score = sum(1 for keyword in keywords if keyword in text)
        if score:
            matches.append((doc_type, score, description))

    if not matches:
        return None

    matches.sort(key=lambda item: item[1], reverse=True)
    top_doc_type, top_score, description = matches[0]
    if top_score < 2 and len(matches) > 1 and matches[1][1] == top_score:
        return None

    return {
        "page_number": page_number,
        "document_type": top_doc_type,
        "confidence": "medium" if top_score == 2 else "high",
        "description": description,
    }
