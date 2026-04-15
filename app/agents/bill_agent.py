"""Itemized bill extraction node."""

from __future__ import annotations

import logging
from typing import Any

from app.core.schemas import AGENT_DOCUMENT_MAP, BillLineItem, ClaimState, ItemizedBillData
from app.utils.llm_client import build_vision_message, call_llm_json, call_llm_json_text_only

logger = logging.getLogger(__name__)

BILL_SYSTEM = """You extract medical billing data from itemized bills, receipts, and investigation reports.

Return one JSON object only. Monetary fields must be numeric or null. Use [] for line_items and {} for extra_fields.
Do not add commentary or markdown. Keep extra_fields brief.

Schema:
{
  "bill_number": "...",
  "bill_date": "...",
  "hospital_name": "...",
  "patient_name": "...",
  "line_items": [
    {
      "date": "...",
      "description": "...",
      "quantity": 1,
      "unit_price": 500.00,
      "amount": 500.00
    }
  ],
  "subtotal": 0.00,
  "tax": 0.00,
  "discount": 0.00,
  "insurance_payment": 0.00,
  "total_amount": 0.00,
  "patient_responsibility": 0.00,
  "payment_method": "...",
  "extra_fields": {}
}"""

BILL_DOC_TYPES = {k for k, v in AGENT_DOCUMENT_MAP.items() if v == "bill_agent"}
TEXT_ONLY_MIN_CHARS = 80
TEXT_SNIPPET_LIMIT = 1000
BILL_CHUNK_SIZE = 1


def bill_agent(state: ClaimState) -> dict:
    logger.info("[Bill Agent] Starting extraction for claim %s", state.claim_id)

    if state.segregator_output is None:
        errors = state.errors.copy() if state.errors else []
        errors.append("Bill Agent: segregator output not available")
        return {"errors": errors}

    doc_map = state.segregator_output.document_page_map
    assigned_pages: list[int] = []
    for doc_type in BILL_DOC_TYPES:
        assigned_pages.extend(doc_map.get(doc_type, []))
    assigned_pages = sorted(set(assigned_pages))
    logger.info("[Bill Agent] Assigned %d page(s)", len(assigned_pages))

    if not assigned_pages:
        logger.info("[Bill Agent] No pages assigned - skipping")
        return {"bill_data": None}

    try:
        merged = _extract_billing_chunks(state, assigned_pages)
        bill_data = _build_bill_data(merged)
        logger.info(
            "[Bill Agent] Extraction complete | total=%s | line_items=%d",
            bill_data.total_amount or "N/A",
            len(bill_data.line_items),
        )
        return {"bill_data": bill_data}
    except Exception as exc:
        logger.exception("[Bill Agent] Extraction failed")
        errors = state.errors.copy() if state.errors else []
        errors.append(f"Bill Agent error: {exc}")
        return {"bill_data": ItemizedBillData(), "errors": errors}


def _extract_billing_chunks(state: ClaimState, assigned_pages: list[int]) -> dict[str, Any]:
    merged: dict[str, Any] = {"line_items": [], "extra_fields": {}}
    for pages in _chunk_pages(assigned_pages, BILL_CHUNK_SIZE):
        logger.info("[Bill Agent] Extracting chunk pages %s", pages)
        prompt = (
            f"Claim ID: {state.claim_id}\n"
            f"Pages: {pages}\n\n"
            f"Extract billing data from only these pages. Keep line_items complete for this chunk."
        )

        if _should_use_text_only(state, pages):
            user_prompt = f"{prompt}\n\nPage text:\n{_build_text_context(state, pages)}"
            chunk_result = call_llm_json_text_only(BILL_SYSTEM, user_prompt, max_tokens=2048)
        else:
            images = [state.page_images[p] for p in pages if p in state.page_images]
            messages = build_vision_message(f"{prompt}\n\nPage text:\n{_build_text_context(state, pages)}", images)
            chunk_result = call_llm_json(BILL_SYSTEM, messages, max_tokens=2048)

        merged = _merge_bill_payloads(merged, chunk_result)

    return merged


def _chunk_pages(pages: list[int], size: int) -> list[list[int]]:
    return [pages[index:index + size] for index in range(0, len(pages), size)]


def _build_text_context(state: ClaimState, pages: list[int]) -> str:
    parts = []
    for page in pages:
        text = (state.page_texts.get(page) or "").strip()
        snippet = text[:TEXT_SNIPPET_LIMIT] if text else "(no extractable text)"
        parts.append(f"Page {page}:\n{snippet}")
    return "\n\n".join(parts)


def _should_use_text_only(state: ClaimState, pages: list[int]) -> bool:
    return all(len((state.page_texts.get(page) or "").strip()) >= TEXT_ONLY_MIN_CHARS for page in pages)


def _merge_bill_payloads(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    scalar_fields = [
        "bill_number",
        "bill_date",
        "hospital_name",
        "patient_name",
        "subtotal",
        "tax",
        "discount",
        "insurance_payment",
        "total_amount",
        "patient_responsibility",
        "payment_method",
    ]
    for field in scalar_fields:
        if base.get(field) is None and incoming.get(field) is not None:
            base[field] = incoming[field]

    base["line_items"] = _dedupe_line_items(
        [*(base.get("line_items") or []), *(incoming.get("line_items") or [])]
    )
    base["extra_fields"] = {**(base.get("extra_fields") or {}), **(incoming.get("extra_fields") or {})}

    if base.get("total_amount") is None:
        computed_total = sum(item.amount for item in _coerce_line_items(base["line_items"]))
        if computed_total:
            base["total_amount"] = computed_total

    return base


def _build_bill_data(result: dict[str, Any]) -> ItemizedBillData:
    valid_fields = ItemizedBillData.model_fields.keys()
    filtered = {k: v for k, v in result.items() if k in valid_fields or k == "extra_fields"}
    filtered["line_items"] = _coerce_line_items(result.get("line_items") or [])
    return ItemizedBillData(**filtered)


def _coerce_line_items(raw_items: list[Any]) -> list[BillLineItem]:
    line_items: list[BillLineItem] = []
    for item in raw_items:
        try:
            payload = item if isinstance(item, dict) else item.model_dump()
            line_items.append(BillLineItem(**{
                key: value for key, value in payload.items() if key in BillLineItem.model_fields
            }))
        except Exception as exc:
            logger.warning("[Bill Agent] Skipping malformed line item %s: %s", item, exc)
    return line_items


def _dedupe_line_items(raw_items: list[Any]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, Any], ...]] = set()
    deduped: list[dict[str, Any]] = []
    for item in raw_items:
        payload = item if isinstance(item, dict) else item.model_dump()
        key = tuple(sorted(payload.items()))
        if key not in seen:
            seen.add(key)
            deduped.append(payload)
    return deduped
