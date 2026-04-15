"""
Itemized Bill Agent
===================
LangGraph node: processes pages classified as:
  - itemized_bill
  - cash_receipt
  - investigation_report

Extracts: bill number, line items with costs, subtotal, tax, total amount,
          payment method, and patient responsibility.
Only receives pages assigned by the Segregator.
"""

from __future__ import annotations
import logging
from typing import Any

from app.core.schemas import ClaimState, ItemizedBillData, BillLineItem, AGENT_DOCUMENT_MAP
from app.utils.llm_client import call_llm_json, build_vision_message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

BILL_SYSTEM = """You are a financial document extraction specialist for medical insurance claim processing.

You will receive images of itemized hospital bills, cash receipts, and investigation/lab reports.

Extract ALL of the following fields where present:

From itemized bills:
- bill_number
- bill_date (YYYY-MM-DD if possible)
- hospital_name
- patient_name
- line_items: array of all individual charges — each with:
    - date (YYYY-MM-DD or null)
    - description (service/item name)
    - quantity (numeric or null)
    - unit_price (numeric or null)
    - amount (numeric, required)
- subtotal (numeric)
- tax (numeric)
- discount (numeric)
- insurance_payment (numeric — amount covered by insurance if stated)
- total_amount (numeric — final total)
- patient_responsibility (numeric — amount owed by patient after insurance)
- payment_method (e.g., "Cash", "Card", "Insurance")

From cash receipts:
- bill_number (receipt number)
- bill_date
- hospital_name
- total_amount
- payment_method
- Add receipt details as line_items if itemised

From investigation reports:
- Add each test/investigation as a line_item with its cost if stated

Rules:
- All monetary values must be numeric (no currency symbols).
- Use null for missing numeric fields, not 0.
- Return ONLY a JSON object. Do not add markdown fences. Do not invent values.

Response format:
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

# Document types routed to this agent
BILL_DOC_TYPES = {k for k, v in AGENT_DOCUMENT_MAP.items() if v == "bill_agent"}

# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def bill_agent(state: ClaimState) -> dict:
    """LangGraph node: extract billing data from assigned pages."""
    logger.info("[Bill Agent] Starting extraction for claim %s", state.claim_id)

    if state.segregator_output is None:
        errors = state.errors.copy() if state.errors else []
        errors.append("Bill Agent: segregator output not available")
        return {"errors": errors}

    # Collect page indices assigned to this agent
    doc_map = state.segregator_output.document_page_map
    assigned_pages: list[int] = []
    for doc_type in BILL_DOC_TYPES:
        assigned_pages.extend(doc_map.get(doc_type, []))
    assigned_pages = sorted(set(assigned_pages))
    logger.info("[Bill Agent] Assigned %d page(s)", len(assigned_pages))

    if not assigned_pages:
        logger.info("[Bill Agent] No pages assigned — skipping.")
        return {"bill_data": None}

    logger.info("[Bill Agent] Processing pages: %s", assigned_pages)

    # Gather only assigned page images and text
    page_images = {p: state.page_images[p] for p in assigned_pages if p in state.page_images}
    text_snippets = "\n\n".join(
        f"--- Page {p} ---\n{state.page_texts.get(p, '(no text)')}"
        for p in assigned_pages
    )

    prompt = (
        f"These {len(assigned_pages)} page(s) are from a medical insurance claim (ID: {state.claim_id}).\n"
        f"They include itemized bills, cash receipts, and/or investigation reports.\n\n"
        f"Extracted text for reference:\n{text_snippets}\n\n"
        f"Please extract ALL billing line items and financial details from these pages.\n"
        f"Return ONLY the JSON object as specified."
    )

    images = [page_images[k] for k in sorted(page_images.keys())]
    messages = build_vision_message(prompt, images)

    try:
        result = call_llm_json(BILL_SYSTEM, messages, max_tokens=4096)

        # Parse line items into BillLineItem models
        raw_items = result.pop("line_items", []) or []
        line_items: list[BillLineItem] = []
        for item in raw_items:
            try:
                line_items.append(BillLineItem(**{
                    k: v for k, v in item.items()
                    if k in BillLineItem.model_fields
                }))
            except Exception as item_exc:
                logger.warning("[Bill Agent] Skipping malformed line item %s: %s", item, item_exc)

        valid_fields = ItemizedBillData.model_fields.keys()
        filtered = {k: v for k, v in result.items() if k in valid_fields or k == "extra_fields"}
        filtered["line_items"] = line_items
        bill_data = ItemizedBillData(**filtered)
        
        logger.info(
            "[Bill Agent] Extraction complete. Total amount: %s | Line items: %d",
            bill_data.total_amount or "N/A",
            len(bill_data.line_items) if bill_data else 0,
        )
        return {"bill_data": bill_data}

    except Exception as exc:
        logger.exception("[Bill Agent] Extraction failed")
        errors = state.errors.copy() if state.errors else []
        errors.append(f"Bill Agent error: {exc}")
        return {"bill_data": ItemizedBillData(), "errors": errors}
