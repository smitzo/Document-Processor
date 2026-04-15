"""
ID Agent
========
LangGraph node: processes pages classified as:
  - identity_document
  - claim_forms
  - cheque_or_bank_details

Extracts: patient name, DOB, ID numbers, policy details, bank/cheque info.
Only receives the page images/text assigned to it by the Segregator — NOT the full PDF.
"""

from __future__ import annotations
import logging

from app.core.schemas import ClaimState, IdentityData, AGENT_DOCUMENT_MAP
from app.utils.llm_client import call_llm_json, build_vision_message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ID_AGENT_SYSTEM = """You are an expert document extraction specialist for medical insurance claim processing.

You will receive images of document pages related to patient identity, claim forms, and bank/cheque details.

Extract ALL of the following fields where present:
- patient_name
- date_of_birth (format: YYYY-MM-DD if possible)
- gender
- blood_group
- address (full address as a single string)
- contact_number
- email
- id_number (government ID / driving licence number)
- policy_number (insurance policy number)
- insurance_provider (insurance company name)
- bank_account_number
- bank_name
- ifsc_routing_number
- swift_code

Return ONLY a JSON object. Use null for fields not found. Do not add markdown fences.
Do not invent or guess values — only extract what is explicitly visible.

Response format:
{
  "patient_name": "...",
  "date_of_birth": "...",
  "gender": "...",
  "blood_group": "...",
  "address": "...",
  "contact_number": "...",
  "email": "...",
  "id_number": "...",
  "policy_number": "...",
  "insurance_provider": "...",
  "bank_account_number": "...",
  "bank_name": "...",
  "ifsc_routing_number": "...",
  "swift_code": "...",
  "extra_fields": {}
}"""

# Document types routed to this agent
ID_DOC_TYPES = {k for k, v in AGENT_DOCUMENT_MAP.items() if v == "id_agent"}

# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def id_agent(state: ClaimState) -> dict:
    """LangGraph node: extract identity and policy data from assigned pages."""
    logger.info("[ID Agent] Starting extraction for claim %s", state.claim_id)

    if state.segregator_output is None:
        errors = state.errors.copy() if state.errors else []
        errors.append("ID Agent: segregator output not available")
        return {"errors": errors}

    # Collect page indices assigned to this agent
    doc_map = state.segregator_output.document_page_map
    assigned_pages: list[int] = []
    for doc_type in ID_DOC_TYPES:
        assigned_pages.extend(doc_map.get(doc_type, []))
    assigned_pages = sorted(set(assigned_pages))
    logger.info("[ID Agent] Assigned %d page(s)", len(assigned_pages))

    if not assigned_pages:
        logger.info("[ID Agent] No pages assigned — skipping.")
        return {"identity_data": None}

    logger.info("[ID Agent] Processing pages: %s", assigned_pages)

    # Gather only assigned page images and text
    page_images = {p: state.page_images[p] for p in assigned_pages if p in state.page_images}
    text_snippets = "\n\n".join(
        f"--- Page {p} ---\n{state.page_texts.get(p, '(no text)')}"
        for p in assigned_pages
    )

    prompt = (
        f"These {len(assigned_pages)} page(s) are from a medical insurance claim (ID: {state.claim_id}).\n"
        f"They include identity documents, claim forms, or bank/cheque details.\n\n"
        f"Extracted text for reference:\n{text_snippets}\n\n"
        f"Please extract all identity and financial details from these pages.\n"
        f"Return ONLY the JSON object as specified."
    )

    images = [page_images[k] for k in sorted(page_images.keys())]
    messages = build_vision_message(prompt, images)

    try:
        result = call_llm_json(ID_AGENT_SYSTEM, messages, max_tokens=2048)
        identity_data = IdentityData(**{
            k: v for k, v in result.items()
            if k in IdentityData.model_fields or k == "extra_fields"
        })
        logger.info("[ID Agent] Extraction complete. Patient: %s", identity_data.patient_name or "N/A")
        return {"identity_data": identity_data}
    except Exception as exc:
        logger.exception("[ID Agent] Extraction failed")
        errors = state.errors.copy() if state.errors else []
        errors.append(f"ID Agent error: {exc}")
        return {"identity_data": IdentityData(), "errors": errors}
