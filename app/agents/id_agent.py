"""Identity extraction node."""

from __future__ import annotations

import logging

from app.core.schemas import AGENT_DOCUMENT_MAP, ClaimState, IdentityData
from app.utils.text_extractors import extract_identity_data_from_text, has_identity_signal
from app.utils.llm_client import build_vision_message, call_llm_json, call_llm_json_text_only

logger = logging.getLogger(__name__)

ID_AGENT_SYSTEM = """Extract identity, policy, and bank details from claim documents.

Return one JSON object only. Use null for missing scalar fields and {} for extra_fields.
Do not add commentary or markdown. Keep extra_fields brief and only for important values not already mapped.

Schema:
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

ID_DOC_TYPES = {k for k, v in AGENT_DOCUMENT_MAP.items() if v == "id_agent"}
TEXT_ONLY_MIN_CHARS = 80
TEXT_SNIPPET_LIMIT = 1400


def id_agent(state: ClaimState) -> dict:
    logger.info("[ID Agent] Starting extraction for claim %s", state.claim_id)

    if state.segregator_output is None:
        errors = state.errors.copy() if state.errors else []
        errors.append("ID Agent: segregator output not available")
        return {"errors": errors}

    doc_map = state.segregator_output.document_page_map
    assigned_pages: list[int] = []
    for doc_type in ID_DOC_TYPES:
        assigned_pages.extend(doc_map.get(doc_type, []))
    assigned_pages = sorted(set(assigned_pages))
    logger.info("[ID Agent] Assigned %d page(s)", len(assigned_pages))

    if not assigned_pages:
        logger.info("[ID Agent] No pages assigned - skipping")
        return {"identity_data": None}

    prompt = (
        f"Claim ID: {state.claim_id}\n"
        f"Pages: {assigned_pages}\n\n"
        "Extract identity, policy, and bank details from only these pages."
    )
    text_context = _build_text_context(state, assigned_pages)
    local_identity = extract_identity_data_from_text(text_context)
    if has_identity_signal(local_identity):
        logger.info("[ID Agent] Returning deterministic text extraction result")
        return {"identity_data": local_identity}

    try:
        if _should_use_text_only(state, assigned_pages):
            result = call_llm_json_text_only(
                ID_AGENT_SYSTEM,
                f"{prompt}\n\nPage text:\n{text_context}",
                max_tokens=1024,
            )
        else:
            images = [state.page_images[p] for p in assigned_pages if p in state.page_images]
            messages = build_vision_message(
                f"{prompt}\n\nPage text:\n{text_context}",
                images,
            )
            result = call_llm_json(ID_AGENT_SYSTEM, messages, max_tokens=1024)

        identity_data = IdentityData(**{
            k: v for k, v in result.items()
            if k in IdentityData.model_fields or k == "extra_fields"
        })
        logger.info("[ID Agent] Extraction complete | patient=%s", identity_data.patient_name or "N/A")
        return {"identity_data": identity_data}
    except Exception as exc:
        logger.exception("[ID Agent] Extraction failed")
        fallback_data = extract_identity_data_from_text(f"{text_context}\n\n{exc}")
        if has_identity_signal(fallback_data):
            logger.info("[ID Agent] Recovered identity data from local text fallback")
            return {"identity_data": fallback_data}

        errors = state.errors.copy() if state.errors else []
        errors.append(f"ID Agent error: {exc}")
        return {"identity_data": IdentityData(), "errors": errors}


def _build_text_context(state: ClaimState, pages: list[int]) -> str:
    parts = []
    for page in pages:
        text = (state.page_texts.get(page) or "").strip()
        snippet = text[:TEXT_SNIPPET_LIMIT] if text else "(no extractable text)"
        parts.append(f"Page {page}:\n{snippet}")
    return "\n\n".join(parts)


def _should_use_text_only(state: ClaimState, pages: list[int]) -> bool:
    return all(len((state.page_texts.get(page) or "").strip()) >= TEXT_ONLY_MIN_CHARS for page in pages)
