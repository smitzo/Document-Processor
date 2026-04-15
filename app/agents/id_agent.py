"""Identity extraction node."""

from __future__ import annotations

import logging
import re

from app.core.schemas import AGENT_DOCUMENT_MAP, ClaimState, IdentityData
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
        fallback_source = f"{text_context}\n\n{exc}"
        fallback_data = _extract_identity_from_text(fallback_source)
        if _has_identity_signal(fallback_data):
            logger.info("[ID Agent] Recovered identity data from local text fallback")
            errors = state.errors.copy() if state.errors else []
            errors.append(f"ID Agent warning: LLM failed, used local text fallback: {exc}")
            return {"identity_data": fallback_data, "errors": errors}

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


def _extract_identity_from_text(text: str) -> IdentityData:
    compact = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return IdentityData(
        patient_name=_search(compact, [
            r"patient name[:\s]+([A-Z][A-Za-z ,.'-]{2,})",
            r"name[:\s]+([A-Z][A-Za-z ,.'-]{2,})",
            r'"patient_name"\s*:\s*"([^"]+)"',
        ]),
        date_of_birth=_search(compact, [
            r"(?:date of birth|dob)[:\s]+([A-Za-z0-9,/\- ]{4,})",
            r'"date_of_birth"\s*:\s*"([^"]+)"',
        ]),
        gender=_search(compact, [
            r"gender[:\s]+([A-Za-z]+)",
            r"\b(sex)[:\s]+([A-Za-z]+)",
            r'"gender"\s*:\s*"([^"]+)"',
        ], group=2),
        blood_group=_search(compact, [
            r"(?:blood group|blood type)[:\s]+([ABO]{1,2}[+-])",
        ]),
        address=_search(compact, [
            r"address[:\s]+([A-Za-z0-9,.\- ]{8,})",
            r'"address"\s*:\s*"([^"]+)"',
        ]) or _search_address_from_lines(lines),
        contact_number=_search(compact, [
            r"(?:contact number|phone|mobile)[:\s]+([+\d][\d\- ]{7,})",
            r'"contact_number"\s*:\s*"([^"]+)"',
        ]),
        email=_search(compact, [
            r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
            r'"email"\s*:\s*"([^"]+)"',
        ]),
        id_number=_search(compact, [
            r"(?:id number|identification number|document number)[:\s]+([A-Za-z0-9\-]{4,})",
            r'"id_number"\s*:\s*"([^"]+)"',
        ]),
        policy_number=_search(compact, [
            r"(?:policy number|policy no)[:\s]+([A-Za-z0-9\-]{4,})",
            r'"policy_number"\s*:\s*"([^"]+)"',
        ]),
        insurance_provider=_search(compact, [
            r"(?:insurance provider|insurance company|insurer)[:\s]+([A-Za-z0-9 &.\-]{4,})",
            r'"insurance_provider"\s*:\s*"([^"]+)"',
        ]),
        bank_account_number=_search(compact, [
            r"(?:account number|a/c number)[:\s]+([A-Za-z0-9\-]{6,})",
            r'"bank_account_number"\s*:\s*"([^"]+)"',
        ]),
        bank_name=_search(compact, [
            r"bank name[:\s]+([A-Za-z0-9 &.\-]{3,})",
            r'"bank_name"\s*:\s*"([^"]+)"',
        ]),
        ifsc_routing_number=_search(compact, [
            r"(?:ifsc|routing number)[:\s]+([A-Za-z0-9\-]{4,})",
            r'"ifsc_routing_number"\s*:\s*"([^"]+)"',
        ]),
        swift_code=_search(compact, [
            r"swift(?: code)?[:\s]+([A-Za-z0-9\-]{4,})",
            r'"swift_code"\s*:\s*"([^"]+)"',
        ]),
        extra_fields={},
    )


def _search(text: str, patterns: list[str], group: int = 1) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            groups = match.groups()
            selected_group = group
            if selected_group > len(groups):
                selected_group = 1
            value = match.group(selected_group).strip(" ,.-")
            if value:
                return value
    return None


def _search_address_from_lines(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if re.search(r"address", line, flags=re.IGNORECASE):
            value = re.sub(r"^.*address[:\s]*", "", line, flags=re.IGNORECASE).strip(" ,.-")
            if value:
                return value
            if index + 1 < len(lines):
                next_line = lines[index + 1].strip(" ,.-")
                if len(next_line) > 6:
                    return next_line
    return None


def _has_identity_signal(data: IdentityData) -> bool:
    strong_fields = [
        data.patient_name,
        data.policy_number,
        data.bank_account_number,
        data.id_number,
    ]
    supporting_fields = [
        data.date_of_birth,
        data.email,
        data.contact_number,
        data.bank_name,
        data.insurance_provider,
    ]
    if any(strong_fields) and any(supporting_fields):
        return True
    if data.patient_name and data.date_of_birth:
        return True
    return sum(1 for field in [*strong_fields, *supporting_fields] if field) >= 2
