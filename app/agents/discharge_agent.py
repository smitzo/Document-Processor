"""Discharge summary extraction node."""

from __future__ import annotations

import logging
from typing import Any

from app.core.schemas import AGENT_DOCUMENT_MAP, ClaimState, DischargeSummaryData
from app.utils.llm_client import build_vision_message, call_llm_json, call_llm_json_text_only

logger = logging.getLogger(__name__)

DISCHARGE_SYSTEM = """You extract clinical claim data from discharge summaries and prescriptions.

Return one JSON object only. Use null for missing scalar fields, [] for empty lists, and {} for extra_fields.
Do not add commentary or markdown. Keep extra_fields brief.

Schema:
{
  "admission_date": "...",
  "discharge_date": "...",
  "length_of_stay": "...",
  "admission_diagnosis": "...",
  "final_diagnosis": "...",
  "attending_physician": "...",
  "hospital_name": "...",
  "mrn": "...",
  "treatment_summary": "...",
  "discharge_medications": ["..."],
  "follow_up_instructions": "...",
  "condition_at_discharge": "...",
  "prescriptions": [
    {
      "drug_name": "...",
      "dosage": "...",
      "frequency": "...",
      "duration": "...",
      "instructions": "..."
    }
  ],
  "extra_fields": {}
}"""

DISCHARGE_DOC_TYPES = {k for k, v in AGENT_DOCUMENT_MAP.items() if v == "discharge_agent"}
TEXT_ONLY_MIN_CHARS = 80
TEXT_SNIPPET_LIMIT = 1200


def discharge_agent(state: ClaimState) -> dict:
    logger.info("[Discharge Agent] Starting extraction for claim %s", state.claim_id)

    if state.segregator_output is None:
        errors = state.errors.copy() if state.errors else []
        errors.append("Discharge Agent: segregator output not available")
        return {"errors": errors}

    doc_map = state.segregator_output.document_page_map
    assigned_pages: list[int] = []
    for doc_type in DISCHARGE_DOC_TYPES:
        assigned_pages.extend(doc_map.get(doc_type, []))
    assigned_pages = sorted(set(assigned_pages))
    logger.info("[Discharge Agent] Assigned %d page(s)", len(assigned_pages))

    if not assigned_pages:
        logger.info("[Discharge Agent] No pages assigned - skipping")
        return {"discharge_data": None}

    summary_pages = sorted(doc_map.get("discharge_summary", []))
    prescription_pages = sorted(doc_map.get("prescription", []))

    try:
        result = _extract_and_merge(state, summary_pages, prescription_pages)
        discharge_data = _build_discharge_data(result)
        logger.info(
            "[Discharge Agent] Extraction complete | diagnosis=%s | prescriptions=%d",
            discharge_data.final_diagnosis or "N/A",
            len(discharge_data.prescriptions),
        )
        return {"discharge_data": discharge_data}
    except Exception as exc:
        logger.exception("[Discharge Agent] Extraction failed")
        errors = state.errors.copy() if state.errors else []
        errors.append(f"Discharge Agent error: {exc}")
        return {"discharge_data": DischargeSummaryData(), "errors": errors}


def _extract_and_merge(
    state: ClaimState,
    summary_pages: list[int],
    prescription_pages: list[int],
) -> dict[str, Any]:
    merged: dict[str, Any] = {"extra_fields": {}, "discharge_medications": [], "prescriptions": []}

    if summary_pages:
        summary_result = _run_discharge_call(
            state=state,
            pages=summary_pages,
            purpose="discharge summary",
            max_tokens=1536,
        )
        merged = _merge_discharge_payloads(merged, summary_result)

    if prescription_pages:
        prescription_result = _run_discharge_call(
            state=state,
            pages=prescription_pages,
            purpose="prescription",
            max_tokens=1536,
        )
        merged = _merge_discharge_payloads(merged, prescription_result)

    if not summary_pages and not prescription_pages:
        assigned_pages = sorted({
            *summary_pages,
            *prescription_pages,
        })
        generic_result = _run_discharge_call(
            state=state,
            pages=assigned_pages,
            purpose="clinical pages",
            max_tokens=1536,
        )
        merged = _merge_discharge_payloads(merged, generic_result)

    return merged


def _run_discharge_call(
    state: ClaimState,
    pages: list[int],
    purpose: str,
    max_tokens: int,
) -> dict[str, Any]:
    logger.info("[Discharge Agent] Extracting %s from pages %s", purpose, pages)
    prompt = (
        f"Claim ID: {state.claim_id}\n"
        f"Document group: {purpose}\n"
        f"Pages: {pages}\n\n"
        f"Extract the discharge and prescription schema from only these pages."
    )

    if _should_use_text_only(state, pages):
        text_prompt = f"{prompt}\n\nPage text:\n{_build_text_context(state, pages)}"
        return call_llm_json_text_only(DISCHARGE_SYSTEM, text_prompt, max_tokens=max_tokens)

    images = [state.page_images[p] for p in pages if p in state.page_images]
    messages = build_vision_message(f"{prompt}\n\nPage text:\n{_build_text_context(state, pages)}", images)
    return call_llm_json(DISCHARGE_SYSTEM, messages, max_tokens=max_tokens)


def _build_text_context(state: ClaimState, pages: list[int]) -> str:
    parts = []
    for page in pages:
        text = (state.page_texts.get(page) or "").strip()
        snippet = text[:TEXT_SNIPPET_LIMIT] if text else "(no extractable text)"
        parts.append(f"Page {page}:\n{snippet}")
    return "\n\n".join(parts)


def _should_use_text_only(state: ClaimState, pages: list[int]) -> bool:
    return all(len((state.page_texts.get(page) or "").strip()) >= TEXT_ONLY_MIN_CHARS for page in pages)


def _merge_discharge_payloads(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    scalar_fields = [
        "admission_date",
        "discharge_date",
        "length_of_stay",
        "admission_diagnosis",
        "final_diagnosis",
        "attending_physician",
        "hospital_name",
        "mrn",
        "treatment_summary",
        "follow_up_instructions",
        "condition_at_discharge",
    ]
    for field in scalar_fields:
        if not base.get(field) and incoming.get(field):
            base[field] = incoming[field]

    base["discharge_medications"] = _dedupe_strings(
        [*(base.get("discharge_medications") or []), *(incoming.get("discharge_medications") or [])]
    )
    base["prescriptions"] = _dedupe_dicts(
        [*(base.get("prescriptions") or []), *(incoming.get("prescriptions") or [])]
    )
    base["extra_fields"] = {**(base.get("extra_fields") or {}), **(incoming.get("extra_fields") or {})}
    return base


def _build_discharge_data(result: dict[str, Any]) -> DischargeSummaryData:
    valid_fields = DischargeSummaryData.model_fields.keys()
    filtered = {k: v for k, v in result.items() if k in valid_fields or k == "extra_fields"}
    return DischargeSummaryData(**filtered)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value:
            continue
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _dedupe_dicts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, Any], ...]] = set()
    result: list[dict[str, Any]] = []
    for value in values:
        if not value:
            continue
        key = tuple(sorted(value.items()))
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
