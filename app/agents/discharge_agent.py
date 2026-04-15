"""
Discharge Summary Agent
=======================
LangGraph node: processes pages classified as:
  - discharge_summary
  - prescription

Extracts: admission/discharge dates, diagnosis, physician, medications,
          follow-up instructions, prescription details.
Only receives pages assigned by the Segregator.
"""

from __future__ import annotations
import logging
from typing import Any

from app.core.schemas import ClaimState, DischargeSummaryData, AGENT_DOCUMENT_MAP
from app.utils.llm_client import call_llm_json, build_vision_message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

DISCHARGE_SYSTEM = """You are a clinical document extraction specialist for medical insurance claim processing.

You will receive images of discharge summaries and prescriptions from a hospital stay.

Extract ALL of the following fields where present:

From discharge summaries:
- admission_date (YYYY-MM-DD if possible)
- discharge_date (YYYY-MM-DD if possible)
- length_of_stay (e.g., "5 days")
- admission_diagnosis (initial diagnosis on admission)
- final_diagnosis (diagnosis at discharge)
- attending_physician (full name and credentials)
- hospital_name
- mrn (Medical Record Number)
- treatment_summary (brief summary of hospital course / treatment given)
- discharge_medications (list of medication names as strings)
- follow_up_instructions (follow-up care instructions)
- condition_at_discharge (e.g., "Stable, improved")

From prescriptions:
- prescriptions: array of objects with fields: drug_name, dosage, frequency, duration, instructions

Return ONLY a JSON object. Use null for missing fields and [] for empty arrays.
Do not add markdown fences. Do not invent values.

Response format:
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

# Document types routed to this agent
DISCHARGE_DOC_TYPES = {k for k, v in AGENT_DOCUMENT_MAP.items() if v == "discharge_agent"}

# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def discharge_agent(state: ClaimState) -> dict:
    """LangGraph node: extract clinical data from discharge/prescription pages."""
    logger.info("[Discharge Agent] Starting extraction for claim %s", state.claim_id)

    if state.segregator_output is None:
        errors = state.errors.copy() if state.errors else []
        errors.append("Discharge Agent: segregator output not available")
        return {"errors": errors}

    # Collect page indices assigned to this agent
    doc_map = state.segregator_output.document_page_map
    assigned_pages: list[int] = []
    for doc_type in DISCHARGE_DOC_TYPES:
        assigned_pages.extend(doc_map.get(doc_type, []))
    assigned_pages = sorted(set(assigned_pages))
    logger.info("[Discharge Agent] Assigned %d page(s)", len(assigned_pages))

    if not assigned_pages:
        logger.info("[Discharge Agent] No pages assigned — skipping.")
        return {"discharge_data": None}

    logger.info("[Discharge Agent] Processing pages: %s", assigned_pages)

    # Gather only assigned page images and text
    page_images = {p: state.page_images[p] for p in assigned_pages if p in state.page_images}
    text_snippets = "\n\n".join(
        f"--- Page {p} ---\n{state.page_texts.get(p, '(no text)')}"
        for p in assigned_pages
    )

    prompt = (
        f"These {len(assigned_pages)} page(s) are from a medical insurance claim (ID: {state.claim_id}).\n"
        f"They include discharge summaries and/or prescriptions.\n\n"
        f"Extracted text for reference:\n{text_snippets}\n\n"
        f"Please extract all clinical and discharge details from these pages.\n"
        f"Return ONLY the JSON object as specified."
    )

    images = [page_images[k] for k in sorted(page_images.keys())]
    messages = build_vision_message(prompt, images)

    try:
        result = call_llm_json(DISCHARGE_SYSTEM, messages, max_tokens=3072)
        valid_fields = DischargeSummaryData.model_fields.keys()
        filtered = {k: v for k, v in result.items() if k in valid_fields or k == "extra_fields"}
        discharge_data = DischargeSummaryData(**filtered)
        logger.info("[Discharge Agent] Extraction complete. Diagnosis: %s", discharge_data.final_diagnosis or "N/A")
        return {"discharge_data": discharge_data}
    except Exception as exc:
        logger.exception("[Discharge Agent] Extraction failed")
        errors = state.errors.copy() if state.errors else []
        errors.append(f"Discharge Agent error: {exc}")
        return {"discharge_data": DischargeSummaryData(), "errors": errors}
