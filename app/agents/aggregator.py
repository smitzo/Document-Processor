"""
Aggregator Node
===============
LangGraph node: final step in the pipeline.

Combines outputs from:
  - ID Agent        → identity_data
  - Discharge Agent → discharge_data
  - Bill Agent      → bill_data

Produces a ClaimResponse stored in the graph's final state.
"""

from __future__ import annotations
import logging
from typing import Any

from app.core.schemas import ClaimState, ClaimResponse

logger = logging.getLogger(__name__)


def aggregator(state: ClaimState) -> dict[str, Any]:
    """LangGraph node: aggregate all extracted data into a final response."""
    unique_errors = _unique_errors(state.errors)
    logger.info("[Aggregator] Building final response for claim %s", state.claim_id)
    logger.info(
        "[Aggregator] Inputs | segregated=%s | identity=%s | discharge=%s | bill=%s | errors=%d",
        state.segregator_output is not None,
        state.identity_data is not None,
        state.discharge_data is not None,
        state.bill_data is not None,
        len(unique_errors),
    )

    # -----------------------------------------------------------------------
    # Page classification summary
    # -----------------------------------------------------------------------
    page_classification: list[dict[str, Any]] = []
    if state.segregator_output:
        for pc in state.segregator_output.pages:
            page_classification.append({
                "page_number": pc.page_number,
                "document_type": pc.document_type,
                "confidence": pc.confidence,
                "description": pc.description,
            })

    # -----------------------------------------------------------------------
    # Extracted data — each agent section only included if data exists
    # -----------------------------------------------------------------------
    extracted_data: dict[str, Any] = {}

    if state.identity_data:
        extracted_data["identity"] = state.identity_data.model_dump(exclude_none=False)

    if state.discharge_data:
        extracted_data["discharge_summary"] = state.discharge_data.model_dump(exclude_none=False)

    if state.bill_data:
        bill_dict = state.bill_data.model_dump(exclude_none=False)
        # Serialize BillLineItem objects that may remain as models
        bill_dict["line_items"] = [
            item if isinstance(item, dict) else item.model_dump()
            for item in (state.bill_data.line_items or [])
        ]
        extracted_data["itemized_bill"] = bill_dict

    # -----------------------------------------------------------------------
    # Processing summary
    # -----------------------------------------------------------------------
    doc_map = state.segregator_output.document_page_map if state.segregator_output else {}
    processing_summary: dict[str, Any] = {
        "total_pages": state.total_pages,
        "document_types_found": list(doc_map.keys()),
        "pages_per_document_type": doc_map,
        "agents_ran": _agents_that_ran(state),
        "errors_count": len(unique_errors),
    }

    # -----------------------------------------------------------------------
    # Determine status
    # -----------------------------------------------------------------------
    status = "success"
    if unique_errors:
        status = "partial" if extracted_data else "failed"
    elif state.segregator_output and all(
        page.document_type == "other" and page.description == "Fallback classification"
        for page in state.segregator_output.pages
    ):
        # Defensive guard in case fallback pages are ever produced without an error.
        status = "failed"

    # Build the response object
    response = ClaimResponse(
        claim_id=state.claim_id,
        status=status,
        page_classification=page_classification,
        extracted_data=extracted_data,
        errors=unique_errors,
        processing_summary=processing_summary,
    )

    logger.info(
        "[Aggregator] Done. Status=%s | Agents ran: %s | Errors: %d",
        status,
        processing_summary["agents_ran"],
        len(unique_errors),
    )
    
    # Persist the response on the graph state so the API can read it reliably.
    return {"final_response": response, "errors": unique_errors}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _agents_that_ran(state: ClaimState) -> list[str]:
    ran = []
    if state.identity_data is not None:
        ran.append("id_agent")
    if state.discharge_data is not None:
        ran.append("discharge_agent")
    if state.bill_data is not None:
        ran.append("bill_agent")
    return ran


def _unique_errors(errors: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for error in errors:
        if error not in seen:
            seen.add(error)
            unique.append(error)
    return unique
