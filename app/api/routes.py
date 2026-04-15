from __future__ import annotations
import logging


from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.graph import claim_graph
from app.core.schemas import ClaimState, ClaimResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/process",
    response_model=ClaimResponse,
    summary="Process a medical insurance claim PDF",
    tags=["Claims"],
)
async def process_claim(
    claim_id: str = Form(..., description="Unique claim identifier"),
    file: UploadFile = File(..., description="Multi-page PDF claim document"),
) -> JSONResponse:
    """
    Upload a multi-page PDF insurance claim and receive structured extracted data.

    The pipeline:
    1. Segregator classifies every page into one of 9 document types.
    2. Three specialist agents extract data in parallel from their assigned pages.
    3. Aggregator combines all results into a single JSON response.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info("Received claim %s | file=%s | size=%d bytes", claim_id, file.filename, len(pdf_bytes))

    # Build initial state
    initial_state = ClaimState(
        claim_id=claim_id,
        pdf_bytes=pdf_bytes,
    )

    try:
        final_state = claim_graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Pipeline failed for claim %s: %s", claim_id, exc)
        raise HTTPException(status_code=500, detail=f"Processing pipeline error: {exc}")

    # Extract state from LangGraph's result (could be dict or other type)
    if isinstance(final_state, dict):
        state_dict = final_state
    else:
        # Try to convert to dict
        try:
            state_dict = dict(final_state) if hasattr(final_state, '__iter__') else final_state.__dict__
        except Exception:
            state_dict = {}

    logger.info(
        "Pipeline completed for claim %s | final_state_type=%s | keys=%s",
        claim_id,
        type(final_state).__name__,
        sorted(state_dict.keys()),
    )
    
    # Retrieve the ClaimResponse built by the aggregator node
    response: ClaimResponse | None = state_dict.get("final_response") or state_dict.get("_response")
    if response is None:
        # Fallback — should not normally happen
        errors = state_dict.get("errors", [])
        logger.error(
            "Aggregator response missing for claim %s | errors=%s | state_snapshot=%s",
            claim_id,
            errors,
            {
                "has_segregator_output": bool(state_dict.get("segregator_output")),
                "has_identity_data": state_dict.get("identity_data") is not None,
                "has_discharge_data": state_dict.get("discharge_data") is not None,
                "has_bill_data": state_dict.get("bill_data") is not None,
            },
        )
        if not errors or (isinstance(errors, list) and len(errors) == 0):
            errors = ["Aggregator did not produce a response."]
        response = ClaimResponse(
            claim_id=claim_id,
            status="failed",
            errors=errors if isinstance(errors, list) else [str(errors)],
        )
    else:
        logger.info(
            "Returning response for claim %s | status=%s | extracted_sections=%s | errors=%d",
            claim_id,
            response.status,
            sorted(response.extracted_data.keys()),
            len(response.errors),
        )

    return JSONResponse(content=response.model_dump())
