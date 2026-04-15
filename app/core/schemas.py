"""
Pydantic schemas for the claim processing pipeline.
"""

from __future__ import annotations
import operator
from typing import Annotated, Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Document type literals
# ---------------------------------------------------------------------------

DOCUMENT_TYPES = [
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]

# Document types handled by extraction agents
AGENT_DOCUMENT_MAP = {
    "identity_document": "id_agent",
    "claim_forms": "id_agent",          # claim forms contain patient identity info
    "discharge_summary": "discharge_agent",
    "prescription": "discharge_agent",  # prescriptions tie to discharge summary
    "itemized_bill": "bill_agent",
    "cash_receipt": "bill_agent",       # receipts tie to billing
    "investigation_report": "bill_agent",  # lab reports often accompany billing
    "cheque_or_bank_details": "id_agent",
    "other": "id_agent",
}


# ---------------------------------------------------------------------------
# Segregator output
# ---------------------------------------------------------------------------

class PageClassification(BaseModel):
    page_number: int
    document_type: str
    confidence: str = Field(default="high", description="high | medium | low")
    description: str = ""


class SegregatorOutput(BaseModel):
    pages: list[PageClassification] = Field(default_factory=list)
    # Maps document_type → list of 0-indexed page numbers
    document_page_map: dict[str, list[int]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extraction agent outputs
# ---------------------------------------------------------------------------

class IdentityData(BaseModel):
    patient_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    blood_group: Optional[str] = None
    address: Optional[str] = None
    contact_number: Optional[str] = None
    email: Optional[str] = None
    id_number: Optional[str] = None
    policy_number: Optional[str] = None
    insurance_provider: Optional[str] = None
    bank_account_number: Optional[str] = None
    bank_name: Optional[str] = None
    ifsc_routing_number: Optional[str] = None
    swift_code: Optional[str] = None
    extra_fields: dict[str, Any] = Field(default_factory=dict)


class DischargeSummaryData(BaseModel):
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None
    length_of_stay: Optional[str] = None
    admission_diagnosis: Optional[str] = None
    final_diagnosis: Optional[str] = None
    attending_physician: Optional[str] = None
    hospital_name: Optional[str] = None
    mrn: Optional[str] = None
    treatment_summary: Optional[str] = None
    discharge_medications: list[str] = Field(default_factory=list)
    follow_up_instructions: Optional[str] = None
    condition_at_discharge: Optional[str] = None
    prescriptions: list[dict[str, Any]] = Field(default_factory=list)
    extra_fields: dict[str, Any] = Field(default_factory=dict)


class BillLineItem(BaseModel):
    date: Optional[str] = None
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: float


class ItemizedBillData(BaseModel):
    bill_number: Optional[str] = None
    bill_date: Optional[str] = None
    hospital_name: Optional[str] = None
    patient_name: Optional[str] = None
    line_items: list[BillLineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    discount: Optional[float] = None
    insurance_payment: Optional[float] = None
    total_amount: Optional[float] = None
    patient_responsibility: Optional[float] = None
    payment_method: Optional[str] = None
    extra_fields: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline graph state
# ---------------------------------------------------------------------------

class ClaimState(BaseModel):
    """Shared state passed between LangGraph nodes."""
    claim_id: str
    pdf_bytes: bytes = b""
    total_pages: int = 0

    # Segregator output
    segregator_output: Optional[SegregatorOutput] = None

    # Pages as base64 strings keyed by 0-index — populated after segregation
    page_images: dict[int, str] = Field(default_factory=dict)   # page_index → base64 PNG
    page_texts: dict[int, str] = Field(default_factory=dict)    # page_index → extracted text

    # Agent outputs
    identity_data: Optional[IdentityData] = None
    discharge_data: Optional[DischargeSummaryData] = None
    bill_data: Optional[ItemizedBillData] = None
    final_response: Optional["ClaimResponse"] = None

    # Errors collected during processing
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# API response
# ---------------------------------------------------------------------------

class ClaimResponse(BaseModel):
    claim_id: str
    status: str
    page_classification: list[dict[str, Any]] = Field(default_factory=list)
    extracted_data: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    processing_summary: dict[str, Any] = Field(default_factory=dict)


ClaimState.model_rebuild()
