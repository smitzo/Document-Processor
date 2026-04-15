"""Deterministic text extraction helpers for common claim document fields."""

from __future__ import annotations

import re
from typing import Any

from app.core.schemas import BillLineItem, DischargeSummaryData, IdentityData, ItemizedBillData


def extract_identity_data_from_text(text: str) -> IdentityData:
    # These helpers are deterministic fallbacks for text-based PDFs.
    compact = _compact(text)
    lines = _lines(text)
    return IdentityData(
        patient_name=_search(compact, [
            r'"patient_name"\s*:\s*"([^"]+)"',
            r"patient name[:\s]+([A-Z][A-Za-z ,.'-]{2,})",
            r"name of patient[:\s]+([A-Z][A-Za-z ,.'-]{2,})",
            r"name[:\s]+([A-Z][A-Za-z ,.'-]{2,})",
        ]),
        date_of_birth=_search(compact, [
            r'"date_of_birth"\s*:\s*"([^"]+)"',
            r"(?:date of birth|dob)[:\s]+([A-Za-z0-9,/\- ]{4,})",
        ]),
        gender=_search(compact, [
            r'"gender"\s*:\s*"([^"]+)"',
            r"gender[:\s]+([A-Za-z]+)",
            r"sex[:\s]+([A-Za-z]+)",
        ]),
        blood_group=_search(compact, [
            r"(?:blood group|blood type)[:\s]+([ABO]{1,2}[+-])",
        ]),
        address=_search(compact, [
            r'"address"\s*:\s*"([^"]+)"',
            r"address[:\s]+([A-Za-z0-9,.\- ]{8,})",
        ]) or _search_following_line(lines, "address"),
        contact_number=_search(compact, [
            r'"contact_number"\s*:\s*"([^"]+)"',
            r"(?:contact number|phone|mobile|telephone)[:\s]+([+\d][\d\- ]{7,})",
        ]),
        email=_search(compact, [
            r'"email"\s*:\s*"([^"]+)"',
            r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
        ]),
        id_number=_search(compact, [
            r'"id_number"\s*:\s*"([^"]+)"',
            r"(?:id number|identification number|document number)[:\s]+([A-Za-z0-9\-]{4,})",
        ]),
        policy_number=_search(compact, [
            r'"policy_number"\s*:\s*"([^"]+)"',
            r"(?:policy number|policy no|policy #)[:\s]+([A-Za-z0-9\-]{4,})",
        ]),
        insurance_provider=_search(compact, [
            r'"insurance_provider"\s*:\s*"([^"]+)"',
            r"(?:insurance provider|insurance company|insurer)[:\s]+([A-Za-z0-9 &.\-]{4,})",
        ]),
        bank_account_number=_search(compact, [
            r'"bank_account_number"\s*:\s*"([^"]+)"',
            r"(?:account number|a/c number|bank account)[:\s]+([A-Za-z0-9\-]{6,})",
        ]),
        bank_name=_search(compact, [
            r'"bank_name"\s*:\s*"([^"]+)"',
            r"bank name[:\s]+([A-Za-z0-9 &.\-]{3,})",
        ]),
        ifsc_routing_number=_search(compact, [
            r'"ifsc_routing_number"\s*:\s*"([^"]+)"',
            r"(?:ifsc|routing number)[:\s]+([A-Za-z0-9\-]{4,})",
        ]),
        swift_code=_search(compact, [
            r'"swift_code"\s*:\s*"([^"]+)"',
            r"swift(?: code)?[:\s]+([A-Za-z0-9\-]{4,})",
        ]),
        extra_fields={},
    )


def has_identity_signal(data: IdentityData) -> bool:
    return bool(
        (data.patient_name and (data.date_of_birth or data.policy_number or data.id_number))
        or (data.bank_account_number and (data.bank_name or data.ifsc_routing_number))
    )


def extract_discharge_data_from_text(text: str) -> DischargeSummaryData:
    compact = _compact(text)
    discharge_medications = _extract_list_items(text, ["discharge medications", "medications"])
    return DischargeSummaryData(
        admission_date=_search(compact, [r"(?:admission date|date of admission)[:\s]+([A-Za-z0-9,/\- ]{4,})"]),
        discharge_date=_search(compact, [r"(?:discharge date|date of discharge)[:\s]+([A-Za-z0-9,/\- ]{4,})"]),
        length_of_stay=_search(compact, [r"(?:length of stay|stay)[:\s]+([A-Za-z0-9 ]{2,})"]),
        admission_diagnosis=_search(compact, [r"(?:admission diagnosis|diagnosis on admission)[:\s]+(.+?)(?:final diagnosis|attending physician|hospital name|mrn|$)"]),
        final_diagnosis=_search(compact, [r"(?:final diagnosis|diagnosis at discharge)[:\s]+(.+?)(?:attending physician|hospital name|mrn|$)"]),
        attending_physician=_search(compact, [r"(?:attending physician|physician|doctor)[:\s]+([A-Za-z ,.]+)"]),
        hospital_name=_search(compact, [r"(?:hospital name|hospital)[:\s]+([A-Za-z0-9 &.\-]{3,})"]),
        mrn=_search(compact, [r"(?:mrn|medical record number)[:\s]+([A-Za-z0-9\-]{3,})"]),
        treatment_summary=_search(compact, [r"(?:treatment summary|hospital course|course in hospital)[:\s]+(.+?)(?:discharge medications|follow up|condition at discharge|$)"]),
        discharge_medications=discharge_medications,
        follow_up_instructions=_search(compact, [r"(?:follow up instructions|follow-up instructions|follow up)[:\s]+(.+?)(?:condition at discharge|$)"]),
        condition_at_discharge=_search(compact, [r"(?:condition at discharge)[:\s]+([A-Za-z ,.-]{3,})"]),
        prescriptions=_extract_prescriptions(text),
        extra_fields={},
    )


def has_discharge_signal(data: DischargeSummaryData) -> bool:
    signals = [
        data.admission_date,
        data.discharge_date,
        data.final_diagnosis,
        data.attending_physician,
        data.hospital_name,
        data.mrn,
        data.treatment_summary,
    ]
    return sum(1 for item in signals if item) >= 2 or bool(data.prescriptions)


def extract_bill_data_from_text(text: str) -> ItemizedBillData:
    compact = _compact(text)
    line_items = _extract_bill_line_items(text)
    total_amount = _search_number(compact, [r"(?:total amount|grand total|amount paid|total)[:\s$]*([0-9]+(?:\.[0-9]{1,2})?)"])
    subtotal = _search_number(compact, [r"subtotal[:\s$]*([0-9]+(?:\.[0-9]{1,2})?)"])
    tax = _search_number(compact, [r"tax[:\s$]*([0-9]+(?:\.[0-9]{1,2})?)"])
    return ItemizedBillData(
        bill_number=_search(compact, [r"(?:bill number|invoice number|receipt number|bill no)[:\s]+([A-Za-z0-9\-\/]{3,})"]),
        bill_date=_search(compact, [r"(?:bill date|invoice date|receipt date|date)[:\s]+([A-Za-z0-9,/\- ]{4,})"]),
        hospital_name=_search(compact, [r"(?:hospital name|hospital|medical center|clinic)[:\s]+([A-Za-z0-9 &.\-]{3,})"]),
        patient_name=_search(compact, [r"(?:patient name|name of patient)[:\s]+([A-Z][A-Za-z ,.'-]{2,})"]),
        line_items=line_items,
        subtotal=subtotal,
        tax=tax,
        discount=_search_number(compact, [r"discount[:\s$]*([0-9]+(?:\.[0-9]{1,2})?)"]),
        insurance_payment=_search_number(compact, [r"(?:insurance payment|insurance paid|covered amount)[:\s$]*([0-9]+(?:\.[0-9]{1,2})?)"]),
        total_amount=total_amount if total_amount is not None else _sum_line_items(line_items),
        patient_responsibility=_search_number(compact, [r"(?:patient responsibility|amount due|patient payable)[:\s$]*([0-9]+(?:\.[0-9]{1,2})?)"]),
        payment_method=_search(compact, [r"(?:payment method|paid by|mode of payment)[:\s]+([A-Za-z ]{3,})"]),
        extra_fields={},
    )


def has_bill_signal(data: ItemizedBillData) -> bool:
    return bool(data.bill_number or data.total_amount or data.line_items)


def _extract_bill_line_items(text: str) -> list[BillLineItem]:
    lines = _lines(text)
    items: list[BillLineItem] = []
    money_re = re.compile(r"([0-9]+(?:\.[0-9]{1,2})?)$")
    date_re = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")

    for raw_line in lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if len(line) < 6:
            continue
        money_match = money_re.search(line)
        if not money_match:
            continue
        if any(token in line.lower() for token in ["total", "subtotal", "tax", "discount", "amount due"]):
            continue
        amount = float(money_match.group(1))
        prefix = line[:money_match.start()].strip(" -:")
        date = None
        date_match = date_re.search(prefix)
        if date_match:
            date = date_match.group(1)
            prefix = prefix.replace(date, "", 1).strip(" -:")
        if len(prefix) < 3:
            continue
        items.append(BillLineItem(description=prefix, amount=amount, date=date))
    return items


def _extract_prescriptions(text: str) -> list[dict[str, Any]]:
    lines = _lines(text)
    prescriptions: list[dict[str, Any]] = []
    for line in lines:
        normalized = re.sub(r"\s+", " ", line).strip()
        if len(normalized) < 5:
            continue
        if not any(token in normalized.lower() for token in ["mg", "tablet", "capsule", "ml", "times", "daily"]):
            continue
        prescriptions.append({
            "drug_name": normalized,
            "dosage": None,
            "frequency": None,
            "duration": None,
            "instructions": None,
        })
    return prescriptions[:10]


def _extract_list_items(text: str, labels: list[str]) -> list[str]:
    lines = _lines(text)
    results: list[str] = []
    capture = False
    for line in lines:
        lower = line.lower()
        if any(label in lower for label in labels):
            capture = True
            continue
        if capture:
            if ":" in line and len(results) > 0:
                break
            cleaned = line.strip(" -*\t")
            if cleaned:
                results.append(cleaned)
    return results[:10]


def _sum_line_items(line_items: list[BillLineItem]) -> float | None:
    if not line_items:
        return None
    total = sum(item.amount for item in line_items)
    return round(total, 2)


def _search(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" ,.-")
            if value:
                return value
    return None


def _search_number(text: str, patterns: list[str]) -> float | None:
    value = _search(text, patterns)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _search_following_line(lines: list[str], label: str) -> str | None:
    for index, line in enumerate(lines):
        if label.lower() in line.lower():
            stripped = re.sub(rf"^.*{re.escape(label)}[:\s]*", "", line, flags=re.IGNORECASE).strip(" ,.-")
            if stripped:
                return stripped
            if index + 1 < len(lines):
                nxt = lines[index + 1].strip(" ,.-")
                if nxt:
                    return nxt
    return None


def _compact(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]
