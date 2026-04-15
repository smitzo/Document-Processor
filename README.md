# Claim Processing Pipeline

FastAPI + LangGraph service for medical claim PDF processing.

This project accepts a PDF claim document, classifies each page into a document type, routes only the relevant pages to specialist extraction agents, and returns one aggregated JSON response.

## Workflow

The LangGraph workflow follows this structure:

`START -> Segregator -> [ID Agent, Discharge Summary Agent, Itemized Bill Agent] -> Aggregator -> END`

### Node Responsibilities

- `Segregator`
  Classifies PDF pages into the required document types:
  `claim_forms`, `cheque_or_bank_details`, `identity_document`, `itemized_bill`, `discharge_summary`, `prescription`, `investigation_report`, `cash_receipt`, `other`

- `ID Agent`
  Processes only pages related to identity, claim forms, and bank details.
  Extracts patient, policy, and bank-related fields.

- `Discharge Summary Agent`
  Processes only discharge summary and prescription pages.
  Extracts diagnosis, admission/discharge dates, physician, and medication details.

- `Itemized Bill Agent`
  Processes only bill, receipt, and investigation pages.
  Extracts billing line items and totals.

- `Aggregator`
  Merges all outputs into one final API response.

## How It Works

### 1. API Input

The service exposes:

- `POST /api/process`

Inputs:

- `claim_id`: string
- `file`: PDF upload

Output:

- JSON with page classifications, extracted structured data, errors, and a processing summary.

### 2. Segregation

The segregator first extracts page text from the PDF and renders page images.

To improve reliability, segregation uses a hybrid strategy:

- local text heuristics first for obvious pages
- Gemini only for pages that are still ambiguous
- fallback page-level classification if a chunk fails

This reduces API usage and keeps the workflow stable for larger PDFs.

### 3. Page Routing

After classification, only relevant pages are sent to each extraction agent:

- `ID Agent` gets `claim_forms`, `cheque_or_bank_details`, `identity_document`
- `Discharge Agent` gets `discharge_summary`, `prescription`
- `Bill Agent` gets `itemized_bill`, `cash_receipt`, `investigation_report`

This follows the assignment rule that extraction agents should not process the whole PDF.

### 4. Extraction

Each extraction agent now uses a text-first strategy:

- deterministic text extraction is attempted first
- Gemini is used only if needed

This makes the pipeline more stable when model JSON output is incomplete or rate-limited.

### 5. Aggregation

The aggregator combines:

- page classifications
- extracted identity data
- extracted discharge summary data
- extracted bill data
- processing summary
- error list

Then it returns one final `ClaimResponse`.

## Project Structure

```text
app/
  agents/
    segregator.py
    id_agent.py
    discharge_agent.py
    bill_agent.py
    aggregator.py
  api/
    routes.py
  core/
    graph.py
    schemas.py
    config.py
  utils/
    llm_client.py
    pdf_utils.py
    text_extractors.py
main.py
requirements.txt
```

## Tech Stack

- FastAPI
- LangGraph
- Pydantic
- PyMuPDF
- Google Gemini API

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add environment variables

Create `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
APP_ENV=development
LOG_LEVEL=INFO
```

### 3. Run the server

```bash
uvicorn main:app --reload
```

### 4. Open Swagger UI

```text
http://127.0.0.1:8000/docs
```

## Example API Flow

1. User uploads a claim PDF.
2. The segregator classifies every page.
3. Relevant pages are routed to the three extraction agents.
4. Each agent extracts structured information from only its assigned pages.
5. The aggregator combines everything into one JSON response.

## Notes

- Small and text-based PDFs work best because deterministic extraction can recover fields even if LLM JSON is incomplete.
- Large PDFs may still depend on Gemini quota and rate limits.
- The system logs every stage to make debugging easier.

## Submission Summary

This project satisfies the assignment requirements by:

- exposing `POST /api/process`
- using LangGraph orchestration
- including one AI-powered segregator node
- routing only relevant pages to the extraction agents
- using three required extraction agents
- aggregating all outputs into one final JSON response
