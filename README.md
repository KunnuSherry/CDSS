# Medical Guideline Study Platform (Scaffold)

This repo scaffolds a **study & reference** platform for medical guideline PDFs.

**Important**: This is **NOT** a chatbot and **NOT** a medical decision system.
Outputs must be **retrieval-first** and **extractive** (verbatim from uploaded PDFs).

## Structure

- `backend/` FastAPI + MongoDB Atlas + ChromaDB
- `frontend/` React (Vite) UI (simple pages)

## Backend setup

1) Create `backend/.env` from `backend/.env.example`.

2) Install deps:

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

3) Run:

```bash
uvicorn main:app --reload --port 8000
```

Backend will serve extracted images at:
- `http://localhost:8000/uploads/...`

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Frontend dev server proxies `/api` and `/uploads` to `http://localhost:8000`.

## Quick test flow

1) Start backend.
2) Start frontend.
3) Go to Landing → **Admin**
4) Signup/Login as admin → upload a PDF
5) Go to Landing → **Doctor**
6) Signup/Login as doctor → search:
   - `NSTEMI`
   - `Acute coronary syndrome`

## Reality checks (required)

- Verify extracted text matches the PDF text layer **exactly**.
- Verify images are linked to **correct page numbers**.
- If content is missing, the doctor endpoint should return:
  - **“Not found in uploaded documents”**

## Phase 1 – Clinician-facing RAG behavior

Phase 1 turns the scaffold into a clinically usable guideline viewer with strict separation between **recall (RAG)** and **explanation (LLM)**.

- **RAG retrieval (backend)**
  - Uses Chroma + local sentence-transformer embeddings to retrieve guideline chunks per uploaded PDF.
  - Normalizes clinician queries (e.g. “high NSTEMI” → `NSTEMI risk`) so retrieval is **semantic**, not literal.
  - Hard-filters non-clinical pages (title pages, writing committees, affiliations, methods, DOI/journal metadata, pure references) so they never appear in results.
  - Assigns each candidate chunk to one of five clinical sections: **Definition/Scope**, **Diagnostic criteria**, **Risk stratification**, **Management**, **Harm/Contraindications** using headings + deterministic keyword rules.
  - Computes a confidence score per chunk (0–1) based on: semantic similarity, section-relevant keywords, text completeness, and page position; selects only the top chunks per section (max 1/2/2/3/1) so a query never returns more than ~9 chunks.
  - Suppressed chunks are not sent to the frontend; coverage notes and images are based only on the selected (visible) chunks.

- **Text repair with Groq (“Grok”)**
  - Backend calls Groq **only to fix PDF artefacts** (mid-word starts, hyphenated line breaks, sentence splits) after section + confidence filtering.
  - Uses an explicit “repair only” prompt: preserve meaning exactly, no summarization, no new facts, fix only formatting and broken words.
  - Cleanup is applied per-chunk, only when heuristics indicate broken text; failures silently fall back to the original text, so retrieval remains fully usable without LLM.

- **Doctor UI (frontend)**
  - Left panel shows cleaned verbatim guideline chunks, grouped by page and/or clinical section; images from matching pages are shown with page numbers.
  - Each chunk has a **“Summarize with AI”** button (bluish/purplish, with subtle hover elevation) that sends only that chunk’s text (truncated to ~1500 chars) to Groq for a plain‑language explanation.
  - Right panel (`AI Explanation (Plain Language)`) stays empty until a chunk is selected; it explains **only** the selected chunk and always includes the disclaimer:  
    “AI-assisted explanation based only on the selected guideline excerpt.”
  - Frontend uses `VITE_GROQ_API_KEY`; AI is never called automatically, and simple safety checks discard explanations that contain obviously prescriptive phrases (e.g. “patients should”, “it is recommended”) instead of showing them.

The net effect of Phase 1 is that doctors see a small, high-confidence set of readable guideline excerpts as the **source of truth**, with optional, tightly-constrained AI assistance that never replaces or overrides the guideline text.

