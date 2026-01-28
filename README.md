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

## TODO (next steps)

- Add OCR (optional) for scanned PDFs (current extraction is **no OCR**).
- Add per-PDF search scoping + “search across all PDFs” toggle.
- Improve Chroma scoring by including distances and proper global ranking.
- Harden auth (rate limiting, password policy, admin creation policy).

