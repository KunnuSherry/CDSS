"""
Clinical Decision Support (CDSS) API — patient-specific advisory output.

SAFETY: No treatment recommendations; no AI-based decision-making.
Scores quantify risk; guidelines contextualize risk; cases show precedent. Doctors decide.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_roles
from models.clinical import (
    ClinicalOutput,
    GuidelineExcerpt,
    PatientClinicalInput,
    PatientSummary,
)
from models.db import get_db
from services.case_studies import get_case_studies_for_profile
from services.grace import GraceValidationError, compute_grace_score
from services.guideline_mapping import get_guideline_queries_for_grace
from services.rag import (
    groq_cleanup_chunks,
    is_non_clinical_noise,
    normalize_query_for_retrieval,
    retrieve_chunks,
    compute_final_chunk_scores,
)
from services.rag import (
    SECTION_DIAGNOSTIC,
    SECTION_HARM,
    SECTION_MANAGEMENT,
    SECTION_RISK,
)

router = APIRouter()

import httpx
from settings import settings

# Hard display limits: at most 5 guideline blocks visible. Rest behind "Show supporting evidence (advanced)".
# Diagnostic 1, Risk stratification 1, Management 2, Harm 1.
MAX_PRIMARY_BY_SECTION: dict[str, int] = {
    SECTION_DIAGNOSTIC: 1,
    SECTION_RISK: 1,
    SECTION_MANAGEMENT: 2,
    SECTION_HARM: 1,
}
MAX_PRIMARY_TOTAL = 5
MAX_CASE_STUDIES = 3

# Exact disclaimer (non-negotiable).
DISCLAIMER_TEXT = (
    "Scores quantify risk; guidelines contextualize risk; cases show precedent. "
    "Final decisions rest with the treating clinician."
)


def _to_excerpt(raw: dict, cleaned_map: dict) -> GuidelineExcerpt:
    cid = raw.get("chunk_id", "")
    text = cleaned_map.get(cid, raw.get("text", ""))
    return GuidelineExcerpt(
        chunk_id=cid,
        pdf_name=raw.get("pdf_name", ""),
        doc_id=raw.get("doc_id", ""),
        page=int(raw.get("page", 0) or 0),
        text=text,
        heading=raw.get("heading"),
        section_label=raw.get("_section_key"),
        section_title=raw.get("_section_title"),
    )


async def _retrieve_guideline_excerpts_for_queries(
    queries: list[str],
) -> tuple[list[GuidelineExcerpt], list[GuidelineExcerpt]]:
    """
    Run RAG for each query; filter non-clinical noise; enforce section limits.
    Returns (primary_excerpts, supporting_evidence). Primary at most 5 blocks:
    diagnostic 1, risk 1, management 2, harm 1. Rest go to supporting_evidence.
    """
    db = get_db()
    documents = []
    async for d in db["documents"].find({}):
        documents.append(d)

    seen_chunk_ids: set[str] = set()
    all_candidates: list[dict] = []

    # Collect candidates from all documents for all queries
    for query in queries:
        normalized = normalize_query_for_retrieval(query)
        for d in documents:
            candidates = retrieve_chunks(
                normalized,
                collection_name=f"pdf_{d['_id']}",
                n_results=32,
            )
            for c in candidates:
                # attach originating query to candidate so scoring may use it
                c["_retrieval_query"] = normalized
                all_candidates.append(c)

    # Filter non-clinical noise early
    filtered = []
    for c in all_candidates:
        cid = c.get("chunk_id") or ""
        if not cid or cid in seen_chunk_ids:
            continue
        if is_non_clinical_noise(c, disease="NSTEMI"):
            continue
        seen_chunk_ids.add(cid)
        filtered.append(c)

    if not filtered:
        return [], []

    # Rank each chunk independently using the final_score formula
    scored = compute_final_chunk_scores(filtered, query=queries[0] if queries else "")

    # Now enforce hard display limits by section using scores (select top-N per section,
    # but overall cap of MAX_PRIMARY_TOTAL)
    section_counts: dict[str, int] = {k: 0 for k in MAX_PRIMARY_BY_SECTION}
    primary_raw: list[dict] = []
    supporting_raw: list[dict] = []

    for item in scored:
        skey = item.get("section_key") or ""
        # only accept primary if section cap not reached
        if skey in MAX_PRIMARY_BY_SECTION:
            cap = MAX_PRIMARY_BY_SECTION[skey]
            if section_counts[skey] < cap and len(primary_raw) < MAX_PRIMARY_TOTAL:
                primary_raw.append(item)
                section_counts[skey] += 1
            else:
                supporting_raw.append(item)
        else:
            supporting_raw.append(item)

    # normalize section keys to previous underscore style for _to_excerpt
    for r in primary_raw + supporting_raw:
        if "section_key" in r:
            r["_section_key"] = r.get("section_key")
        if "section_title" in r:
            r["_section_title"] = r.get("section_title")

    cleaned_map = await groq_cleanup_chunks(primary_raw + supporting_raw)

    primary = [_to_excerpt(r, cleaned_map) for r in primary_raw]
    supporting = [_to_excerpt(r, cleaned_map) for r in supporting_raw]
    return primary, supporting


@router.post("/cdss", response_model=ClinicalOutput)
async def patient_decision_support(
    body: PatientClinicalInput,
    _user=Depends(require_roles("admin", "doctor")),
) -> ClinicalOutput:
    """
    Patient-specific decision support for NSTEMI.
    Order: 1. Patient summary, 2. GRACE + risk, 3. Guideline excerpts (≤5), 4. Case studies (≤4), 5. Disclaimer.
    """
    try:
        grace_result = compute_grace_score(body)
    except GraceValidationError as e:
        # Block downstream when GRACE validation fails (deviation >3 from expected).
        raise HTTPException(status_code=503, detail=str(e)) from e

    queries = get_guideline_queries_for_grace(
        grace_result.risk_category,
        total_grace_score=grace_result.total_score,
    )
    guideline_excerpts, supporting_evidence = await _retrieve_guideline_excerpts_for_queries(queries)

    case_studies = get_case_studies_for_profile(
        risk_level=grace_result.risk_category,
        max_results=MAX_CASE_STUDIES,
        grace_total=grace_result.total_score,
    )

    patient_summary = PatientSummary(
        age=body.age,
        heart_rate=body.heart_rate,
        systolic_bp=body.systolic_bp,
        serum_creatinine=body.serum_creatinine,
        killip_class=body.killip_class,
        cardiac_arrest_at_admission=body.cardiac_arrest_at_admission,
        st_deviation_ecg=body.st_deviation_ecg,
        elevated_cardiac_enzymes=body.elevated_cardiac_enzymes,
        disease_context="NSTEMI",
    )

    return ClinicalOutput(
        patient_summary=patient_summary,
        grace_score=grace_result,
        guideline_excerpts=guideline_excerpts,
        supporting_evidence=supporting_evidence,
        case_studies=case_studies,
        disclaimer=DISCLAIMER_TEXT,
    )


@router.post("/cdss/explain_chunk")
async def explain_chunk(
    chunk: GuidelineExcerpt,
    _user=Depends(require_roles("admin", "doctor")),
) -> dict:
    """
    Controlled AI explanation endpoint.

    - Runs ONLY on explicit user click and accepts a single chunk.
    - Returns up to 5 bullet points in plain clinical language.
    - MUST NOT contain treatment recommendations or prescriptive language.
    - If API fails or output contains recommendation language, return unavailable.
    """
    if not settings.groq_api_key:
        return {"error": "AI explanation unavailable"}

    text = (chunk.text or "").strip()
    if not text:
        return {"error": "No chunk text provided"}

    system = (
        "You are an objective clinical assistant. Produce up to 5 concise bullet points "
        "that restate only the factual content of the INPUT TEXT in plain clinical language. "
        "Do NOT provide recommendations, suggestions, or guidance (do not use words like 'should', 'recommend', 'consider', 'must'). "
        "Do not add any information not present in the input. Output JSON: {\"bullets\": [..]}"
    )

    user_prompt = f"INPUT:\n""\"\n{text}\n\"\"\n"

    try:
        async with httpx.AsyncClient(base_url=settings.groq_base_url, timeout=20) as client:
            resp = await client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                json={
                    "model": settings.groq_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return {"error": "AI explanation unavailable"}

    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    try:
        # Expect JSON from model
        parsed = content and __import__("json").loads(content)
    except Exception:
        return {"error": "AI explanation unavailable"}

    bullets = parsed.get("bullets") if isinstance(parsed, dict) else None
    if not bullets or not isinstance(bullets, list):
        return {"error": "AI explanation unavailable"}

    # Post-validate: ensure no recommendation words
    forbidden = ("should", "recommend", "consider", "must", "ought")
    for b in bullets:
        lb = (b or "").lower()
        if any(f in lb for f in forbidden):
            return {"error": "AI explanation unavailable"}

    # Truncate to max 5 bullets and return
    return {"bullets": [str(b).strip() for b in bullets][:5]}
