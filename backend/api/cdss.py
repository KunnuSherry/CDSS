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
    build_sectioned_display,
    groq_cleanup_chunks,
    is_non_clinical_noise,
    normalize_query_for_retrieval,
    retrieve_chunks,
)
from services.rag import (
    SECTION_DIAGNOSTIC,
    SECTION_HARM,
    SECTION_MANAGEMENT,
    SECTION_RISK,
)

router = APIRouter()

# Hard display limits: at most 5 guideline blocks visible. Rest behind "Show supporting evidence (advanced)".
# Diagnostic 1, Risk stratification 1, Management 2, Harm 1.
MAX_PRIMARY_BY_SECTION: dict[str, int] = {
    SECTION_DIAGNOSTIC: 1,
    SECTION_RISK: 1,
    SECTION_MANAGEMENT: 2,
    SECTION_HARM: 1,
}
MAX_PRIMARY_TOTAL = 5
MAX_CASE_STUDIES = 4

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
    all_selected_raw: list[dict] = []

    for query in queries:
        normalized = normalize_query_for_retrieval(query)
        all_candidates = []
        for d in documents:
            candidates = retrieve_chunks(
                normalized,
                collection_name=f"pdf_{d['_id']}",
                n_results=32,
            )
            all_candidates.extend(candidates)

        section_groups_meta, selected_chunks_raw = build_sectioned_display(
            query=normalized,
            chunks=all_candidates,
        )

        for g in section_groups_meta:
            skey = g.get("section_key", "")
            stitle = g.get("section_title", "")
            for item in g.get("items", []):
                cid = item.get("chunk_id") or ""
                if not cid or cid in seen_chunk_ids:
                    continue
                # PART 4: Filter non-clinical noise at backend (TOC, STEMI system-of-care, etc.).
                if is_non_clinical_noise(item, disease="NSTEMI"):
                    continue
                seen_chunk_ids.add(cid)
                item["_section_key"] = skey
                item["_section_title"] = stitle
                all_selected_raw.append(item)

    if not all_selected_raw:
        return [], []

    cleaned_map = await groq_cleanup_chunks(all_selected_raw)

    # Enforce hard display limits: diagnostic 1, risk 1, management 2, harm 1.
    section_counts: dict[str, int] = {k: 0 for k in MAX_PRIMARY_BY_SECTION}
    primary_raw: list[dict] = []
    supporting_raw: list[dict] = []

    for raw in all_selected_raw:
        skey = raw.get("_section_key") or ""
        if skey in MAX_PRIMARY_BY_SECTION:
            cap = MAX_PRIMARY_BY_SECTION[skey]
            if section_counts[skey] < cap and len(primary_raw) < MAX_PRIMARY_TOTAL:
                primary_raw.append(raw)
                section_counts[skey] += 1
            else:
                supporting_raw.append(raw)
        else:
            supporting_raw.append(raw)

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
