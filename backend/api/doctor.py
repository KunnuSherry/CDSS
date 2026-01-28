from fastapi import APIRouter, Depends

from api.auth import require_roles
from models.db import get_db
from models.document import ImagePublic, SearchResponse, ChunkPublic, SectionGroup
from services.rag import (
    retrieve_chunks,
    _is_recommendation_chunk,
    build_coverage_note,
    groq_cleanup_chunks,
    build_sectioned_display,
)


def _sort_key(c: dict) -> tuple[int, float]:
    rec = 0 if _is_recommendation_chunk(c.get("text") or "") else 1
    return (rec, c.get("distance") or 0.0)


router = APIRouter()


def _normalize_query_for_retrieval(query: str) -> str:
    """
    Normalize clinician-facing queries into retrieval-friendly forms.

    Examples:
    - "high NSTEMI" -> "NSTEMI risk"
    - "severe ACS" -> "ACS risk"
    - "management of NSTEMI" -> "NSTEMI management"

    This is semantic, not literal: we avoid overfitting to exact phrases in the PDF.
    """
    q = (query or "").strip()
    if not q:
        return q

    uq = q.upper()
    has_risk_modifier = any(w in uq for w in ("HIGH", "LOW", "SEVERE", "MILD", "RISK"))

    # Normalize "management of X" / "treatment of X" -> "X management"
    lowered = q.lower()
    for prefix in ("management of ", "treatment of "):
        if lowered.startswith(prefix):
            rest = q[len(prefix) :].strip()
            if rest:
                return f"{rest} management"

    if "NSTEMI" in uq:
        if has_risk_modifier:
            return "NSTEMI risk"
        return "NSTEMI"
    if "STEMI" in uq:
        if has_risk_modifier:
            return "STEMI risk"
        return "STEMI"
    if "ACS" in uq:
        if has_risk_modifier:
            return "ACS risk"
        return "ACS"

    return q


@router.get("/search", response_model=SearchResponse)
async def search(query: str, _user=Depends(require_roles("admin", "doctor"))):
    """
    Retrieval-first search across all uploaded PDFs.
    Returns section-grouped extractive chunks + images + coverage note.
    """
    db = get_db()

    # Normalize clinician phrasing into retrieval-friendly semantics without
    # requiring exact phrases from the PDF.
    normalized_query = _normalize_query_for_retrieval(query)

    documents = []
    async for d in db["documents"].find({}):
        documents.append(d)

    all_candidates = []
    for d in documents:
        # Retrieval is semantic (embeddings), not literal string match. We use
        # the normalized query here but always return the clinician's original
        # query in the API response.
        candidates = retrieve_chunks(normalized_query, collection_name=f"pdf_{d['_id']}", n_results=32)
        all_candidates.extend(candidates)

    # Section-aware organization with confidence scoring and hard per-section
    # limits. This function:
    # - assigns each chunk to exactly one of the canonical sections
    # - computes a confidence score in [0, 1]
    # - selects only the highest-confidence chunks per section
    # - suppresses all remaining chunks (they are never sent to the frontend).
    section_groups_meta, selected_chunks_raw = build_sectioned_display(
        query=normalized_query,
        chunks=all_candidates,
    )

    # Groq is used ONLY to clean up PDF artifacts in chunk text BEFORE display.
    # It is never used for selection, summarization, or interpretation.
    cleaned_map = await groq_cleanup_chunks(selected_chunks_raw)

    def _to_chunk_public(raw: dict) -> ChunkPublic:
        cid = raw["chunk_id"]
        text = cleaned_map.get(cid, raw.get("text", ""))
        return ChunkPublic(
            chunk_id=cid,
            pdf_name=raw.get("pdf_name", ""),
            doc_id=raw.get("doc_id", ""),
            page=int(raw.get("page", 0) or 0),
            text=text,
            heading=raw.get("heading"),
        )

    # Flattened list of all selected (high-confidence) chunks across sections.
    results: list[ChunkPublic] = [_to_chunk_public(c) for c in selected_chunks_raw]

    # Retrieval-first: do not gate on LLM answer. If retrieval finds nothing at all,
    # the coverage note will communicate that no verbatim excerpts matched.
    note = None

    # Build section groups for the response, including standardized placeholders
    # when no high-confidence content exists for a section.
    groups: list[SectionGroup] | None = []
    for g in section_groups_meta:
        section_label = g["section_key"]
        section_title = g["section_title"]
        raw_items = g["items"]

        if g["placeholder"]:
            # Per spec: do NOT invent content. We surface a standardized
            # message so the frontend can clearly indicate absence.
            placeholder = ChunkPublic(
                chunk_id=f"__placeholder__{section_label}",
                pdf_name="",
                doc_id="",
                page=0,
                text="No clear standalone content found for this section in this guideline.",
                heading=None,
            )
            groups.append(
                SectionGroup(
                    section_label=section_label,
                    section_title=section_title,
                    items=[placeholder],
                )
            )
            continue

        items = [_to_chunk_public(x) for x in raw_items]
        if not items:
            continue
        groups.append(
            SectionGroup(
                section_label=section_label,
                section_title=section_title,
                items=items,
            )
        )

    if not groups:
        groups = None

    # Coverage note is based on the pages represented in the selected chunks
    # only. Suppressed chunks never influence the note.
    pages_for_note = sorted({r.page for r in results if r.page > 0})
    coverage_note: str | None = build_coverage_note(pages_for_note, query) if results else None
    source_pdf: str | None = results[0].pdf_name if results else None

    pages_set = {(r.doc_id, r.page) for r in results}
    images_out: list[ImagePublic] = []
    if pages_set:
        query_or = [{"doc_id": doc_id, "page": page} for (doc_id, page) in pages_set]
        async for im in db["images"].find({"$or": query_or}):
            images_out.append(
                ImagePublic(
                    url=f"/uploads/{im['rel_path']}",
                    page=im["page"],
                    image_index=im["image_index"],
                )
            )

    return SearchResponse(
        query=query,
        results=results,
        advanced_results=None,
        images=images_out,
        note=note,
        groups=groups,
        coverage_note=coverage_note,
        source_pdf=source_pdf,
    )

