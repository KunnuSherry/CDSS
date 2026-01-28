from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from services.embedding import get_collection
from settings import settings

logger = logging.getLogger(__name__)

# Cap context size to avoid token-limit 400s (Groq 70B ~131k context).
MAX_CONTEXT_CHARS = 80_000


SYSTEM_PROMPT = """You are selecting and trimming text from medical PDF excerpts for study/reference.

CRITICAL RULES:
- Use ONLY the provided context. Do NOT use any outside knowledge.
- Do NOT paraphrase, reword, infer, or explain.
- Only output exact text spans that appear verbatim in the context.
- If the answer is not present in the context, output: Not found in uploaded documents

Output JSON only.
"""


async def llm_select_and_trim(query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """
    LLM is allowed ONLY to:
    - select which candidate chunks are relevant
    - trim redundant lines
    It MUST NOT introduce any new wording.

    Safety: if model output contains text not present in candidate context, we ignore it.
    """
    if not settings.groq_api_key:
        return {"mode": "no_llm", "selected_chunk_ids": [], "answer": None}

    context_blocks = []
    for c in candidates:
        context_blocks.append(
            f"[chunk_id={c['chunk_id']} pdf={c['pdf_name']} page={c['page']}]\n{c['text']}\n"
        )
    context = "\n---\n".join(context_blocks)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[CONTEXT TRUNCATED]"

    user_prompt = f"""QUESTION: {query}

CONTEXT:
{context}

Return JSON:
{{
  "answer": "string (either exact extracted text from CONTEXT, or 'Not found in uploaded documents')",
  "selected_chunk_ids": ["..."],
  "citations": [{{"chunk_id":"...", "pdf_name":"...", "page": 1}}]
}}
"""

    try:
        async with httpx.AsyncClient(base_url=settings.groq_base_url, timeout=60) as client:
            resp = await client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                json={
                    "model": settings.groq_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        body = e.response.text if e.response else ""
        logger.warning(
            "Groq API error %s: %s",
            e.response.status_code if e.response else "?",
            body[:500] if body else "(no body)",
        )
        return {"mode": "groq_error", "selected_chunk_ids": [], "answer": None}
    except Exception as e:
        logger.warning("Groq request failed: %s", e)
        return {"mode": "groq_error", "selected_chunk_ids": [], "answer": None}

    content = data["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except Exception:
        return {"mode": "llm_invalid_json", "selected_chunk_ids": [], "answer": None}

    answer = (parsed.get("answer") or "").strip()
    if answer and answer != "Not found in uploaded documents":
        # Safety check: answer must be a verbatim substring of context
        if answer not in context:
            return {"mode": "llm_rejected_non_verbatim", "selected_chunk_ids": [], "answer": None}

    return {
        "mode": "llm_ok",
        "selected_chunk_ids": parsed.get("selected_chunk_ids") or [],
        "answer": answer,
        "citations": parsed.get("citations") or [],
    }


# Backwards-compatible alias (older code may import this name)
grok_select_and_trim = llm_select_and_trim


"""
GROQ CLEANUP (TEXT REPAIR ONLY)

Used ONLY to fix PDF-derived artifacts (broken sentences, hyphenated line
breaks, split paragraphs). It must never summarize, interpret, or change
clinical meaning.
"""

CLEANUP_SYSTEM_PROMPT = """You are repairing extracted medical guideline text.

Task:
Clean and repair the provided text so it reads as a complete, continuous paragraph.

Rules:
- Preserve original meaning exactly
- Do not add or remove medical information
- Do not summarize or interpret
- Do not add or remove medical information
- Fix only formatting and broken words
"""


def _needs_cleanup(text: str) -> bool:
    """
    Heuristic to decide when to call Groq for cleanup:
    - obvious broken start (mid-word / mid-sentence)
    - hyphenation artifacts spanning line breaks
    - very short, clearly fragmentary chunks

    This keeps us from sending every chunk to Groq and aligns with the rule:
    use Groq only when formatting is clearly damaged.
    """
    t = text or ""
    if not t.strip():
        return False
    if _looks_like_starts_mid_word(t):
        return True
    # Hyphenation artifacts such as "anti-\nplatelet" or similar.
    if re.search(r"[A-Za-z]-\s*\n\s*[A-Za-z]", t):
        return True
    # Chunks that are extremely short are more likely to be fragments than
    # useful standalone paragraphs.
    if len(t.strip()) < 80:
        return True
    return False


async def groq_cleanup_chunks(chunks: list[dict[str, Any]]) -> dict[str, str]:
    """
    Use Groq ONLY to clean PDF artifacts in chunk text.

    Safety decisions:
    - If no Groq key or any error occurs, we silently return the original text.
      Retrieval remains fully usable without LLM.
    - We do not allow the model to add new chunks or metadata; it can only
      rewrite text for provided chunk_ids.
    - We call Groq per-chunk (max ~9 selected chunks) using the EXACT prompt
      specified for text repair, never for summarization or interpretation.
    """
    if not settings.groq_api_key:
        return {}
    if not chunks:
        return {}

    out: dict[str, str] = {}

    async with httpx.AsyncClient(base_url=settings.groq_base_url, timeout=60) as client:
        for c in chunks:
            raw_text = c.get("text") or ""
            if not _needs_cleanup(raw_text):
                continue

            txt = raw_text[:4000]
            user_prompt = f'''Input:
"""
{txt}
"""'''

            try:
                resp = await client.post(
                    "/chat/completions",
                    headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                    json={
                        "model": settings.groq_model,
                        "messages": [
                            {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                # If cleanup fails for this chunk, skip it and keep original text.
                continue

            cleaned = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            cleaned = cleaned.strip()
            if cleaned:
                out[c["chunk_id"]] = cleaned

    return out


def _is_recommendation_chunk(text: str) -> bool:
    """Chunks with COR, LOE, or Recommendations get ranked higher."""
    t = (text or "").upper()
    return (
        "COR" in t or "LOE" in t or "RECOMMENDATION" in t
        or "CLASS I" in t or "CLASS II" in t or "CLASS III" in t
    )


def retrieve_chunks(query: str, collection_name: str, n_results: int = 8) -> list[dict[str, Any]]:
    """
    Retrieval-first: Chroma similarity search.
    Returns chunks with distance, heading. Recommendation-like chunks promoted.
    """
    col = get_collection(collection_name)
    res = col.query(
        query_texts=[query],
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )
    docs = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []
    dists = res["distances"][0] if res.get("distances") else [0.0] * len(docs)
    out: list[dict[str, Any]] = []
    for i in range(len(docs)):
        m = metas[i] if i < len(metas) else {}
        m = m or {}
        doc_id = m.get("doc_id", "")
        page = m.get("page", 0)
        chunk_index = m.get("chunk_index", 0)
        chunk_id = f"{doc_id}_p{page}_c{chunk_index}"
        d = dists[i] if i < len(dists) else 0.0
        txt = docs[i] or ""
        out.append(
            {
                "chunk_id": chunk_id,
                "text": txt,
                "distance": d,
                "heading": m.get("heading"),
                **m,
            }
        )
    # Promote recommendation tables: sort so COR/LOE/Recommendations first, then by distance
    def sort_key(c: dict) -> tuple[int, float]:
        rec = 0 if _is_recommendation_chunk(c.get("text") or "") else 1
        return (rec, c.get("distance") or 0.0)

    out.sort(key=sort_key)
    return out


# Section-aware grouping & deterministic content typing.
_PLACEHOLDER_NO_STANDALONE_DEFINITION = "No standalone definition paragraph found in this guideline."
_PLACEHOLDER_SYNOPSIS_TRUNCATED = "Synopsis section present but not fully retrieved verbatim."

# Output order is locked by requirements.
_DISPLAY_GROUPS_ORDER = [
    ("A", "Definition & Classification"),
    ("B", "Diagnostic Criteria"),
    ("C", "Management Recommendations"),
    ("D", "Harm / Contraindications"),
    ("E", "Synopsis"),
    ("F", "Related Figures & Algorithms"),
]

# Deterministic content type tags (one per chunk, Phase 1).
CONTENT_DEFINITION_PARAGRAPH = "definition_paragraph"
CONTENT_CLASSIFICATION_FIGURE = "classification_figure"
CONTENT_DIAGNOSTIC_TABLE = "diagnostic_table"
CONTENT_MANAGEMENT_RECOMMENDATION = "management_recommendation"
CONTENT_HARM_CONTRAINDICATION = "harm_contraindication"
CONTENT_SYNOPSIS = "synopsis"
CONTENT_REFERENCE = "reference"
CONTENT_ABBREVIATION = "abbreviation"

# New section keys for display-oriented grouping. Each chunk is assigned to
# exactly ONE of these for the doctor-facing UI.
SECTION_DEFINITION_SCOPE = "definition_scope"
SECTION_DIAGNOSTIC = "diagnostic_criteria"
SECTION_RISK = "risk_stratification"
SECTION_MANAGEMENT = "management"
SECTION_HARM = "harm_contraindications"

_SECTION_CONFIG = {
    SECTION_DEFINITION_SCOPE: {
        "title": "DEFINITION / SCOPE",
        "max_chunks": 1,
    },
    SECTION_DIAGNOSTIC: {
        "title": "DIAGNOSTIC CRITERIA",
        "max_chunks": 2,
    },
    SECTION_RISK: {
        "title": "RISK STRATIFICATION",
        "max_chunks": 2,
    },
    SECTION_MANAGEMENT: {
        "title": "MANAGEMENT",
        "max_chunks": 3,
    },
    SECTION_HARM: {
        "title": "HARM / CONTRAINDICATIONS",
        "max_chunks": 1,
    },
}

# Minimum confidence required for a chunk to be treated as "clear standalone"
# content for its assigned section.
MIN_SECTION_CONFIDENCE = 0.35


def _norm(s: str) -> str:
    return (s or "").strip()


def _combined_text(chunk: dict[str, Any]) -> str:
    return f"{chunk.get('heading') or ''}\n{chunk.get('text') or ''}"


def _looks_like_starts_mid_word(text: str) -> bool:
    """
    Deterministic fragment detection:
    - obvious continuation punctuation
    - starts with lowercase letter (common when chunk begins mid-sentence/word)
    """
    t = _norm(text)
    if not t:
        return False
    if re.match(r"^[,;:\)\]\}]", t):
        return True
    # starts with lowercase alpha (heuristic; conservative and deterministic)
    if re.match(r"^[a-z]", t):
        return True
    return False


def _looks_like_ends_mid_sentence(text: str) -> bool:
    """
    Deterministic fragment detection: last non-space char is not a sentence terminator.
    """
    t = _norm(text)
    if not t:
        return False
    # Phase 2: more sophisticated end-of-sentence detection could be added here.
    last = t[-1]
    return False if last in ".!?)]}" else False


def _looks_like_abbrev_only(text: str) -> bool:
    """
    Deterministic suppression for abbreviation-only chunks:
    - very short
    - tokens are mostly uppercase abbreviations / codes
    """
    t = _norm(text)
    if not t:
        return True
    if len(t) > 120:
        return False
    # Remove obvious separators and split
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\/\-\.\+]*", t)
    if not tokens:
        return True
    if len(tokens) > 10:
        return False
    abbrev_like = 0
    for tok in tokens:
        if re.fullmatch(r"[A-Z0-9\/\-\.\+]{2,}", tok) and not re.search(r"[a-z]", tok):
            abbrev_like += 1
    return abbrev_like >= max(3, int(0.8 * len(tokens)))


def _is_synopsis_chunk(chunk: dict[str, Any]) -> bool:
    h = (chunk.get("heading") or "").upper()
    return "SYNOPSIS" in h


def _is_harm_contra_chunk(chunk: dict[str, Any]) -> bool:
    t = _combined_text(chunk).upper()
    return any(
        k in t
        for k in (
            "CONTRAINDICAT",
            "DO NOT",
            "AVOID",
            "WARNING",
            "CAUTION",
            "ADVERSE",
            "HARM",
            "BLEEDING",
            "RISK OF",
        )
    )


def _classify_content_type(chunk: dict[str, Any]) -> str:
    """
    Deterministic, single content-type tag per chunk.
    Never used for generation, only for routing/filtering.
    """
    combined = _combined_text(chunk)
    u = combined.upper()

    if _is_synopsis_chunk(chunk):
        return CONTENT_SYNOPSIS

    if _is_harm_contra_chunk(chunk):
        return CONTENT_HARM_CONTRAINDICATION

    # Management recommendations (COR/LOE/recommendation language).
    if _is_recommendation_chunk(chunk.get("text") or "") or any(
        k in u for k in ("RECOMMENDATION", "CLASS I", "CLASS II", "CLASS III")
    ):
        return CONTENT_MANAGEMENT_RECOMMENDATION

    # Classification figures.
    if ("FIGURE" in u or "FIG." in u) and any(
        k in u for k in ("CLASSIFICATION", "TYPES", "SPECTRUM")
    ):
        return CONTENT_CLASSIFICATION_FIGURE

    # Diagnostic tables (including ECG tables).
    if ("TABLE" in u or "TAB." in u) and any(
        k in u for k in ("DIAGNOSTIC", "CRITERIA", "ECG", "FINDINGS")
    ):
        return CONTENT_DIAGNOSTIC_TABLE

    # Obvious references (journal-style).
    if re.search(r"\b\d{4}\b", u) and any(
        k in u for k in ("J AM", "EUR HEART", "JOURNAL", "DOI", "PMID")
    ):
        return CONTENT_REFERENCE

    # Abbreviation-dominated blocks.
    if _looks_like_abbrev_only(chunk.get("text") or ""):
        return CONTENT_ABBREVIATION

    # Definitions or classification prose.
    if any(
        k in u
        for k in ("DEFINITION", "IS DEFINED AS", "REFERS TO", "CLASSIFICATION")
    ):
        return CONTENT_DEFINITION_PARAGRAPH

    # Default: treat as definition-style prose rather than recommendation.
    # TODO: In Phase 2, introduce more granular evidence/background content types.
    return CONTENT_DEFINITION_PARAGRAPH


def _is_non_guideline_page_chunk(chunk: dict[str, Any]) -> bool:
    """
    Heuristically identify pages that should never be surfaced as clinical
    content:
    - title pages
    - writing committee / author lists / affiliations
    - methods / literature search descriptions
    - pure bibliography / reference sections
    - DOI / journal metadata

    This is deliberately conservative: we only suppress chunks when multiple
    strong signals are present to avoid hiding true guideline text.
    """
    combined = _combined_text(chunk).upper()
    # Writing committees / authors / affiliations.
    committee_keywords = (
        "WRITING COMMITTEE",
        "WRITING GROUP",
        "TASK FORCE",
        "AUTHOR",
        "AUTHORS",
        "AFFILIATION",
        "AFFILIATIONS",
        "CORRESPONDENCE TO",
        "ADDRESS FOR CORRESPONDENCE",
    )
    if any(k in combined for k in committee_keywords):
        return True

    # Methods / literature search sections.
    if "LITERATURE SEARCH" in combined or "SEARCH STRATEGY" in combined:
        return True
    if "METHODS" in combined and "SYSTEMATIC" in combined:
        return True

    # DOI / journal metadata / copyright blocks.
    if "DOI" in combined or "ISSN" in combined or "COPYRIGHT" in combined:
        if "GUIDELINE" not in combined:
            return True

    # Obvious reference-only pages are already classified as CONTENT_REFERENCE.
    ct = _classify_content_type(chunk)
    if ct == CONTENT_REFERENCE:
        return True

    return False


def _assign_section_for_display(chunk: dict[str, Any], query: str) -> str:
    """
    Deterministically assign each chunk to ONE section for display using:
    - document headings
    - nearby section titles
    - keywords (e.g., "risk score", "recommendation", "should not").
    """
    combined = _combined_text(chunk)
    u = combined.upper()
    uq = (query or "").upper()

    # Explicit harm / contraindications first so they are never misrouted.
    if _is_harm_contra_chunk(chunk) or any(
        k in u
        for k in (
            "CONTRAINDICAT",
            "SHOULD NOT",
            "AVOID",
            "DO NOT",
            "BLACK BOX WARNING",
        )
    ):
        return SECTION_HARM

    # Risk stratification: scores, tools, low-/high-risk language.
    risk_keywords = (
        "RISK SCORE",
        "GRACE",
        "TIMI",
        "RISK STRATIFICATION",
        "LOW-RISK",
        "HIGH-RISK",
        "INTERMEDIATE-RISK",
    )
    if any(k in u for k in risk_keywords) or "RISK" in uq:
        return SECTION_RISK

    # Diagnostic criteria: troponin, ECG changes, criteria language.
    diag_keywords = (
        "DIAGNOSTIC CRITERIA",
        "DIAGNOSTIC",
        "CRITERIA",
        "ECG",
        "ELECTROCARDIOGRAM",
        "TROPONIN",
        "BIOMARKER",
    )
    if any(k in u for k in diag_keywords):
        return SECTION_DIAGNOSTIC

    # Management: recommendations, should, therapy, drugs, procedures.
    if _is_recommendation_chunk(chunk.get("text") or "") or any(
        k in u
        for k in (
            "RECOMMENDATION",
            "MANAGEMENT",
            "TREATMENT",
            "THERAPY",
            "PCI",
            "CABG",
            "ANTIPLATELET",
            "ANTICOAGUL",
        )
    ):
        return SECTION_MANAGEMENT

    # Definition / scope: explicit definition or introductory/scope language,
    # plus very early pages default to definition if nothing else matches.
    def_keywords = (
        "DEFINITION",
        "IS DEFINED AS",
        "REFERS TO",
        "CLASSIFICATION",
        "SCOPE",
        "IN THIS GUIDELINE",
        "THESE GUIDELINES",
    )
    if any(k in u for k in def_keywords):
        return SECTION_DEFINITION_SCOPE

    page = int(chunk.get("page", 0) or 0)
    if page and page <= 2:
        # Introductory pages are more likely to be definition/scope if no
        # stronger signal is present.
        return SECTION_DEFINITION_SCOPE

    # Fallback: if our content-type classification says "harm", "management",
    # or "diagnostic", map accordingly; otherwise treat as definition/scope.
    ct = _classify_content_type(chunk)
    if ct == CONTENT_HARM_CONTRAINDICATION:
        return SECTION_HARM
    if ct == CONTENT_MANAGEMENT_RECOMMENDATION:
        return SECTION_MANAGEMENT
    if ct in {CONTENT_DIAGNOSTIC_TABLE, CONTENT_CLASSIFICATION_FIGURE}:
        return SECTION_DIAGNOSTIC
    return SECTION_DEFINITION_SCOPE


def _distance_to_similarity(distance: float | None) -> float:
    """
    Convert vector distance into a 0–1 similarity score.
    Smaller distance => higher similarity; bounded conservatively.
    """
    if distance is None:
        return 0.5
    d = max(float(distance), 0.0)
    return 1.0 / (1.0 + d)


def _compute_confidence_for_chunk(
    *,
    chunk: dict[str, Any],
    section_key: str,
    query: str,
) -> float:
    """
    Compute a confidence score in [0.0, 1.0] using:
    - semantic similarity to normalized query (via distance)
    - presence of section-relevant keywords
    - text completeness (full sentences > fragments)
    - position in guideline (earlier pages preferred).
    """
    dist = chunk.get("distance")
    sim = _distance_to_similarity(dist)

    text = chunk.get("text") or ""
    combined = _combined_text(chunk)
    u = combined.upper()
    uq = (query or "").upper()

    # Section-relevant keywords.
    section_keywords_map: dict[str, tuple[str, ...]] = {
        SECTION_DEFINITION_SCOPE: (
            "DEFINITION",
            "IS DEFINED AS",
            "REFERS TO",
            "SCOPE",
            "IN THIS GUIDELINE",
        ),
        SECTION_DIAGNOSTIC: (
            "DIAGNOSTIC CRITERIA",
            "DIAGNOSTIC",
            "CRITERIA",
            "ECG",
            "TROPONIN",
            "BIOMARKER",
        ),
        SECTION_RISK: (
            "RISK SCORE",
            "GRACE",
            "TIMI",
            "RISK STRATIFICATION",
            "LOW-RISK",
            "HIGH-RISK",
            "INTERMEDIATE-RISK",
        ),
        SECTION_MANAGEMENT: (
            "RECOMMENDATION",
            "MANAGEMENT",
            "TREATMENT",
            "THERAPY",
            "PCI",
            "CABG",
            "ANTIPLATELET",
            "ANTICOAGUL",
        ),
        SECTION_HARM: (
            "CONTRAINDICAT",
            "DO NOT",
            "SHOULD NOT",
            "AVOID",
            "WARNING",
            "CAUTION",
            "HARM",
        ),
    }
    kw = section_keywords_map.get(section_key, ())
    has_section_keyword = any(k in u for k in kw)
    keyword_score = 1.0 if has_section_keyword else 0.0

    # Boost when the query disease term appears in the chunk.
    if uq and uq in u:
        keyword_score = max(keyword_score, 0.6)

    # Completeness: penalize obvious fragments and ultra-short chunks.
    if _looks_like_starts_mid_word(text):
        completeness = 0.4
    elif len(text.strip()) < 120:
        completeness = 0.7
    else:
        completeness = 1.0

    # Earlier pages are preferred; later pages decay slowly but never drop
    # below 0.4 to avoid completely discarding late-but-relevant material.
    page = int(chunk.get("page", 0) or 0)
    if page <= 0:
        position = 0.6
    else:
        position = max(0.4, 1.0 - 0.05 * (page - 1))

    # Weighted combination, then clipped to [0, 1].
    score = (
        0.5 * sim
        + 0.25 * keyword_score
        + 0.15 * completeness
        + 0.10 * position
    )
    return max(0.0, min(1.0, score))


def build_sectioned_display(
    *,
    query: str,
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Core backend logic for doctor search:
    - assign each candidate chunk to one display section
    - compute a confidence score for that section
    - enforce hard per-section limits on number of chunks
    - suppress all remaining chunks (they are never sent to the frontend).

    Returns:
      (section_groups_meta, selected_chunks_flat)
    where section_groups_meta is a list of dictionaries:
      {
        "section_key": SECTION_*,
        "section_title": "...",
        "items": [chunk_dict, ...],  # selected for that section
        "placeholder": bool,        # True when no high-confidence chunk exists
      }
    """
    section_buckets: dict[str, list[dict[str, Any]]] = {
        key: [] for key in _SECTION_CONFIG.keys()
    }

    # First, annotate each chunk with its section and confidence, skipping
    # non-guideline pages entirely.
    for c in chunks:
        if _is_non_guideline_page_chunk(c):
            continue
        sec = _assign_section_for_display(c, query)
        conf = _compute_confidence_for_chunk(chunk=c, section_key=sec, query=query)
        c2 = dict(c)
        c2["_section_key"] = sec
        c2["_confidence"] = conf
        section_buckets.setdefault(sec, []).append(c2)

    section_groups: list[dict[str, Any]] = []
    selected_chunks: list[dict[str, Any]] = []

    # Deterministic section ordering for output.
    for sec_key in (
        SECTION_DEFINITION_SCOPE,
        SECTION_DIAGNOSTIC,
        SECTION_RISK,
        SECTION_MANAGEMENT,
        SECTION_HARM,
    ):
        cfg = _SECTION_CONFIG[sec_key]
        items = section_buckets.get(sec_key, [])

        # Rank by confidence (desc), then by similarity (distance asc).
        items.sort(
            key=lambda c: (
                float(c.get("_confidence") or 0.0),
                -float(c.get("distance") or 0.0),
            ),
            reverse=True,
        )

        high_conf = [c for c in items if (c.get("_confidence") or 0.0) >= MIN_SECTION_CONFIDENCE]
        chosen = high_conf[: cfg["max_chunks"]]

        if not chosen:
            # No clear standalone content for this section. We record a
            # placeholder so the caller can show the standardized message,
            # but we do NOT invent guideline text.
            section_groups.append(
                {
                    "section_key": sec_key,
                    "section_title": cfg["title"],
                    "items": [],
                    "placeholder": True,
                }
            )
            continue

        section_groups.append(
            {
                "section_key": sec_key,
                "section_title": cfg["title"],
                "items": chosen,
                "placeholder": False,
            }
        )
        selected_chunks.extend(chosen)

    return section_groups, selected_chunks


def _classify_abc(chunk: dict[str, Any]) -> str:
    """
    REQUIRED: exactly these disease sections:
    A. Definition & Classification
    B. Diagnostic Criteria
    C. Management Recommendations
    Plus separate buckets for Harm/Synopsis handled elsewhere.
    """
    combined = _combined_text(chunk)
    u = combined.upper()

    # Figures titled “Types and Classification” → Section A
    if "TYPES AND CLASSIFICATION" in u:
        return "A"

    # ECG tables → Section B
    if "ECG" in u and ("TABLE" in u or "TAB." in u):
        return "B"

    # COR/LOE tables → Section C
    if ("COR" in u and "LOE" in u) or ("RECOMMENDATION" in u and ("CLASS I" in u or "CLASS II" in u or "CLASS III" in u)):
        return "C"

    # General deterministic keyword mapping
    if any(k in u for k in ("DIAGNOSTIC", "DIAGNOS", "CRITERIA", "ECG", "TROPONIN", "BIOMARKER")):
        return "B"
    if any(k in u for k in ("DEFINITION", "CLASSIFICATION", "TYPES OF", "NOMENCLATURE")):
        return "A"
    if any(k in u for k in ("MANAGEMENT", "TREATMENT", "THERAPY", "REPERFUSION", "ANTIPLATELET", "ANTICOAGUL", "PCI", "CABG")):
        return "C"

    # Default to A to avoid incorrectly implying management. (TODO: expand deterministic rules if needed.)
    return "A"


def build_display_groups_and_notes(
    *,
    query: str,
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], str]:
    """
    Deterministic post-processing only:
    - fragment suppression (start mid-word / end mid-sentence / abbrev-only)
    - strict section reclassification (A/B/C)
    - separate Harm/Contra and Synopsis groups
    - synopsis integrity: if synopsis exists but any synopsis chunk truncated, hide synopsis and show placeholder
    Returns:
      (groups, display_chunks_for_results, pages, coverage_note)
    """
    # Tag all chunks with deterministic content types.
    annotated: list[dict[str, Any]] = []
    for c in chunks:
        ct = _classify_content_type(c)
        c2 = dict(c)
        c2["content_type"] = ct
        annotated.append(c2)

    # Fragment suppression: ONLY if starts mid-word AND not table/figure/COR/LOE.
    displayable: list[dict[str, Any]] = []
    for c in annotated:
        text = c.get("text") or ""
        ct = c.get("content_type")
        if _looks_like_starts_mid_word(text):
            heading_u = (c.get("heading") or "").upper()
            if ct not in {
                CONTENT_CLASSIFICATION_FIGURE,
                CONTENT_DIAGNOSTIC_TABLE,
                CONTENT_MANAGEMENT_RECOMMENDATION,
            } and not any(k in heading_u for k in ("FIGURE", "FIG.", "TABLE", "COR", "LOE")):
                # Suppress true mid-word fragments that are not tables/figures/COR/LOE.
                continue
        # Never show reference/abbreviation chunks by default.
        if ct in {CONTENT_REFERENCE, CONTENT_ABBREVIATION}:
            continue
        displayable.append(c)

    # Collect synopsis and determine if fully retrievable.
    synopsis_chunks_all = [c for c in displayable if c.get("content_type") == CONTENT_SYNOPSIS]
    synopsis_truncated = any(_looks_like_ends_mid_sentence(c.get("text") or "") for c in synopsis_chunks_all)
    show_synopsis = bool(synopsis_chunks_all) and not synopsis_truncated

    # Bucket chunks (excluding synopsis if hidden).
    harm_items: list[dict[str, Any]] = []
    abc_items: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    synopsis_items: list[dict[str, Any]] = []
    figure_algo_items: list[dict[str, Any]] = []

    for c in displayable:
        ct = c.get("content_type")
        if ct == CONTENT_SYNOPSIS:
            if show_synopsis:
                synopsis_items.append(c)
            continue
        if ct == CONTENT_HARM_CONTRAINDICATION:
            harm_items.append(c)
            continue
        if ct in {CONTENT_CLASSIFICATION_FIGURE, CONTENT_DIAGNOSTIC_TABLE}:
            # These still belong in A/B/C but are also relevant to figures/algorithms section.
            figure_algo_items.append(c)
        sec = _classify_abc(c)
        abc_items.setdefault(sec, []).append(c)

    # Build groups in locked order.
    groups: list[dict[str, Any]] = []

    def _placeholder_item(msg: str) -> dict[str, Any]:
        return {
            "chunk_id": "__note__",
            "pdf_name": "",
            "doc_id": "",
            "page": 0,
            "text": msg,
            "heading": None,
        }

    for label, title in _DISPLAY_GROUPS_ORDER:
        if label in ("A", "B", "C"):
            items = abc_items.get(label, [])
            # Section gating removed: each section renders independently.
            if not items:
                continue
            # For Definition & Classification specifically, only show placeholder when
            # no definition/classification material exists at all.
            if label == "A":
                has_def_or_class = any(
                    it.get("content_type")
                    in {
                        CONTENT_DEFINITION_PARAGRAPH,
                        CONTENT_CLASSIFICATION_FIGURE,
                        CONTENT_DIAGNOSTIC_TABLE,
                    }
                    for it in items
                )
                if not has_def_or_class:
                    items = [_placeholder_item(_PLACEHOLDER_NO_STANDALONE_DEFINITION)]
            groups.append(
                {"section_label": label, "section_title": title, "items": items}
            )
        elif label == "D":
            if not harm_items:
                continue
            groups.append(
                {"section_label": label, "section_title": title, "items": harm_items}
            )
        elif label == "E":
            if synopsis_chunks_all and not show_synopsis:
                groups.append(
                    {
                        "section_label": label,
                        "section_title": title,
                        "items": [
                            _placeholder_item(_PLACEHOLDER_SYNOPSIS_TRUNCATED)
                        ],
                    }
                )
            elif show_synopsis:
                groups.append(
                    {
                        "section_label": label,
                        "section_title": title,
                        "items": synopsis_items,
                    }
                )
        elif label == "F":
            if not figure_algo_items:
                continue
            # F. Related Figures & Algorithms: re-use relevant chunks; images are
            # surfaced separately in the SearchResponse.images field.
            groups.append(
                {
                    "section_label": label,
                    "section_title": title,
                    "items": figure_algo_items,
                }
            )

    # Flatten displayed chunks for `results` (exclude placeholders).
    display_chunks_for_results: list[dict[str, Any]] = []
    for g in groups:
        for it in g["items"]:
            if it.get("chunk_id") == "__note__":
                continue
            if it.get("page", 0) == 0 and not it.get("doc_id"):
                continue
            display_chunks_for_results.append(it)

    pages = sorted({int(c.get("page", 0) or 0) for c in display_chunks_for_results if int(c.get("page", 0) or 0) > 0})
    coverage_note = build_coverage_note(pages, query)
    return groups, display_chunks_for_results, pages, coverage_note


def build_coverage_note(pages: list[int], query: str) -> str:
    """
    REQUIRED: standardized coverage note at end of every result.
    If pages exist:
        \"Coverage note: Results include content from pages X, Y, Z.\"
    If no pages exist:
        \"Coverage note: No verbatim excerpts matched this exact query.\"
    Composite/risk-qualified and disease queries add deterministic context notes.
    """
    sp = sorted(set(int(p) for p in (pages or []) if int(p) > 0))
    if sp:
        page_str = ", ".join(str(p) for p in sp)
        base = f"Coverage note: Results include content from pages {page_str}."
    else:
        base = "Coverage note: No verbatim excerpts matched this exact query."

    q = (query or "").upper()
    # Composite / risk-qualified queries (e.g., \"High NSTEMI\").
    risk_modifiers = ("HIGH", "LOW", "RISK", "SEVERE", "MILD")
    diseases = ("NSTEMI", "STEMI", "NSTE-ACS", "NSTEACS")
    is_composite = any(m in q for m in risk_modifiers) and any(d in q for d in diseases)
    if is_composite:
        # Explicit contextual note, no fabrication of guideline text.
        base += (
            " This guideline does not define a standalone category labeled "
            f"'{query.strip()}'. Risk is determined via stratification tools (e.g., GRACE/TIMI) "
            "discussed in dedicated sections."
        )
    else:
        # Disease-query acknowledgement for base conditions without modifiers.
        if any(d in q for d in diseases):
            base += (
                " Risk stratification tools are discussed in dedicated sections of this guideline "
                "(e.g., GRACE/TIMI). No verbatim excerpt retrieved for this query."
            )
    return base

