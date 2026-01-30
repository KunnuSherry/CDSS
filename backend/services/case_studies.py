"""
Case Study Retrieval — PubMed E-utilities (live from internet).

Match on: Disease (NSTEMI), Risk level. Case studies are extractive and cited; no AI-generated content.
Uses NCBI E-utilities: esearch (PMID list) + esummary (title, journal). No API key required (3 req/sec limit).
"""

from __future__ import annotations

import logging
import httpx

from models.clinical import CaseStudyRef

logger = logging.getLogger(__name__)

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov/"

# Rate limit: 3 req/sec without API key. We do 1 esearch + 1 esummary per call.
REQUEST_TIMEOUT = 15


def _build_pubmed_query(risk_level: str, grace_total: int | None = None) -> str:
    """
    Build PubMed search query for NSTEMI case reports.
    Disease = NSTEMI; risk level must match. Prefer elderly + high-risk when GRACE >140.
    """
    base = "NSTEMI acute coronary syndrome"
    if risk_level == "high":
        # Prefer elderly + high-risk NSTEMI when GRACE >140 (trials and case reports).
        if grace_total is not None and grace_total > 140:
            return f'({base}) AND (case report[pt] OR case reports[ptyp] OR clinical trial[pt]) AND (high risk OR GRACE OR elderly OR octogenarian)'
        return f'({base}) AND (case report[pt] OR case reports[ptyp]) AND (high risk OR GRACE)'
    if risk_level == "low":
        return f'({base}) AND (case report[pt] OR case reports[ptyp]) AND (low risk OR conservative)'
    return f'({base}) AND (case report[pt] OR case reports[ptyp])'


def _esearch_pmids(query: str, max_results: int = 10) -> list[str]:
    """Search PubMed; return list of PMIDs. No AI; extractive only."""
    try:
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance",
        }
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.get(PUBMED_ESEARCH, params=params)
            resp.raise_for_status()
        data = resp.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        return id_list[:max_results]
    except Exception as e:
        logger.warning("PubMed esearch failed: %s", e)
        return []


def _esummary_details(pmids: list[str]) -> list[dict]:
    """Fetch title/source for PMIDs via esummary. Returns list of {pmid, title, source}."""
    if not pmids:
        return []
    try:
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.get(PUBMED_ESUMMARY, params=params)
            resp.raise_for_status()
        data = resp.json()
        result = data.get("result", {})
        uids = result.get("uids", [])
        out = []
        for uid in uids:
            if uid in ("uids",):
                continue
            item = result.get(uid, {})
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            source = (item.get("fulljournalname") or item.get("source") or "PubMed").strip()
            out.append({"pmid": uid, "title": title or "(No title)", "source": source or "PubMed"})
        return out
    except Exception as e:
        logger.warning("PubMed esummary failed: %s", e)
        return []


def get_case_studies_for_profile(
    risk_level: str,
    intervention_type: str | None = None,
    max_results: int = 3,
    grace_total: int | None = None,
) -> list[CaseStudyRef]:
    """
    Retrieve case studies ONLY if disease (NSTEMI) and risk category match.
    Max 3–4 case studies. Prefer elderly + high-risk NSTEMI when GRACE >140.
    Extractive, cited, non-AI generated (PubMed).
    """
    risk_level = (risk_level or "").strip().lower()
    if risk_level not in ("low", "intermediate", "high"):
        risk_level = "intermediate"

    query = _build_pubmed_query(risk_level, grace_total=grace_total)
    pmids = _esearch_pmids(query, max_results=max_results)
    if not pmids:
        # Fallback: broader NSTEMI case report query
        query = "(NSTEMI) AND (case report[pt] OR case reports[ptyp])"
        pmids = _esearch_pmids(query, max_results=max_results)

    details = _esummary_details(pmids)
    out: list[CaseStudyRef] = []
    for d in details[:max_results]:
        pmid = d.get("pmid", "")
        out.append(
            CaseStudyRef(
                title=d.get("title", ""),
                source=d.get("source", "PubMed"),
                pmid=pmid or None,
                url=f"{PUBMED_ARTICLE_URL}{pmid}/" if pmid else None,
                risk_level=risk_level,
                intervention_type=intervention_type,
            )
        )
    return out
