"""
Guideline Mapping — rule-based mapping from disease + GRACE score to RAG queries.

Disease = NSTEMI (fixed). Uses rules only; no LLMs.
Risk-conditioned: GRACE >140 vs ≤140 drives different prioritization (early invasive vs selective/delayed).
"""

from __future__ import annotations

# Disease context for this CDSS module: NSTEMI only.
DISEASE_CONTEXT = "NSTEMI"

# NSTEMI in-hospital risk cutoffs (do not mix with 6-month).
GRACE_HIGH_THRESHOLD = 140

RISK_LOW = "low"
RISK_INTERMEDIATE = "intermediate"
RISK_HIGH = "high"


def get_guideline_queries_for_grace(
    grace_risk_category: str,
    total_grace_score: int | None = None,
) -> list[str]:
    """
    Map NSTEMI + GRACE (category and raw score) to retrieval queries. Rule-based only.

    IF disease = NSTEMI AND GRACE >140:
      - Prioritize early invasive strategy sections
      - Prioritize cardiogenic shock considerations
      - Prioritize contraindications to routine invasive approach

    IF disease = NSTEMI AND GRACE ≤140:
      - Prioritize selective or delayed invasive strategy sections

    total_grace_score used for >140 vs ≤140 split when available; else risk_category.
    """
    cat = (grace_risk_category or "").strip().lower()
    if cat not in (RISK_LOW, RISK_INTERMEDIATE, RISK_HIGH):
        cat = RISK_INTERMEDIATE

    # Rule-based: use raw score for >140 vs ≤140 when available (more precise than category).
    is_high_grace = (
        total_grace_score is not None and total_grace_score > GRACE_HIGH_THRESHOLD
    ) or cat == RISK_HIGH

    if is_high_grace:
        # NSTEMI + GRACE >140: early invasive, cardiogenic shock, contraindications.
        return [
            "NSTEMI early invasive strategy coronary angiography",
            "NSTEMI cardiogenic shock high risk",
            "NSTEMI contraindications invasive",
            "NSTEMI high risk management",
        ]
    # NSTEMI + GRACE ≤140: selective or delayed invasive.
    return [
        "NSTEMI selective invasive delayed strategy",
        "NSTEMI risk stratification management",
        "NSTEMI conservative strategy",
    ]


def get_priority_sections_for_grace(grace_risk_category: str) -> list[str]:
    """Section keys to prioritize when displaying (risk_stratification, management, etc.)."""
    cat = (grace_risk_category or "").strip().lower()
    if cat == RISK_HIGH:
        return ["management", "risk_stratification", "harm_contraindications"]
    if cat == RISK_LOW:
        return ["risk_stratification", "management", "definition_scope"]
    return ["risk_stratification", "management", "harm_contraindications"]
