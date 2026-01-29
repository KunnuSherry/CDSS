"""
Clinical Decision Support — structured schemas for NSTEMI/ACS patient input and output.

SAFETY: No free-text clinical input; no AI-generated recommendations.
All fields are structured to avoid hallucination and to keep the system advisory only.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# PART 1 — CLINICAL INPUT MODULE (NSTEMI / ACS)
# ---------------------------------------------------------------------------
# Disease-context: NSTEMI. Required fields only; no free-text input.
# Reject missing required fields via Pydantic (no default values for required).
# ---------------------------------------------------------------------------


class PatientClinicalInput(BaseModel):
    """
    Structured patient input for ACS/NSTEMI decision support.
    All fields are required. No free-text; structured only.
    Clinical safety: Reject missing required fields to avoid partial/invalid scoring.
    """

    # Age in years (numeric).
    age: int = Field(..., ge=0, le=120, description="Patient age in years")

    # Heart rate at presentation (bpm).
    heart_rate: int = Field(..., ge=0, le=300, description="Heart rate in bpm")

    # Systolic blood pressure at presentation (mmHg).
    systolic_bp: int = Field(..., ge=0, le=300, description="Systolic BP in mmHg")

    # Serum creatinine (mg/dL).
    serum_creatinine: float = Field(..., ge=0.0, description="Serum creatinine in mg/dL")

    # Killip class I–IV (integer 1–4).
    killip_class: Literal[1, 2, 3, 4] = Field(
        ...,
        description="Killip class I (no CHF) to IV (cardiogenic shock)",
    )

    # Cardiac arrest at admission (boolean).
    cardiac_arrest_at_admission: bool = Field(
        ...,
        description="Cardiac arrest at admission (yes/no)",
    )

    # ST-segment deviation on ECG (boolean).
    st_deviation_ecg: bool = Field(
        ...,
        description="ST-segment deviation on ECG (yes/no)",
    )

    # Elevated cardiac enzymes (boolean).
    elevated_cardiac_enzymes: bool = Field(
        ...,
        description="Elevated cardiac enzymes (e.g. troponin) (yes/no)",
    )

    @field_validator("age", "heart_rate", "systolic_bp", mode="before")
    @classmethod
    def coerce_int(cls, v):  # noqa: ANN001
        """Reject missing; allow int from JSON number."""
        if v is None:
            raise ValueError("Field is required")
        return int(v) if not isinstance(v, int) else v

    @field_validator("serum_creatinine", mode="before")
    @classmethod
    def coerce_float(cls, v):  # noqa: ANN001
        """Reject missing; allow float from JSON number."""
        if v is None:
            raise ValueError("Field is required")
        return float(v) if not isinstance(v, (int, float)) else v

    @field_validator("killip_class", mode="before")
    @classmethod
    def coerce_killip(cls, v):  # noqa: ANN001
        """Reject missing; allow 1–4 only."""
        if v is None:
            raise ValueError("Killip class is required")
        k = int(v) if not isinstance(v, int) else v
        if k not in (1, 2, 3, 4):
            raise ValueError("Killip class must be 1, 2, 3, or 4")
        return k


# ---------------------------------------------------------------------------
# PART 2 — GRACE SCORE OUTPUT (from deterministic engine)
# ---------------------------------------------------------------------------

# NSTEMI in-hospital mortality risk categories (official GRACE tertiles).
# Low ≤108, Intermediate 109–140, High >140. Do not use AI; use these cutoffs only.
GRACE_RISK_LOW_MAX = 108
GRACE_RISK_INTERMEDIATE_MIN = 109
GRACE_RISK_INTERMEDIATE_MAX = 140
# High: >140


class GraceScoreResult(BaseModel):
    """Result of deterministic GRACE score calculation. No AI."""

    total_score: int = Field(..., description="Sum of all GRACE points (1–372)")
    risk_category: Literal["low", "intermediate", "high"] = Field(
        ...,
        description="NSTEMI in-hospital risk category",
    )
    # Verbatim interpretation text for display; from official tertiles only.
    category_description: str = Field(
        ...,
        description="Short description of risk category (e.g. mortality range)",
    )


# ---------------------------------------------------------------------------
# GUIDELINE EXCERPTS (from existing RAG pipeline; verbatim after Grok cleanup)
# ---------------------------------------------------------------------------

class GuidelineExcerpt(BaseModel):
    """Single guideline chunk from RAG; verbatim text, no AI summarization for content."""

    chunk_id: str
    pdf_name: str
    doc_id: str
    page: int
    text: str
    heading: str | None = None
    section_label: str | None = None
    section_title: str | None = None


# ---------------------------------------------------------------------------
# PART 4 — CASE STUDY (extractive, cited, non–AI generated)
# ---------------------------------------------------------------------------

class CaseStudyRef(BaseModel):
    """Reference to a cited case report; extractive only (e.g. from PubMed)."""

    title: str = Field(..., description="Case report or study title")
    source: str = Field(..., description="Source (e.g. journal, PubMed)")
    pmid: str | None = Field(None, description="PubMed ID if available")
    url: str | None = Field(None, description="Link to article (e.g. PubMed URL)")
    risk_level: str | None = Field(None, description="Matched risk level (low/intermediate/high)")
    intervention_type: str | None = Field(None, description="Intervention type if available")


# ---------------------------------------------------------------------------
# PART 5 — FINAL CLINICAL OUTPUT (advisory only; no treatment recommendations)
# ---------------------------------------------------------------------------

class PatientSummary(BaseModel):
    """Structured patient summary for display; no free-text generation."""

    age: int
    heart_rate: int
    systolic_bp: int
    serum_creatinine: float
    killip_class: int
    cardiac_arrest_at_admission: bool
    st_deviation_ecg: bool
    elevated_cardiac_enzymes: bool
    disease_context: str = "NSTEMI"


class ClinicalOutput(BaseModel):
    """
    Final CDSS output. Advisory only.
    Does NOT generate treatment recommendations; does NOT replace clinician judgment.
    Order: 1. Patient summary, 2. GRACE + risk, 3. Guideline excerpts (≤5), 4. Case studies (≤4), 5. Disclaimer.
    """

    patient_summary: PatientSummary
    grace_score: GraceScoreResult
    guideline_excerpts: list[GuidelineExcerpt] = Field(default_factory=list)
    # Additional guideline content hidden behind "Show supporting evidence (advanced)". Max 5 visible primary.
    supporting_evidence: list[GuidelineExcerpt] = Field(default_factory=list)
    case_studies: list[CaseStudyRef] = Field(default_factory=list)
    # Exact disclaimer text (non-negotiable).
    disclaimer: str = Field(
        default=(
            "Scores quantify risk; guidelines contextualize risk; cases show precedent. "
            "Final decisions rest with the treating clinician."
        ),
        description="Advisory disclaimer",
    )
