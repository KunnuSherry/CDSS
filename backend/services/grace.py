"""
GRACE Risk Score Calculation Engine — deterministic, no AI.

Model: GRACE 1.0 IN-HOSPITAL MORTALITY for NSTEMI only.
We do NOT use 6-month post-discharge model; we do NOT mix models.
Source: outcomes-umassmed.org/grace/grace_risk_table.aspx; fpnotebook.com (bin-based).
MDCalc uses the same underlying point tables; we use bin-based scoring only (no linear formulas).
"""

from __future__ import annotations

from models.clinical import (
    GRACE_RISK_INTERMEDIATE_MAX,
    GRACE_RISK_INTERMEDIATE_MIN,
    GRACE_RISK_LOW_MAX,
    GraceScoreResult,
    PatientClinicalInput,
)


# ---------------------------------------------------------------------------
# OFFICIAL GRACE POINT LOOKUP TABLES (bin-based; no linear formulas, no AI)
# Source: outcomes-umassmed.org/grace; fpnotebook.com/CV/Exam/GrcScr.htm
# Safety: Use exact brackets only; do not approximate.
# ---------------------------------------------------------------------------

# Age (years): points by bracket. <30 → 0; 30–39 → 8; …; ≥90 → 100.
_GRACE_AGE: list[tuple[tuple[int, int], int]] = [
    ((0, 29), 0),
    ((30, 39), 8),
    ((40, 49), 25),
    ((50, 59), 41),
    ((60, 69), 58),
    ((70, 79), 75),
    ((80, 89), 91),
    ((90, 999), 100),
]

# Heart rate (bpm): points by bracket
_GRACE_HR: list[tuple[tuple[int, int], int]] = [
    ((0, 49), 0),
    ((50, 69), 3),
    ((70, 89), 9),
    ((90, 109), 15),
    ((110, 149), 24),
    ((150, 199), 38),
    ((200, 999), 46),
]

# Systolic BP (mmHg): higher SBP = lower points
_GRACE_SBP: list[tuple[tuple[int, int], int]] = [
    ((0, 79), 58),
    ((80, 99), 53),
    ((100, 119), 43),
    ((120, 139), 34),
    ((140, 159), 24),
    ((160, 199), 10),
    ((200, 999), 0),
]

# Serum creatinine (mg/dL): points by bracket
_GRACE_CREATININE: list[tuple[tuple[float, float], int]] = [
    ((0.0, 0.39), 1),
    ((0.40, 0.79), 4),
    ((0.80, 1.19), 7),
    ((1.20, 1.59), 10),
    ((1.60, 1.99), 13),
    ((2.00, 3.99), 21),
    ((4.00, 999.0), 28),
]

# Killip class I–IV: 0, 20, 39, 59
_GRACE_KILLIP: dict[int, int] = {
    1: 0,
    2: 20,
    3: 39,
    4: 59,
}

_GRACE_CARDIAC_ARREST_POINTS = 39
_GRACE_ST_DEVIATION_POINTS = 28
_GRACE_ELEVATED_ENZYMES_POINTS = 14

# Max allowed deviation from expected (5 test patients). If > this, block downstream.
_GRACE_VALIDATION_MAX_DEVIATION = 3


class GraceValidationError(Exception):
    """Raised when GRACE engine output deviates from expected by more than allowed (e.g. >3 points). Blocks downstream logic."""


def _lookup_bracket(
    value: int | float,
    brackets: list[tuple[tuple[int | float, int | float], int]],
) -> int:
    """Deterministic lookup: bracket (min_inclusive, max_inclusive) → points."""
    for (lo, hi), points in brackets:
        if lo <= value <= hi:
            return points
    if value < brackets[0][0][0]:
        return brackets[0][1]
    return brackets[-1][1]


# ---------------------------------------------------------------------------
# Validation: 5 test patients with expected totals (from same bin tables).
# If computed score deviates from expected by > _GRACE_VALIDATION_MAX_DEVIATION, raise.
# Safety: Ensures we do not ship a broken GRACE implementation.
# ---------------------------------------------------------------------------
_GRACE_TEST_PATIENTS: list[tuple[dict, int]] = [
    # (input dict for PatientClinicalInput, expected_total)
    ({"age": 45, "heart_rate": 80, "systolic_bp": 130, "serum_creatinine": 1.0, "killip_class": 1, "cardiac_arrest_at_admission": False, "st_deviation_ecg": False, "elevated_cardiac_enzymes": True}, 25 + 9 + 34 + 7 + 0 + 0 + 0 + 14),  # 89 low
    ({"age": 72, "heart_rate": 95, "systolic_bp": 105, "serum_creatinine": 1.4, "killip_class": 2, "cardiac_arrest_at_admission": False, "st_deviation_ecg": True, "elevated_cardiac_enzymes": True}, 75 + 15 + 43 + 10 + 20 + 0 + 28 + 14),  # 205 high
    ({"age": 58, "heart_rate": 65, "systolic_bp": 145, "serum_creatinine": 0.9, "killip_class": 1, "cardiac_arrest_at_admission": False, "st_deviation_ecg": False, "elevated_cardiac_enzymes": False}, 41 + 3 + 24 + 7 + 0 + 0 + 0 + 0),  # 75 low
    ({"age": 82, "heart_rate": 115, "systolic_bp": 85, "serum_creatinine": 2.2, "killip_class": 3, "cardiac_arrest_at_admission": True, "st_deviation_ecg": True, "elevated_cardiac_enzymes": True}, 91 + 24 + 53 + 21 + 39 + 39 + 28 + 14),  # 309 high
    ({"age": 62, "heart_rate": 88, "systolic_bp": 118, "serum_creatinine": 1.25, "killip_class": 1, "cardiac_arrest_at_admission": False, "st_deviation_ecg": True, "elevated_cardiac_enzymes": True}, 58 + 9 + 43 + 10 + 0 + 0 + 28 + 14),  # 162 high
]


def _validate_grace_engine() -> None:
    """
    Run 5 test patients; if any deviation from expected > _GRACE_VALIDATION_MAX_DEVIATION, raise.
    Safety: Block downstream logic when GRACE implementation is inconsistent.
    """
    for inp, expected_total in _GRACE_TEST_PATIENTS:
        p = PatientClinicalInput(**inp)
        result = _compute_grace_total_only(p)
        dev = abs(result - expected_total)
        if dev > _GRACE_VALIDATION_MAX_DEVIATION:
            raise GraceValidationError(
                f"GRACE validation failed: expected {expected_total}, got {result} (deviation {dev} > {_GRACE_VALIDATION_MAX_DEVIATION})"
            )


def _compute_grace_total_only(patient: PatientClinicalInput) -> int:
    """Compute total GRACE points only (no risk category). Used for validation and scoring."""
    age_pts = _lookup_bracket(patient.age, _GRACE_AGE)
    hr_pts = _lookup_bracket(patient.heart_rate, _GRACE_HR)
    sbp_pts = _lookup_bracket(patient.systolic_bp, _GRACE_SBP)
    cr_pts = _lookup_bracket(patient.serum_creatinine, _GRACE_CREATININE)
    killip_pts = _GRACE_KILLIP.get(patient.killip_class, 0)
    arrest_pts = _GRACE_CARDIAC_ARREST_POINTS if patient.cardiac_arrest_at_admission else 0
    st_pts = _GRACE_ST_DEVIATION_POINTS if patient.st_deviation_ecg else 0
    enzyme_pts = _GRACE_ELEVATED_ENZYMES_POINTS if patient.elevated_cardiac_enzymes else 0
    return age_pts + hr_pts + sbp_pts + cr_pts + killip_pts + arrest_pts + st_pts + enzyme_pts


def compute_grace_score(patient: PatientClinicalInput) -> GraceScoreResult:
    """
    Compute GRACE risk score using official bin-based lookup tables.
    Model: IN-HOSPITAL MORTALITY for NSTEMI only (we do NOT use 6-month model).
    No AI; no linear formulas; deterministic mapping per variable.

    Risk categories (NSTEMI in-hospital): Low ≤108, Intermediate 109–140, High >140.

    Safety: Before returning, validate engine against 5 test patients; if deviation >3, raise and block downstream.
    """
    _validate_grace_engine()

    total = _compute_grace_total_only(patient)

    if total <= GRACE_RISK_LOW_MAX:
        risk_category = "low"
        category_description = "Low risk (score ≤108): in-hospital mortality <1%."
    elif GRACE_RISK_INTERMEDIATE_MIN <= total <= GRACE_RISK_INTERMEDIATE_MAX:
        risk_category = "intermediate"
        category_description = "Intermediate risk (109–140): in-hospital mortality 1–3%."
    else:
        risk_category = "high"
        category_description = "High risk (>140): in-hospital mortality >3%."

    return GraceScoreResult(
        total_score=total,
        risk_category=risk_category,
        category_description=category_description,
    )
