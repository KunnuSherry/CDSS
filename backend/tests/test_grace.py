from services.grace import _GRACE_TEST_PATIENTS, compute_grace_score
from models.clinical import PatientClinicalInput


def test_grace_internal_vectors():
    """Validate internal GRACE test vectors compute as expected (deterministic)."""
    for inp, expected in _GRACE_TEST_PATIENTS:
        p = PatientClinicalInput(**inp)
        res = compute_grace_score(p)
        assert res.total_score == expected, f"Expected {expected} got {res.total_score} for {inp}"
