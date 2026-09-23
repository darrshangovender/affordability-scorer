import numpy as np
import pytest

from affordability.calibrate import (
    Calibrator,
    brier_score,
    default_calibrator,
    random_feature_vectors,
    synthetic_labelled_set,
)


def test_synthetic_set_is_deterministic():
    a = synthetic_labelled_set(n=200, seed=3)
    b = synthetic_labelled_set(n=200, seed=3)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    assert set(a[2].tolist()) == {0, 1}


def test_calibrator_is_monotone_and_bounded():
    cal = default_calibrator()
    probs = [cal.predict(s) for s in range(0, 101, 5)]
    assert probs == sorted(probs)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert cal.predict(100) > cal.predict(0)


def test_calibrator_requires_fit():
    with pytest.raises(RuntimeError):
        Calibrator().predict(50)


def test_calibrator_rejects_tiny_sets():
    with pytest.raises(ValueError):
        Calibrator().fit([1, 2, 3], [0, 1, 0])


def test_predict_many_matches_predict():
    cal = default_calibrator()
    many = cal.predict_many([10, 50, 90])
    assert list(many) == [cal.predict(10), cal.predict(50), cal.predict(90)]


def test_brier_score():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        brier_score([], [])


def test_calibration_beats_base_rate_on_holdout():
    cal = default_calibrator()
    _, scores, outcomes = synthetic_labelled_set(n=1500, seed=42)
    fitted = brier_score(cal.predict_many(scores), outcomes)
    baseline = brier_score([float(outcomes.mean())] * len(outcomes), outcomes)
    assert fitted < baseline


def test_random_feature_vectors_are_valid():
    fvs = random_feature_vectors(30, seed=1)
    assert len(fvs) == 30
    assert all(fv.months_observed >= 3 and 0 <= fv.gambling_share <= 1 for fv in fvs)
