import pytest

from affordability.calibrate import random_feature_vectors
from affordability.features import FeatureVector, build_features
from affordability.score import (
    BAND_THRESHOLDS,
    BASE_POINTS,
    MAX_POINTS,
    MIN_POINTS,
    RULES,
    band_for,
    explain,
    format_value,
    score_features,
)
from tests.conftest import make_transactions


def _fv(**overrides) -> FeatureVector:
    base = dict(
        net_income_median=20_000.0, income_cv=0.0, salary_regularity=0.0, fixed_commitments=2_000.0,
        commitment_ratio=0.10, discretionary=4_000.0, proposed_rent=5_000.0, rent_to_income=0.25,
        residual_ratio=0.45, min_balance=6_000.0, overdraft_days=0, gambling_share=0.0, months_observed=6,
    )
    base.update(overrides)
    return FeatureVector(**base)


def _points(feature: str, value: float) -> int:
    rule = next(r for r in RULES if r.feature == feature)
    return rule.pick(value).points


# --------------------------------------------------------------------------- band boundaries
@pytest.mark.parametrize(
    "feature,value,points",
    [
        ("rent_to_income", 0.30, 20), ("rent_to_income", 0.3001, 5),
        ("rent_to_income", 0.40, 5), ("rent_to_income", 0.4001, -15),
        ("rent_to_income", 0.50, -15), ("rent_to_income", 0.5001, -30),
        ("commitment_ratio", 0.15, 10), ("commitment_ratio", 0.1501, 3),
        ("commitment_ratio", 0.30, 3), ("commitment_ratio", 0.3001, -8),
        ("commitment_ratio", 0.45, -8), ("commitment_ratio", 0.4501, -15),
        ("residual_ratio", -0.0501, -25), ("residual_ratio", -0.05, -15),
        ("residual_ratio", 0.0499, -15), ("residual_ratio", 0.05, 0),
        ("residual_ratio", 0.1999, 0), ("residual_ratio", 0.20, 10),
        ("income_cv", 0.05, 8), ("income_cv", 0.0501, 4),
        ("income_cv", 0.15, 4), ("income_cv", 0.1501, 0),
        ("income_cv", 0.30, 0), ("income_cv", 0.3001, -10),
        ("salary_regularity", 1.0, 5), ("salary_regularity", 1.01, 2),
        ("salary_regularity", 3.0, 2), ("salary_regularity", 3.01, 0),
        ("salary_regularity", 7.0, 0), ("salary_regularity", 7.01, -5),
        ("min_balance", -2000.0, -15), ("min_balance", -1999.99, -5),
        ("min_balance", -0.01, -5), ("min_balance", 0.0, 3),
        ("min_balance", 4999.99, 3), ("min_balance", 5000.0, 6),
        ("overdraft_days", 0, 5), ("overdraft_days", 1, 0),
        ("overdraft_days", 5, 0), ("overdraft_days", 6, -8),
        ("overdraft_days", 20, -8), ("overdraft_days", 21, -15),
        ("gambling_share", 0.01, 3), ("gambling_share", 0.0101, 0),
        ("gambling_share", 0.05, 0), ("gambling_share", 0.0501, -10),
        ("gambling_share", 0.15, -10), ("gambling_share", 0.1501, -20),
        ("months_observed", 5, 0), ("months_observed", 6, 3),
    ],
)
def test_band_boundaries(feature, value, points):
    assert _points(feature, value) == points


def test_every_rule_is_monotone():
    """Points never increase as a risk-side feature worsens (or decrease as a good one improves)."""
    increasing_good = {"residual_ratio", "min_balance", "months_observed"}
    for rule in RULES:
        pts = [b.points for b in rule.bands]
        if rule.feature in increasing_good:
            assert pts == sorted(pts), rule.feature
        else:
            assert pts == sorted(pts, reverse=True), rule.feature


@pytest.mark.parametrize(
    "score,band",
    [(0, "insufficient"), (44, "insufficient"), (45, "marginal"), (59, "marginal"),
     (60, "adequate"), (74, "adequate"), (75, "strong"), (100, "strong")],
)
def test_score_band_boundaries(score, band):
    assert band_for(score) == band


def test_band_thresholds_cover_0_to_100_exactly():
    covered = sorted(v for lo, hi in BAND_THRESHOLDS.values() for v in range(lo, hi + 1))
    assert covered == list(range(101))
    with pytest.raises(ValueError):
        band_for(101)


# --------------------------------------------------------------------------- totals
def test_score_is_base_plus_points_clipped():
    res = score_features(_fv())
    assert res.score == max(0, min(100, BASE_POINTS + res.raw_points))
    assert res.raw_points == sum(c.points for c in res.contributions)


def test_best_and_worst_cases_clip():
    best = _fv()
    assert score_features(best).score == 100
    assert BASE_POINTS + MAX_POINTS >= 100
    worst = _fv(rent_to_income=0.9, commitment_ratio=0.9, residual_ratio=-2.0, income_cv=0.9,
                salary_regularity=10.0, min_balance=-9_000.0, overdraft_days=60, gambling_share=0.5,
                months_observed=3)
    assert score_features(worst).score == 0
    assert BASE_POINTS + MIN_POINTS <= 0


def test_explain_sums_to_total():
    for fv in random_feature_vectors(50, seed=3):
        res = score_features(fv)
        rows = explain(res)
        assert len(rows) == len(RULES)
        assert sum(p for _, _, p, _ in rows) == res.raw_points
        assert res.score == max(0, min(100, res.base + sum(p for _, _, p, _ in rows)))


def test_explain_is_ordered_by_absolute_impact():
    rows = explain(score_features(_fv(gambling_share=0.5, min_balance=-3_000.0)))
    magnitudes = [abs(p) for _, _, p, _ in rows]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(isinstance(r, str) and r for _, _, _, r in rows)


def test_explain_tuple_shape():
    feature, value, points, reason = explain(score_features(_fv()))[0]
    assert isinstance(feature, str) and isinstance(value, float)
    assert isinstance(points, int) and isinstance(reason, str)


def test_monotone_more_income_never_lowers_score():
    """Scale the salary up on the same ledger; the score must not fall."""
    prev = None
    for salary in (8_000.0, 12_000.0, 16_000.0, 20_000.0, 30_000.0, 50_000.0):
        score = score_features(build_features(make_transactions(salary=salary), 6_000.0)).score
        if prev is not None:
            assert score >= prev, salary
        prev = score


def test_monotone_on_random_vectors():
    for fv in random_feature_vectors(100, seed=5):
        inc = fv.net_income_median * 1.25
        richer = fv.model_copy(
            update={
                "net_income_median": inc,
                "commitment_ratio": fv.fixed_commitments / inc,
                "rent_to_income": fv.proposed_rent / inc,
                "residual_ratio": (inc - fv.fixed_commitments - fv.discretionary - fv.proposed_rent) / inc,
            }
        )
        assert score_features(richer).score >= score_features(fv).score


def test_deterministic():
    fv = _fv()
    assert score_features(fv).to_dict() == score_features(fv).to_dict()


def test_format_value():
    assert format_value(0.256, "pct") == "25.6%"
    assert format_value(1234.5, "rand") == "R1,234.50"
    assert format_value(2.0, "days") == "2.0 days"
    assert format_value(6, "int") == "6"
    assert format_value(1.2345, "num") == "1.23"


def test_to_dict_is_json_friendly():
    d = score_features(_fv()).to_dict()
    assert {"score", "band", "base", "raw_points", "contributions"} <= set(d)
    assert d["contributions"][0].keys() == {"feature", "label", "value", "display", "points", "reason"}
