from datetime import date

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from affordability.fairness import (
    PROTECTED_GROUNDS,
    ProtectedAttributeError,
    assert_no_protected_fields,
    disparate_impact_test,
    proxy_audit,
)
from affordability.features import FeatureVector
from affordability.parse.statement import Transaction, classify

_BASE = dict(
    net_income_median=1.0, income_cv=0.0, salary_regularity=0.0, fixed_commitments=0.0,
    commitment_ratio=0.0, discretionary=0.0, proposed_rent=0.0, rent_to_income=0.0,
    residual_ratio=0.0, min_balance=0.0, overdraft_days=0, gambling_share=0.0, months_observed=3,
)


@pytest.mark.parametrize("attr", ["race", "gender", "age", "religion", "nationality", "suburb", "surname"])
def test_feature_vector_cannot_hold_protected_attribute(attr):
    with pytest.raises(ValidationError):
        FeatureVector(**_BASE, **{attr: "x"})


def test_feature_vector_schema_passes_the_guard():
    assert_no_protected_fields(FeatureVector)
    assert not (set(FeatureVector.model_fields) & set(PROTECTED_GROUNDS))


def test_guard_rejects_a_smuggled_field():
    class Bad(BaseModel):
        net_income_median: float
        applicant_age: int

    with pytest.raises(ProtectedAttributeError):
        assert_no_protected_fields(Bad)


def test_guard_rejects_non_pydantic():
    with pytest.raises(TypeError):
        assert_no_protected_fields(dict)


def _t(desc: str, amount: float = -100.0, line: int = 1) -> Transaction:
    return Transaction(date(2026, 3, 1), desc, amount, 0.0, classify(desc), None, line)


@pytest.mark.parametrize(
    "desc,ground",
    [
        ("TITHE - GRACE FAMILY CHURCH", "religion"),
        ("ZAKAAT MASJID AL NOOR", "religion"),
        ("MAINTENANCE PAYMENT S DLAMINI", "family_responsibility"),
        ("LITTLE STARS CRECHE", "family_responsibility"),
        ("DEBIT ORDER DISCOVERY MEDICAL AID", "health_disability"),
        ("PATHCARE LABORATORIES", "health_disability"),
        ("DEPT OF HOME AFFAIRS PERMIT", "nationality"),
        ("VFS GLOBAL VISA FEE", "nationality"),
        ("WESTERN UNION REMITTANCE", "nationality"),
    ],
)
def test_proxy_audit_excludes_and_explains(desc, ground):
    report = proxy_audit([_t(desc), _t("CHECKERS HYPER", line=2)])
    assert [t.description for t in report.kept] == ["CHECKERS HYPER"]
    assert len(report.excluded) == 1
    item = report.excluded[0]
    assert item.ground == ground
    assert item.action == "exclude"
    assert item.why  # every exclusion says why
    assert desc.split()[0].lower() in item.matched.lower() or item.matched


def test_suburb_is_stripped_but_amount_kept():
    report = proxy_audit([_t("PNP UMLAZI MEGA CITY", -250.0)])
    assert report.excluded == []
    assert len(report.stripped) == 1
    assert report.kept[0].description == "PNP [LOCATION] MEGA CITY"
    assert report.kept[0].amount == -250.0
    assert report.stripped[0].ground == "location"


def test_salary_credit_from_hospital_employer_is_kept():
    """Employer names are not health data; excluding them would zero a nurse's income."""
    txns = [
        _t("COASTAL CARE HOSPITAL GROUP SALARY REF 0620260825", 18_500.0),
        _t("RIDGE MEDICAL PARTNERS SALARY REF 1220260825", 85_000.0, 2),
        _t("DEBIT ORDER DISCOVERY MEDICAL AID", -2_100.0, 3),
    ]
    report = proxy_audit(txns)
    assert [t.amount for t in report.kept] == [18_500.0, 85_000.0]
    assert len(report.excluded) == 1


def test_salary_credit_still_has_location_stripped():
    report = proxy_audit([_t("UMHLANGA CLINIC SALARY", 9_000.0)])
    assert report.kept[0].description == "[LOCATION] CLINIC SALARY"


def test_audit_report_summary_and_dict():
    report = proxy_audit([_t("TITHE CHURCH"), _t("SPAR KLOOF", line=2), _t("KFC", line=3)])
    assert report.by_ground == {"location": 1, "religion": 1}
    assert "excluded 1" in report.summary()
    d = report.to_dict()
    assert d["kept"] == 2 and d["excluded"][0]["why"]
    assert proxy_audit([_t("KFC")]).summary().startswith("no proxy indicators")


def test_audit_description_is_redacted():
    report = proxy_audit([_t("TITHE CHURCH ID 9205244136087")])
    assert "9205244136087" not in report.excluded[0].description


def test_disparate_impact_is_about_one_on_random_groups():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 101, size=4000)
    ratios = [
        disparate_impact_test(scores.tolist(), rng.choice(["A", "B"], size=len(scores)).tolist()).ratio
        for _ in range(20)
    ]
    assert np.mean(ratios) > 0.95
    assert min(ratios) > 0.85


def test_disparate_impact_fails_on_skewed_groups():
    scores = [80] * 100 + [30] * 100
    groups = ["A"] * 100 + ["B"] * 100
    result = disparate_impact_test(scores, groups)
    assert result.ratio == 0.0 and not result.passes
    assert result.rates == {"A": 1.0, "B": 0.0}
    assert result.to_dict()["threshold"] == 0.8


def test_disparate_impact_validation():
    with pytest.raises(ValueError):
        disparate_impact_test([1, 2], ["A"])
    with pytest.raises(ValueError):
        disparate_impact_test([], [])
    assert disparate_impact_test([1, 2], ["A", "B"]).ratio == 1.0  # nobody passes: no impact
