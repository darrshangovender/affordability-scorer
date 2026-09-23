from dataclasses import replace
from datetime import date

import pytest
from pydantic import ValidationError

from affordability.features import (
    FeatureVector,
    InsufficientHistoryError,
    build_features,
    overdraft_days,
)
from affordability.parse.statement import Transaction
from tests.conftest import make_transactions


def test_feature_maths_on_hand_built_ledger(transactions):
    fv = build_features(transactions, proposed_rent=6_000.0)
    assert fv.net_income_median == 20_000.0
    assert fv.income_cv == 0.0
    assert fv.salary_regularity == 0.0
    assert fv.fixed_commitments == 3_000.0  # debit order 1000 + loan 2000
    assert fv.commitment_ratio == pytest.approx(0.15)
    assert fv.discretionary == 3_987.25  # retail 3000 + cash 500 + gambling 487.25
    assert fv.proposed_rent == 6_000.0
    assert fv.rent_to_income == pytest.approx(0.30)
    assert fv.residual_ratio == pytest.approx((20_000 - 3_000 - 3_987.25 - 6_000) / 20_000, abs=1e-4)
    assert fv.gambling_share == pytest.approx(487.25 / 12_096.25, abs=1e-4)
    assert fv.overdraft_days == 0
    assert fv.months_observed == 4
    assert fv.min_balance == min(t.balance for t in transactions)


def test_current_rent_is_not_a_fixed_commitment(transactions):
    """The proposed rent replaces the current one; counting both would double-charge."""
    fv = build_features(transactions, proposed_rent=0.0)
    assert fv.fixed_commitments == 3_000.0


def test_insufficient_history_raises(transactions):
    two_months = [t for t in transactions if t.date.month <= 4]
    with pytest.raises(InsufficientHistoryError):
        build_features(two_months, 5_000.0)


def test_negative_rent_rejected(transactions):
    with pytest.raises(ValueError):
        build_features(transactions, -1.0)


def test_income_cv_reflects_variable_income():
    txns = make_transactions(salary=20_000.0)
    bumped = [replace(t, amount=30_000.0) if t.category == "salary" and t.date.month == 4 else t for t in txns]
    assert build_features(bumped, 5_000.0).income_cv > 0.1


def test_salary_regularity_measures_pay_day_drift(transactions):
    drift = [replace(t, date=date(t.date.year, t.date.month, 15)) if t.category == "salary" and t.date.month == 5 else t
             for t in transactions]
    assert build_features(drift, 5_000.0).salary_regularity > 3


def test_overdraft_days_counts_calendar_days():
    txns = [
        Transaction(date(2026, 3, 1), "A", -100.0, -50.0, "retail", None, 1),
        Transaction(date(2026, 3, 5), "B", 100.0, 50.0, "other", None, 2),
        Transaction(date(2026, 3, 9), "C", -100.0, -50.0, "retail", None, 3),
        Transaction(date(2026, 3, 10), "D", 100.0, 50.0, "other", None, 4),
    ]
    assert overdraft_days(txns) == 5  # 1-4 March and 9 March


def test_overdraft_days_without_balances_is_zero():
    assert overdraft_days([Transaction(date(2026, 3, 1), "A", -1.0, None, "retail", None, 1)]) == 0


def test_zero_income_caps_ratios():
    txns = [t for t in make_transactions() if t.category != "salary"]
    fv = build_features(txns, 5_000.0)
    assert fv.net_income_median == 0.0
    assert fv.rent_to_income == 9.99
    assert fv.income_cv == 1.0
    assert fv.residual_ratio == -9.99


def test_partial_edge_month_without_salary_is_ignored(transactions):
    extra = Transaction(date(2026, 7, 2), "CHECKERS HYPER", -50.0, 100.0, "retail", None, 999)
    fv = build_features([*transactions, extra], 5_000.0)
    assert fv.net_income_median == 20_000.0
    assert fv.months_observed == 5


def test_feature_vector_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        FeatureVector(
            net_income_median=1.0, income_cv=0.0, salary_regularity=0.0, fixed_commitments=0.0,
            commitment_ratio=0.0, discretionary=0.0, proposed_rent=0.0, rent_to_income=0.0,
            residual_ratio=0.0, min_balance=0.0, overdraft_days=0, gambling_share=0.0,
            months_observed=3, postcode="4001",
        )


def test_feature_vector_is_frozen(transactions):
    fv = build_features(transactions, 5_000.0)
    with pytest.raises(ValidationError):
        fv.net_income_median = 1.0  # type: ignore[misc]
    assert set(fv.to_dict()) == set(FeatureVector.model_fields)
