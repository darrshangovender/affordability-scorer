"""Feature derivation from at least three months of classified transactions.

Every feature is a number about money flows. There is deliberately no field for
who the applicant is: :class:`FeatureVector` forbids extra fields, and the
fairness module checks its schema against the protected-ground list at import.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import timedelta
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from affordability.fairness import AuditReport, assert_no_protected_fields, proxy_audit
from affordability.parse.statement import Transaction

MIN_MONTHS = 3
RATIO_CAP = 9.99


class InsufficientHistoryError(ValueError):
    """Fewer than :data:`MIN_MONTHS` distinct months of transactions."""


class FeatureVector(BaseModel):
    """Money-only inputs to the scorer. ``extra="forbid"`` blocks any other field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    net_income_median: float = Field(ge=0, description="median monthly salary credit, R")
    income_cv: float = Field(ge=0, description="std/mean of monthly salary (0 = perfectly stable)")
    salary_regularity: float = Field(ge=0, description="std dev of salary day-of-month, days")
    fixed_commitments: float = Field(ge=0, description="median monthly debit orders + loans, R")
    commitment_ratio: float = Field(ge=0, description="fixed_commitments / net_income_median")
    discretionary: float = Field(ge=0, description="median monthly retail + cash + gambling, R")
    proposed_rent: float = Field(ge=0, description="rent being applied for, R/month")
    rent_to_income: float = Field(ge=0, description="proposed_rent / net_income_median")
    residual_ratio: float = Field(description="(income - fixed - discretionary - rent) / income")
    min_balance: float = Field(description="lowest closing balance in the period, R")
    overdraft_days: int = Field(ge=0, description="calendar days with a negative balance")
    gambling_share: float = Field(ge=0, le=1, description="gambling debits / all debits")
    months_observed: int = Field(ge=MIN_MONTHS)

    def to_dict(self) -> dict:
        return self.model_dump()


assert_no_protected_fields(FeatureVector)


def _month_key(t: Transaction) -> tuple[int, int]:
    return (t.date.year, t.date.month)


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return RATIO_CAP
    return min(RATIO_CAP, round(numerator / denominator, 4))


def overdraft_days(transactions: list[Transaction]) -> int:
    """Calendar days on which the carried-forward closing balance was negative."""
    dated = [t for t in transactions if t.balance is not None]
    if not dated:
        return 0
    closing: dict = {}
    for t in dated:
        closing[t.date] = t.balance  # last line of the day wins
    days = 0
    current = dated[0].date
    last = dated[-1].date
    balance = closing[current]
    while current <= last:
        if current in closing:
            balance = closing[current]
        if balance < 0:
            days += 1
        current += timedelta(days=1)
    return days


def build_features(
    transactions: Iterable[Transaction],
    proposed_rent: float,
    *,
    audit: AuditReport | None = None,
) -> FeatureVector:
    """Derive the :class:`FeatureVector` for a proposed rent.

    The proxy audit runs first (or is supplied) so that excluded transactions never
    feed a category sum. Balance-based features use the full ledger because the
    balance column is a fact about the account, not about any one payment.
    """
    ledger = sorted(transactions, key=lambda t: (t.date, t.line_no))
    if proposed_rent < 0:
        raise ValueError("proposed_rent must be non-negative")
    months = sorted({_month_key(t) for t in ledger})
    if len(months) < MIN_MONTHS:
        raise InsufficientHistoryError(
            f"need at least {MIN_MONTHS} months of transactions, got {len(months)}"
        )
    report = audit if audit is not None else proxy_audit(ledger)
    kept = report.kept

    by_month: dict[tuple[int, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for t in kept:
        by_month[_month_key(t)][t.category] += t.amount

    salary_credits = [t for t in kept if t.category == "salary" and t.is_credit]
    salary_months = {_month_key(t) for t in salary_credits}
    # A partial first/last month with no pay day yet is not evidence of missing income.
    income_series = [
        by_month[m]["salary"]
        for m in months
        if m in salary_months or m not in (months[0], months[-1])
    ]
    net_income_median = round(_median(income_series), 2)
    if income_series and net_income_median > 0 and len(income_series) > 1:
        income_cv = round(statistics.pstdev(income_series) / statistics.fmean(income_series), 4)
    elif net_income_median > 0:
        income_cv = 0.0
    else:
        income_cv = 1.0

    pay_days = [t.date.day for t in salary_credits]
    salary_regularity = round(statistics.pstdev(pay_days), 2) if len(pay_days) > 1 else 0.0

    fixed_series = [-(by_month[m]["debit_order"] + by_month[m]["loan"]) for m in months]
    fixed_commitments = round(max(0.0, _median(fixed_series)), 2)
    disc_series = [
        -(by_month[m]["retail"] + by_month[m]["cash"] + by_month[m]["gambling"]) for m in months
    ]
    discretionary = round(max(0.0, _median(disc_series)), 2)

    debits = [-t.amount for t in kept if t.amount < 0]
    total_debits = sum(debits)
    gambling = sum(-t.amount for t in kept if t.amount < 0 and t.category == "gambling")
    gambling_share = round(gambling / total_debits, 4) if total_debits > 0 else 0.0

    balances = [t.balance for t in ledger if t.balance is not None]
    min_balance = round(min(balances), 2) if balances else 0.0

    if net_income_median > 0:
        residual = (net_income_median - fixed_commitments - discretionary - proposed_rent)
        residual_ratio = max(-RATIO_CAP, round(residual / net_income_median, 4))
    else:
        residual_ratio = -RATIO_CAP

    return FeatureVector(
        net_income_median=net_income_median,
        income_cv=income_cv,
        salary_regularity=salary_regularity,
        fixed_commitments=fixed_commitments,
        commitment_ratio=_safe_ratio(fixed_commitments, net_income_median),
        discretionary=discretionary,
        proposed_rent=float(proposed_rent),
        rent_to_income=_safe_ratio(proposed_rent, net_income_median),
        residual_ratio=residual_ratio,
        min_balance=min_balance,
        overdraft_days=overdraft_days(ledger),
        gambling_share=min(1.0, gambling_share),
        months_observed=len(months),
    )
