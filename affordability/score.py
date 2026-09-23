"""Transparent points-based affordability model.

Every feature maps to points through a documented, monotone band table. The
score starts at :data:`BASE_POINTS` (50), each feature adds or subtracts its
band's points, and the total is clipped to 0–100. Because the bands are the
model, :func:`explain` is not an approximation of the score — it *is* the score.

Bands (``upper`` is inclusive; the last band is the catch-all):

======================  =========================================================
feature                 bands -> points
======================  =========================================================
rent_to_income          <=0.30 +20 · <=0.40 +5 · <=0.50 -15 · >0.50 -30
commitment_ratio        <=0.15 +10 · <=0.30 +3 · <=0.45 -8 · >0.45 -15
residual_ratio          <-0.05 -25 · <0.05 -15 · <0.20 0 · >=0.20 +10
income_cv               <=0.05 +8 · <=0.15 +4 · <=0.30 0 · >0.30 -10
salary_regularity       <=1 +5 · <=3 +2 · <=7 0 · >7 -5
min_balance             <=-2000 -15 · <0 -5 · <5000 +3 · >=5000 +6
overdraft_days          0 +5 · <=5 0 · <=20 -8 · >20 -15
gambling_share          <=0.01 +3 · <=0.05 0 · <=0.15 -10 · >0.15 -20
months_observed         <6 0 · >=6 +3
======================  =========================================================

The affordability features (rent share, fixed commitments, residual) carry the
weight; stability features can lift or sink a score by a band but cannot make an
unaffordable rent look strong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from affordability.features import FeatureVector

BASE_POINTS = 50

BAND_THRESHOLDS: dict[str, tuple[int, int]] = {
    "strong": (75, 100),
    "adequate": (60, 74),
    "marginal": (45, 59),
    "insufficient": (0, 44),
}


@dataclass(frozen=True)
class Band:
    upper: float | None  # value <= upper falls in this band; None = catch-all
    points: int
    reason: str


@dataclass(frozen=True)
class FeatureRule:
    feature: str
    label: str
    fmt: str  # "pct" | "rand" | "days" | "num" | "int"
    bands: tuple[Band, ...]

    def pick(self, value: float) -> Band:
        for band in self.bands:
            if band.upper is None or value <= band.upper:
                return band
        return self.bands[-1]


# Values are compared with a tiny epsilon so 0.30 sits inside the "<= 30%" band
# even after float arithmetic on the ratio.
_EPS = 1e-9

RULES: tuple[FeatureRule, ...] = (
    FeatureRule(
        "rent_to_income",
        "Rent as a share of net income",
        "pct",
        (
            Band(0.30 + _EPS, 20, "rent is at most 30% of net income, the usual comfort line"),
            Band(0.40 + _EPS, 5, "rent is between 30% and 40% of net income; workable but tight"),
            Band(0.50 + _EPS, -15, "rent is between 40% and 50% of net income; little room for shocks"),
            Band(None, -30, "rent is more than half of net income"),
        ),
    ),
    FeatureRule(
        "commitment_ratio",
        "Fixed commitments (debit orders + loans) as a share of income",
        "pct",
        (
            Band(0.15 + _EPS, 10, "fixed commitments are at most 15% of income"),
            Band(0.30 + _EPS, 3, "fixed commitments are between 15% and 30% of income"),
            Band(0.45 + _EPS, -8, "fixed commitments are between 30% and 45% of income"),
            Band(None, -15, "fixed commitments exceed 45% of income"),
        ),
    ),
    FeatureRule(
        "residual_ratio",
        "Income left after fixed costs, typical spending and the proposed rent",
        "pct",
        (
            Band(-0.05 - _EPS, -25, "spending plus the proposed rent already exceeds income"),
            Band(0.05 - _EPS, -15, "less than 5% of income would be left after rent"),
            Band(0.20 - _EPS, 0, "between 5% and 20% of income would be left after rent"),
            Band(None, 10, "at least 20% of income would be left after rent"),
        ),
    ),
    FeatureRule(
        "income_cv",
        "Month-to-month income variability",
        "pct",
        (
            Band(0.05 + _EPS, 8, "income varies by 5% or less month to month"),
            Band(0.15 + _EPS, 4, "income varies by up to 15% month to month"),
            Band(0.30 + _EPS, 0, "income varies by up to 30% month to month"),
            Band(None, -10, "income varies by more than 30% month to month"),
        ),
    ),
    FeatureRule(
        "salary_regularity",
        "Consistency of the pay day",
        "days",
        (
            Band(1.0 + _EPS, 5, "salary lands on the same day each month"),
            Band(3.0 + _EPS, 2, "salary lands within a few days of the same date each month"),
            Band(7.0 + _EPS, 0, "salary day drifts by up to a week"),
            Band(None, -5, "salary day drifts by more than a week"),
        ),
    ),
    FeatureRule(
        "min_balance",
        "Lowest balance in the period",
        "rand",
        (
            Band(-2000.0, -15, "the account went more than R2 000 overdrawn"),
            Band(-0.005, -5, "the account went overdrawn"),
            Band(4999.995, 3, "the balance never went below zero"),
            Band(None, 6, "the balance never fell below R5 000"),
        ),
    ),
    FeatureRule(
        "overdraft_days",
        "Days spent with a negative balance",
        "int",
        (
            Band(0, 5, "no days overdrawn"),
            Band(5, 0, "overdrawn on five days or fewer"),
            Band(20, -8, "overdrawn on up to 20 days"),
            Band(None, -15, "overdrawn on more than 20 days"),
        ),
    ),
    FeatureRule(
        "gambling_share",
        "Share of spending that went to gambling",
        "pct",
        (
            Band(0.01 + _EPS, 3, "no meaningful gambling spend"),
            Band(0.05 + _EPS, 0, "gambling is under 5% of spending"),
            Band(0.15 + _EPS, -10, "gambling is between 5% and 15% of spending"),
            Band(None, -20, "gambling is more than 15% of spending"),
        ),
    ),
    FeatureRule(
        "months_observed",
        "Months of statement history",
        "int",
        (
            Band(5, 0, "three to five months of history"),
            Band(None, 3, "six or more months of history"),
        ),
    ),
)

MAX_POINTS = sum(max(b.points for b in r.bands) for r in RULES)
MIN_POINTS = sum(min(b.points for b in r.bands) for r in RULES)


@dataclass(frozen=True)
class Contribution:
    feature: str
    label: str
    value: float
    display: str
    points: int
    reason: str

    def as_tuple(self) -> tuple[str, float, int, str]:
        return (self.feature, self.value, self.points, self.reason)


@dataclass
class ScoreResult:
    score: int
    band: str
    raw_points: int
    base: int
    contributions: list[Contribution] = field(default_factory=list)
    probability: float | None = None

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "band": self.band,
            "base": self.base,
            "raw_points": self.raw_points,
            "probability_no_default_12m": self.probability,
            "contributions": [
                {
                    "feature": c.feature,
                    "label": c.label,
                    "value": c.value,
                    "display": c.display,
                    "points": c.points,
                    "reason": c.reason,
                }
                for c in self.contributions
            ],
        }


def format_value(value: float, fmt: str) -> str:
    if fmt == "pct":
        return f"{value * 100:.1f}%"
    if fmt == "rand":
        return f"R{value:,.2f}"
    if fmt == "days":
        return f"{value:.1f} days"
    if fmt == "int":
        return f"{int(value)}"
    return f"{value:.2f}"


def band_for(score: int) -> str:
    for name, (lo, hi) in BAND_THRESHOLDS.items():
        if lo <= score <= hi:
            return name
    raise ValueError(f"score out of range: {score}")


def score_features(features: FeatureVector) -> ScoreResult:
    """Apply the band table. Deterministic, no fitting, no randomness."""
    contributions: list[Contribution] = []
    for rule in RULES:
        value = float(getattr(features, rule.feature))
        band = rule.pick(value)
        contributions.append(
            Contribution(
                feature=rule.feature,
                label=rule.label,
                value=value,
                display=format_value(value, rule.fmt),
                points=band.points,
                reason=band.reason,
            )
        )
    raw = sum(c.points for c in contributions)
    score = max(0, min(100, BASE_POINTS + raw))
    return ScoreResult(
        score=score,
        band=band_for(score),
        raw_points=raw,
        base=BASE_POINTS,
        contributions=contributions,
    )


def explain(result: ScoreResult) -> list[tuple[str, float, int, str]]:
    """``(feature, value, points, reason)`` ordered by absolute impact, largest first."""
    ordered = sorted(result.contributions, key=lambda c: (-abs(c.points), c.feature))
    return [c.as_tuple() for c in ordered]
