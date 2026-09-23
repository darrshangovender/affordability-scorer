"""Isotonic calibration of the 0–100 points score to P(no default in 12 months).

The points model is hand-built and monotone; calibration only re-labels the
score axis as a probability. It is fitted on a **synthetic** labelled set:
random feature vectors are drawn, a latent default process (that the points
model does not know about) labels them, the points model scores them, and
isotonic regression maps score -> observed no-default rate. No real default
data is involved, so the probabilities are illustrative until refit on real
outcomes — see the README's Limitations.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

from affordability.features import FeatureVector
from affordability.score import score_features


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def random_feature_vectors(n: int, seed: int = 7) -> list[FeatureVector]:
    """Draw plausible SA rental applicants across the income distribution."""
    rng = np.random.default_rng(seed)
    out: list[FeatureVector] = []
    income = np.exp(rng.normal(np.log(18_000), 0.55, n)).clip(4_000, 150_000)
    rent_ratio = rng.uniform(0.15, 0.65, n)
    fixed_ratio = rng.uniform(0.03, 0.55, n)
    disc_ratio = rng.uniform(0.10, 0.55, n)
    cv = rng.beta(1.2, 6, n) * 0.8
    regularity = rng.gamma(1.5, 1.5, n)
    min_balance = rng.normal(2_000, 6_000, n)
    overdraft = np.where(min_balance < 0, rng.poisson(12, n), 0) + rng.poisson(0.3, n)
    gambling = np.where(rng.random(n) < 0.25, rng.beta(1.2, 6, n) * 0.5, 0.0)
    months = rng.integers(3, 7, n)
    for i in range(n):
        inc = float(income[i])
        rent = round(inc * rent_ratio[i], 2)
        fixed = round(inc * fixed_ratio[i], 2)
        disc = round(inc * disc_ratio[i], 2)
        out.append(
            FeatureVector(
                net_income_median=round(inc, 2),
                income_cv=round(float(cv[i]), 4),
                salary_regularity=round(float(regularity[i]), 2),
                fixed_commitments=fixed,
                commitment_ratio=round(fixed / inc, 4),
                discretionary=disc,
                proposed_rent=rent,
                rent_to_income=round(rent / inc, 4),
                residual_ratio=round((inc - fixed - disc - rent) / inc, 4),
                min_balance=round(float(min_balance[i]), 2),
                overdraft_days=int(overdraft[i]),
                gambling_share=round(float(min(gambling[i], 1.0)), 4),
                months_observed=int(months[i]),
            )
        )
    return out


def latent_no_default_probability(fv: FeatureVector, rng: np.random.Generator) -> float:
    """A latent outcome process the points model does not see (adds noise on purpose)."""
    z = (
        0.4
        + 3.0 * max(-1.0, min(0.6, fv.residual_ratio))
        - 0.06 * fv.overdraft_days
        - 5.0 * fv.gambling_share
        - 2.0 * fv.income_cv
        + (0.5 if fv.min_balance > 0 else -0.3)
        - 1.5 * max(0.0, fv.rent_to_income - 0.35)
        + rng.normal(0.0, 0.9)
    )
    return float(_sigmoid(np.array(z)))


def synthetic_labelled_set(
    n: int = 3000, seed: int = 7
) -> tuple[list[FeatureVector], np.ndarray, np.ndarray]:
    """Returns ``(features, scores, outcomes)``; ``outcomes`` is 1 for no default."""
    rng = np.random.default_rng(seed + 1)
    fvs = random_feature_vectors(n, seed=seed)
    scores = np.array([score_features(fv).score for fv in fvs], dtype=float)
    probs = np.array([latent_no_default_probability(fv, rng) for fv in fvs])
    outcomes = (rng.random(n) < probs).astype(int)
    return fvs, scores, outcomes


class Calibrator:
    """Monotone score -> probability map. ``fit`` then ``predict``."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        self.fitted = False
        self.n_samples = 0

    def fit(self, scores: Sequence[float], outcomes: Sequence[int]) -> Calibrator:
        x = np.asarray(scores, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        if x.shape != y.shape or x.size < 10:
            raise ValueError("need at least 10 aligned (score, outcome) pairs")
        self._iso.fit(x, y)
        self.fitted = True
        self.n_samples = int(x.size)
        return self

    def predict(self, score: float) -> float:
        if not self.fitted:
            raise RuntimeError("calibrator is not fitted")
        return float(self._iso.predict(np.array([float(score)]))[0])

    def predict_many(self, scores: Sequence[float]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator is not fitted")
        return self._iso.predict(np.asarray(scores, dtype=float))


@lru_cache(maxsize=4)
def default_calibrator(seed: int = 7, n: int = 3000) -> Calibrator:
    """The calibrator the pipeline uses when none is supplied (synthetic fit, cached)."""
    _, scores, outcomes = synthetic_labelled_set(n=n, seed=seed)
    return Calibrator().fit(scores, outcomes)


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probabilities and outcomes must be non-empty and aligned")
    return float(np.mean((p - y) ** 2))
