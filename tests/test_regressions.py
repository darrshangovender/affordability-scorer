"""Regression tests for 0.1.1."""

import re
from datetime import date
from pathlib import Path

import pytest

from affordability.privacy import is_sa_id
from affordability.rationale import MockModel, build_rationale, polish
from affordability.score import score_features
from tests.test_score import _fv

ROOT = Path(__file__).resolve().parents[1]

# --- a clipped score is explained, not left for the reader to mis-add -------------------


def test_rationale_states_the_cap_when_points_exceed_100():
    """+70 on a base of 50 used to be listed under a score of 100 with no explanation."""
    fv = _fv()
    res = score_features(fv)
    assert res.score == 100 and res.base + res.raw_points > 100
    r = build_rationale(res, fv)
    line = f"({res.base + res.raw_points}); the score is capped to the 0-100 range."
    assert line in r.landlord and line in r.tenant
    assert f"{res.raw_points:+d} on a base of {res.base}" in r.landlord


def test_rationale_states_the_cap_when_points_go_below_0():
    fv = _fv(rent_to_income=0.9, commitment_ratio=0.9, residual_ratio=-2.0, income_cv=0.9,
             salary_regularity=10.0, min_balance=-9_000.0, overdraft_days=60,
             gambling_share=0.5, months_observed=3)
    res = score_features(fv)
    assert res.score == 0 and res.base + res.raw_points < 0
    assert "capped to the 0-100 range" in build_rationale(res, fv).tenant


def test_no_cap_line_when_the_score_is_not_clipped():
    fv = _fv(rent_to_income=0.45, commitment_ratio=0.35, residual_ratio=0.1)
    res = score_features(fv)
    assert res.score == res.base + res.raw_points
    assert "capped" not in build_rationale(res, fv).landlord


def test_cap_line_survives_the_polish_integrity_check():
    fv = _fv()
    res = score_features(fv)
    r = build_rationale(res, fv)
    out = polish(r, MockModel())
    assert "capped" in out.landlord


# --- demo/test ID numbers must not be possible real IDs ----------------------------------


def _ids(text: str) -> list[str]:
    return re.findall(r"(?<!\d)\d{13}(?!\d)", text)


FILES = sorted((ROOT / "demo" / "data").rglob("*.txt")) + sorted((ROOT / "tests").glob("*.py"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_shipped_id_numbers_have_impossible_birth_dates(path):
    """Synthetic IDs used random plausible dates and sequence numbers, so some were very
    likely real people's IDs. Every ID that passes ``is_sa_id`` now carries 30/31 February."""
    for n in _ids(path.read_text(encoding="utf-8")):
        if not is_sa_id(n):
            continue
        yy, mm, dd = int(n[:2]), int(n[2:4]), int(n[4:6])
        with pytest.raises(ValueError):
            date(1900 + yy, mm, dd)
