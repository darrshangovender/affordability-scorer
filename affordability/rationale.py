"""Plain-English rationale for both parties, built deterministically from ``explain()``.

The landlord text and the tenant text are generated from the same contributions,
so neither side can be told something the other was not. An optional
:class:`ModelClient` may polish the wording; :func:`polish` asserts that every
number and every reason sentence in the deterministic text survives, and the
prompt is PII-redacted before it leaves the process.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from affordability.fairness import AuditReport
from affordability.features import FeatureVector
from affordability.parse.tamper import TamperReport
from affordability.privacy import redact
from affordability.score import ScoreResult, explain


class RationaleIntegrityError(RuntimeError):
    """The polished text dropped or altered a number or a reason."""


@dataclass(frozen=True)
class Rationale:
    landlord: str
    tenant: str
    facts: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "landlord": self.landlord,
            "tenant": self.tenant,
            "facts": list(self.facts),
            "reasons": list(self.reasons),
        }


_BAND_SENTENCE = {
    "strong": "comfortably affordable on the evidence supplied",
    "adequate": "affordable, with a reasonable buffer",
    "marginal": "affordable only with little margin for surprises",
    "insufficient": "not affordable on the evidence supplied",
}


def _fmt_r(value: float) -> str:
    return f"R{value:,.0f}"


def _display_for(result: ScoreResult, feature: str) -> str:
    for c in result.contributions:
        if c.feature == feature:
            return c.display
    return ""


def build_rationale(
    result: ScoreResult,
    features: FeatureVector,
    *,
    tamper: TamperReport | None = None,
    audit: AuditReport | None = None,
) -> Rationale:
    contributions = explain(result)
    plus = [(f, v, p, r) for (f, v, p, r) in contributions if p > 0]
    minus = [(f, v, p, r) for (f, v, p, r) in contributions if p < 0]
    # Zero-point items are still stated: every reason in ``reasons`` must appear in
    # both texts, which is what ``polish`` verifies afterwards.
    neutral = [(f, v, p, r) for (f, v, p, r) in contributions if p == 0]
    facts = (
        f"Proposed rent {_fmt_r(features.proposed_rent)} per month.",
        (
            f"Median net monthly income {_fmt_r(features.net_income_median)} "
            f"over {features.months_observed} months."
        ),
        f"Rent would take {_display_for(result, 'rent_to_income')} of net income.",
        (
            f"Fixed commitments {_fmt_r(features.fixed_commitments)} per month "
            f"({_display_for(result, 'commitment_ratio')} of income)."
        ),
        f"Score {result.score}/100, band '{result.band}'.",
    )
    reasons = tuple(r for (_, _, _, r) in contributions)

    def lines(items: list[tuple[str, float, int, str]]) -> str:
        return "\n".join(f"  {p:+d}: {r}" for (_, _, p, r) in items)

    prob = (
        f" Calibrated likelihood of 12 months without a missed payment: {result.probability:.0%}."
        if result.probability is not None
        else ""
    )
    tamper_line = ""
    if tamper is not None and not tamper.ok:
        tamper_line = f"\n\nDocument check: {tamper.summary()}. Verify the originals before relying on this score."
    audit_line = ""
    if audit is not None and (audit.excluded or audit.stripped):
        audit_line = f"\n\nFairness audit: {audit.summary()}."

    landlord = (
        f"Affordability: {result.score}/100 ({result.band}). The proposed rent of "
        f"{_fmt_r(features.proposed_rent)} is {_BAND_SENTENCE[result.band]}.{prob}\n\n"
        + "\n".join(facts)
        + "\n\nWhat helped:\n"
        + (lines(plus) or "  (nothing)")
        + "\n\nWhat counted against:\n"
        + (lines(minus) or "  (nothing)")
        + "\n\nNoted, no points either way:\n"
        + (lines(neutral) or "  (nothing)")
        + "\n\nThe score is the sum of these points added to a base of "
        f"{result.base}; there are no hidden factors. It uses only money flows "
        "from the documents supplied and no personal characteristics."
        + tamper_line
        + audit_line
    )
    tenant = (
        f"Your affordability score is {result.score}/100 ({result.band}). On the documents "
        f"you supplied, rent of {_fmt_r(features.proposed_rent)} is "
        f"{_BAND_SENTENCE[result.band]}.{prob}\n\n"
        + "\n".join(facts)
        + "\n\nIn your favour:\n"
        + (lines(plus) or "  (nothing)")
        + "\n\nWhat lowered the score:\n"
        + (lines(minus) or "  (nothing)")
        + "\n\nNoted, no points either way:\n"
        + (lines(neutral) or "  (nothing)")
        + "\n\nEvery point above is explained; the landlord sees exactly the same reasons. "
        "Nothing about who you are was used, only how money moves through the account."
        + audit_line
    )
    return Rationale(landlord=landlord, tenant=tenant, facts=facts, reasons=reasons)


# --------------------------------------------------------------------------- optional polish
class ModelClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class MockModel:
    """Echoes the text back with a wrapper so integrity checks can be exercised offline."""

    def __init__(self, *, drop_numbers: bool = False) -> None:
        self.prompts: list[str] = []
        self.drop_numbers = drop_numbers

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        body = prompt.split("TEXT:\n", 1)[1] if "TEXT:\n" in prompt else prompt
        if self.drop_numbers:
            body = re.sub(r"\d", "#", body)
        return "Here is a friendlier version:\n\n" + body


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


def numbers_in(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def verify_preserved(original: str, polished: str, reasons: tuple[str, ...]) -> None:
    missing_numbers = [n for n in numbers_in(original) if n not in polished]
    missing_reasons = [r for r in reasons if r not in polished]
    if missing_numbers or missing_reasons:
        raise RationaleIntegrityError(
            f"polished text lost numbers {missing_numbers[:5]} and reasons {missing_reasons[:3]}"
        )


def _prompt(text: str) -> str:
    return (
        "Rewrite the following tenant-affordability rationale in warmer plain English. "
        "Keep every number and every reason sentence exactly as written. Do not add facts.\n"
        "TEXT:\n" + redact(text)
    )


def polish(rationale: Rationale, client: ModelClient) -> Rationale:
    """Ask a model to re-word both texts; raise if any number or reason is lost."""
    landlord = client.complete(_prompt(rationale.landlord))
    verify_preserved(rationale.landlord, landlord, rationale.reasons)
    tenant = client.complete(_prompt(rationale.tenant))
    verify_preserved(rationale.tenant, tenant, rationale.reasons)
    return Rationale(
        landlord=landlord, tenant=tenant, facts=rationale.facts, reasons=rationale.reasons
    )
