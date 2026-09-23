"""Fairness guarantees: no protected attributes, no proxies, and an offline impact test.

South African law lists the grounds on which discrimination is prohibited
(Employment Equity Act s.6, PEPUDA s.1, Constitution s.9). The scorer enforces
three things:

1. :func:`assert_no_protected_fields` — the ``FeatureVector`` schema cannot carry a
   protected attribute. It is checked at import time and in the test suite.
2. :func:`proxy_audit` — transaction descriptions that reveal a protected ground
   (a tithe reveals religion, a maintenance order reveals family status, a Home
   Affairs fee reveals nationality, a clinic reveals health) are **excluded** from
   every feature. Suburb names are stripped from descriptions but the amount is
   kept, because the spend is real and only the location is a proxy.
3. :func:`disparate_impact_test` — the four-fifths rule for offline auditing on
   synthetic or held-out groups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from affordability.parse.statement import Transaction
from affordability.privacy import redact

PROTECTED_GROUNDS: tuple[str, ...] = (
    "race",
    "gender",
    "sex",
    "pregnancy",
    "marital_status",
    "ethnic_origin",
    "social_origin",
    "colour",
    "sexual_orientation",
    "age",
    "disability",
    "religion",
    "hiv_status",
    "conscience",
    "belief",
    "political_opinion",
    "culture",
    "language",
    "birth",
    "nationality",
    "family_responsibility",
)

# Field-name tokens that would smuggle a protected ground into the schema.
PROTECTED_FIELD_TOKENS: frozenset[str] = frozenset(
    {
        "race",
        "ethnicity",
        "ethnic",
        "gender",
        "sex",
        "female",
        "male",
        "pregnancy",
        "pregnant",
        "marital",
        "married",
        "spouse",
        "age",
        "dob",
        "birthdate",
        "birth",
        "disability",
        "disabled",
        "religion",
        "religious",
        "faith",
        "nationality",
        "citizenship",
        "citizen",
        "foreign",
        "language",
        "hiv",
        "dependants",
        "dependents",
        "children",
        "kids",
        "surname",
        "name",
        "id_number",
        "suburb",
        "postcode",
        "postal_code",
        "address",
    }
)


class ProtectedAttributeError(ValueError):
    """A protected ground found its way into the scoring schema."""


def assert_no_protected_fields(model_cls: type) -> None:
    """Raise if any pydantic field of ``model_cls`` names a protected ground."""
    fields = getattr(model_cls, "model_fields", None)
    if fields is None:
        raise TypeError(f"{model_cls!r} is not a pydantic model")
    for name in fields:
        tokens = set(re.split(r"[_\W]+", name.lower())) | {name.lower()}
        hit = tokens & PROTECTED_FIELD_TOKENS
        if hit:
            raise ProtectedAttributeError(
                f"field {name!r} on {model_cls.__name__} names protected ground(s) {sorted(hit)}"
            )


# --------------------------------------------------------------------------- proxy audit
@dataclass(frozen=True)
class ProxyRule:
    ground: str
    pattern: re.Pattern[str]
    why: str
    action: str = "exclude"  # "exclude" drops the transaction; "strip" removes the match


SA_SUBURBS: tuple[str, ...] = (
    "umlazi",
    "kwamashu",
    "phoenix",
    "chatsworth",
    "inanda",
    "ntuzuma",
    "soweto",
    "alexandra",
    "tembisa",
    "diepsloot",
    "khayelitsha",
    "mitchells plain",
    "gugulethu",
    "langa",
    "mamelodi",
    "soshanguve",
    "umhlanga",
    "sandton",
    "constantia",
    "bishopscourt",
    "ballito",
    "hillcrest",
    "kloof",
    "westville",
    "durban north",
    "musgrave",
    "morningside",
    "berea",
    "wentworth",
    "lenasia",
    "laudium",
    "eldorado park",
    "mount edgecombe",
    "la lucia",
)

PROXY_RULES: tuple[ProxyRule, ...] = (
    ProxyRule(
        "religion",
        re.compile(
            r"\b(tithe|tithing|offering|church|kerk|mosque|masjid|zakat|zakaat|sadaqah|temple|"
            r"mandir|synagogue|shul|parish|ministries|ministry|gospel|cathedral|diocese)\b",
            re.I,
        ),
        "payments to religious bodies reveal religion or belief",
    ),
    ProxyRule(
        "family_responsibility",
        re.compile(
            r"\b(maintenance|child support|child maint|creche|cr[eè]che|daycare|day care|"
            r"nursery school|school fees|aftercare|nanny|au pair)\b",
            re.I,
        ),
        "maintenance and child-care payments reveal family status or pregnancy",
    ),
    ProxyRule(
        "health_disability",
        re.compile(
            r"\b(medical aid|medical|clinic|hospital|pharmacy|dis-?chem pharm|doctor|dr\.? [a-z]+ inc|"
            r"pathcare|lancet|ampath|oncology|dialysis|physio|psychologist|psychiatr|"
            r"disability grant|sassa)\b",
            re.I,
        ),
        "health and disability spend, and social grants, reveal health or disability status",
    ),
    ProxyRule(
        "nationality",
        re.compile(
            r"\b(home affairs|dha\b|vfs global|vfs\b|embassy|consulate|permit renewal|"
            r"asylum|refugee|western union|mukuru|hello paisa|mama money|remit(?:tance)?)\b",
            re.I,
        ),
        "immigration fees and remittance corridors reveal nationality or origin",
    ),
    ProxyRule(
        "location",
        re.compile(r"\b(" + "|".join(re.escape(s) for s in SA_SUBURBS) + r")\b", re.I),
        "suburb names proxy for race and social origin; amount kept, location stripped",
        action="strip",
    ),
)


@dataclass(frozen=True)
class AuditItem:
    line_no: int
    date: str
    amount: float
    ground: str
    matched: str
    action: str
    why: str
    description: str  # PII-redacted, for the audit trail only


@dataclass
class AuditReport:
    kept: list[Transaction] = field(default_factory=list)
    excluded: list[AuditItem] = field(default_factory=list)
    stripped: list[AuditItem] = field(default_factory=list)

    @property
    def by_ground(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.excluded + self.stripped:
            counts[item.ground] = counts.get(item.ground, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> str:
        if not self.excluded and not self.stripped:
            return "no proxy indicators found; all transactions used"
        parts = [f"{g}: {n}" for g, n in self.by_ground.items()]
        return (
            f"excluded {len(self.excluded)} transaction(s), stripped location from "
            f"{len(self.stripped)} ({', '.join(parts)})"
        )

    def to_dict(self) -> dict:
        def row(i: AuditItem) -> dict:
            return {
                "line": i.line_no,
                "date": i.date,
                "amount": i.amount,
                "ground": i.ground,
                "matched": i.matched,
                "action": i.action,
                "why": i.why,
            }

        return {
            "kept": len(self.kept),
            "excluded": [row(i) for i in self.excluded],
            "stripped": [row(i) for i in self.stripped],
            "by_ground": self.by_ground,
        }


def proxy_audit(transactions: Iterable[Transaction]) -> AuditReport:
    """Remove proxy transactions and strip location tokens before feature building.

    Salary credits are never *excluded*: an employer called "Coastal Care Hospital"
    or "Ridge Medical Partners" says nothing about the applicant's health, and
    dropping the credit would zero a nurse's income. Location tokens are still
    stripped from salary lines.
    """
    report = AuditReport()
    for t in transactions:
        desc = t.description
        dropped = False
        is_salary_credit = t.category == "salary" and t.is_credit
        for rule in PROXY_RULES:
            if rule.action == "exclude" and is_salary_credit:
                continue
            m = rule.pattern.search(desc)
            if not m:
                continue
            item = AuditItem(
                line_no=t.line_no,
                date=t.date.isoformat(),
                amount=t.amount,
                ground=rule.ground,
                matched=m.group(0),
                action=rule.action,
                why=rule.why,
                description=redact(t.description),
            )
            if rule.action == "exclude":
                report.excluded.append(item)
                dropped = True
                break
            report.stripped.append(item)
            desc = rule.pattern.sub("[LOCATION]", desc)
        if dropped:
            continue
        report.kept.append(t if desc == t.description else t.with_description(desc))
    return report


# --------------------------------------------------------------------------- disparate impact
@dataclass(frozen=True)
class DisparateImpactResult:
    rates: dict[str, float]
    counts: dict[str, int]
    ratio: float
    passes: bool
    pass_score: int
    threshold: float

    def to_dict(self) -> dict:
        return {
            "rates": self.rates,
            "counts": self.counts,
            "ratio": self.ratio,
            "passes": self.passes,
            "pass_score": self.pass_score,
            "threshold": self.threshold,
        }


def disparate_impact_test(
    scores: Sequence[float],
    groups: Sequence[str],
    *,
    pass_score: int = 60,
    threshold: float = 0.8,
) -> DisparateImpactResult:
    """Four-fifths rule: min group selection rate / max group selection rate >= 0.8.

    ``scores`` and ``groups`` are aligned. A candidate is "selected" when the score is
    at least ``pass_score`` (the *adequate* band boundary by default).
    """
    if len(scores) != len(groups):
        raise ValueError("scores and groups must be the same length")
    if not scores:
        raise ValueError("no scores supplied")
    totals: dict[str, int] = {}
    passed: dict[str, int] = {}
    for s, g in zip(scores, groups, strict=True):
        key = str(g)
        totals[key] = totals.get(key, 0) + 1
        passed[key] = passed.get(key, 0) + (1 if s >= pass_score else 0)
    rates = {g: passed[g] / totals[g] for g in totals}
    hi = max(rates.values())
    lo = min(rates.values())
    ratio = 1.0 if hi == 0 else lo / hi
    return DisparateImpactResult(
        rates={g: round(r, 4) for g, r in sorted(rates.items())},
        counts=dict(sorted(totals.items())),
        ratio=round(ratio, 4),
        passes=ratio >= threshold,
        pass_score=pass_score,
        threshold=threshold,
    )
