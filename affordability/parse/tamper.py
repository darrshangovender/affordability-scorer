"""Document-consistency checks that catch naive edits to statements and payslips.

Five checks, each producing zero or more :class:`TamperFlag`:

1. **Running balance** — ``balance[i-1] + amount[i]`` must equal ``balance[i]``.
   Editing a single amount in a real export breaks every balance after it.
2. **Payslip reconciliation** — the salary credit in the payslip's pay month must
   match the payslip net within 2 % (and name the employer).
3. **Date gaps** — an unexplained hole in activity suggests deleted pages.
4. **Duplicate references** — the same bank reference twice, or the same credit
   twice on one day, is a copy-pasted line.
5. **Round-number density** — real spend is rarely a multiple of R100; fabricated
   statements usually are.

The result is a :class:`TamperReport` with a coarse ``risk`` of ``low``/``medium``/``high``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from itertools import pairwise

from affordability.parse.payslip import Payslip
from affordability.parse.statement import Transaction

Severity = str  # "low" | "medium" | "high"


@dataclass(frozen=True)
class TamperFlag:
    check: str
    severity: Severity
    message: str
    line_no: int | None = None


@dataclass
class TamperReport:
    flags: list[TamperFlag] = field(default_factory=list)
    risk: str = "low"

    @property
    def ok(self) -> bool:
        return self.risk == "low"

    def by_check(self, check: str) -> list[TamperFlag]:
        return [f for f in self.flags if f.check == check]

    def summary(self) -> str:
        if not self.flags:
            return "no consistency issues found"
        counts = Counter(f.check for f in self.flags)
        parts = ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))
        return f"risk={self.risk}: {parts}"

    def to_dict(self) -> dict:
        return {
            "risk": self.risk,
            "flags": [
                {"check": f.check, "severity": f.severity, "message": f.message, "line": f.line_no}
                for f in self.flags
            ],
        }


def _risk(flags: list[TamperFlag]) -> str:
    if any(f.severity == "high" for f in flags):
        return "high"
    if any(f.severity == "medium" for f in flags):
        return "medium"
    return "low"


def check_running_balance(txns: list[Transaction], tolerance: float = 0.011) -> list[TamperFlag]:
    flags: list[TamperFlag] = []
    prev = None
    for t in txns:
        if t.balance is None:
            continue
        if prev is not None:
            expected = round(prev.balance + t.amount, 2)
            if abs(expected - t.balance) > tolerance:
                flags.append(
                    TamperFlag(
                        "running_balance",
                        "high",
                        f"line {t.line_no}: expected balance {expected:.2f} "
                        f"(prev {prev.balance:.2f} + {t.amount:.2f}) but statement shows "
                        f"{t.balance:.2f}",
                        t.line_no,
                    )
                )
        prev = t
    return flags


def check_payslip(
    txns: list[Transaction], payslip: Payslip | None, tolerance: float = 0.02
) -> list[TamperFlag]:
    if payslip is None:
        return []
    flags: list[TamperFlag] = []
    salaries = [t for t in txns if t.category == "salary" and t.is_credit]
    if not salaries:
        return [TamperFlag("payslip_match", "medium", "no salary credit found to compare with payslip")]

    if payslip.pay_date is not None:
        window = [t for t in salaries if abs((t.date - payslip.pay_date).days) <= 7]
    else:
        window = []
    candidates = window or salaries[-1:]
    for t in candidates:
        diff = abs(t.amount - payslip.net) / payslip.net if payslip.net else 1.0
        if diff > tolerance:
            flags.append(
                TamperFlag(
                    "payslip_match",
                    "high",
                    f"line {t.line_no}: salary credit {t.amount:.2f} differs from payslip net "
                    f"{payslip.net:.2f} by {diff:.1%} (limit {tolerance:.0%})",
                    t.line_no,
                )
            )

    if payslip.employer:
        tokens = [w for w in payslip.employer.upper().replace("(", " ").split() if len(w) >= 4]
        tokens = [w for w in tokens if w not in {"(PTY)", "PTY)", "LTD", "LIMITED", "GROUP", "THE"}]
        if tokens and not any(tok in t.description.upper() for t in salaries for tok in tokens):
            flags.append(
                TamperFlag(
                    "payslip_match",
                    "medium",
                    f"no salary credit names the payslip employer ({payslip.employer})",
                )
            )
    return flags


def check_date_gaps(txns: list[Transaction], max_gap_days: int = 40) -> list[TamperFlag]:
    flags: list[TamperFlag] = []
    for a, b in pairwise(txns):
        gap = (b.date - a.date).days
        if gap > max_gap_days:
            flags.append(
                TamperFlag(
                    "date_gap",
                    "medium",
                    f"{gap} days with no activity between {a.date} and {b.date}",
                    b.line_no,
                )
            )
        elif gap < 0:
            flags.append(
                TamperFlag("date_gap", "high", f"dates run backwards at line {b.line_no}", b.line_no)
            )
    return flags


def check_duplicates(txns: list[Transaction]) -> list[TamperFlag]:
    flags: list[TamperFlag] = []
    seen_refs: dict[str, Transaction] = {}
    for t in txns:
        if t.txn_id:
            if t.txn_id in seen_refs:
                first = seen_refs[t.txn_id]
                flags.append(
                    TamperFlag(
                        "duplicate_id",
                        "high",
                        f"reference {t.txn_id} appears at lines {first.line_no} and {t.line_no}",
                        t.line_no,
                    )
                )
            else:
                seen_refs[t.txn_id] = t
    credits = Counter(t.fingerprint for t in txns if t.is_credit and t.amount >= 1000)
    for t in txns:
        if t.is_credit and t.amount >= 1000 and credits[t.fingerprint] > 1 and not t.txn_id:
            flags.append(
                TamperFlag(
                    "duplicate_id",
                    "medium",
                    f"credit of {t.amount:.2f} on {t.date} appears {credits[t.fingerprint]} times",
                    t.line_no,
                )
            )
            credits[t.fingerprint] = 0  # report once
    return flags


def check_round_numbers(
    txns: list[Transaction], threshold: float = 0.6, min_count: int = 20
) -> list[TamperFlag]:
    amounts = [abs(t.amount) for t in txns if abs(t.amount) >= 100]
    if len(amounts) < min_count:
        return []
    round_share = sum(1 for a in amounts if abs(a % 100) < 0.005) / len(amounts)
    if round_share >= threshold:
        return [
            TamperFlag(
                "round_numbers",
                "medium",
                f"{round_share:.0%} of amounts over R100 are exact multiples of R100 "
                f"(threshold {threshold:.0%})",
            )
        ]
    return []


def check_document(
    txns: list[Transaction],
    payslip: Payslip | None = None,
    *,
    max_gap_days: int = 40,
) -> TamperReport:
    """Run every check and grade the document."""
    flags: list[TamperFlag] = []
    flags += check_running_balance(txns)
    flags += check_payslip(txns, payslip)
    flags += check_date_gaps(txns, max_gap_days=max_gap_days)
    flags += check_duplicates(txns)
    flags += check_round_numbers(txns)
    if txns and (txns[-1].date - txns[0].date) < timedelta(days=60):
        flags.append(
            TamperFlag("coverage", "low", "statement covers less than two months of activity")
        )
    return TamperReport(flags=flags, risk=_risk(flags))
