"""Payslip text extraction: employer, pay date, gross, net and itemised deductions.

Works on the plain text you get from a PDF payslip after text extraction. The
parser is line-oriented and label-driven (``Net Pay: R 18,500.00``), so it does
not care about column alignment or the payroll vendor's layout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from affordability.parse.statement import parse_amount, parse_date


class PayslipParseError(ValueError):
    """Raised when a net figure cannot be found (the one field scoring needs)."""


@dataclass(frozen=True)
class Payslip:
    employer: str | None
    pay_date: date | None
    gross: float | None
    net: float
    deductions: dict[str, float] = field(default_factory=dict)

    @property
    def total_deductions(self) -> float:
        return round(sum(self.deductions.values()), 2)


_MONEY = r"R?\s*\(?-?[\d][\d\s,.']*\)?"
_EMPLOYER_RE = re.compile(r"^\s*(?:employer|company)(?:\s*name)?\s*[:\-]\s*(.+?)\s*$", re.I | re.M)
_GROSS_RE = re.compile(
    rf"^\s*(?:total\s+)?gross\s+(?:pay|salary|earnings|income|remuneration)\s*[:\-]?\s*({_MONEY})\s*$",
    re.I | re.M,
)
_NET_RE = re.compile(
    rf"^\s*(?:nett?\s+(?:pay|salary|income|amount)|take[\s-]?home(?:\s+pay)?|amount\s+paid)"
    rf"\s*[:\-]?\s*({_MONEY})\s*$",
    re.I | re.M,
)
_PAYDATE_RE = re.compile(
    r"^\s*(?:pay(?:ment)?\s+date|date\s+paid|paid\s+on)\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4}"
    r"|[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})",
    re.I | re.M,
)
_DEDUCTION_NAMES = (
    "paye",
    "tax",
    "uif",
    "pension",
    "provident",
    "medical aid",
    "medical",
    "garnishee",
    "union",
    "loan",
    "advance",
    "sdl",
    "group life",
    "funeral",
)
_DEDUCTION_RE = re.compile(
    rf"^\s*((?:{'|'.join(re.escape(n) for n in _DEDUCTION_NAMES)})[a-z /&()-]*?)\s*[:\-]?\s*({_MONEY})\s*$",
    re.I | re.M,
)
_TOTAL_DED_RE = re.compile(rf"^\s*total\s+deductions?\s*[:\-]?\s*({_MONEY})\s*$", re.I | re.M)


def _money(m: str | None) -> float | None:
    if m is None:
        return None
    try:
        return parse_amount(m)
    except ValueError:
        return None


def parse_payslip(text: str) -> Payslip:
    """Extract the fields scoring and the tamper check need."""
    employer_m = _EMPLOYER_RE.search(text)
    employer = employer_m.group(1).strip() if employer_m else None

    pay_date: date | None = None
    pd_m = _PAYDATE_RE.search(text)
    if pd_m:
        try:
            pay_date = parse_date(pd_m.group(1))
        except ValueError:
            pay_date = None

    gross = _money(_GROSS_RE.search(text).group(1)) if _GROSS_RE.search(text) else None
    net_m = _NET_RE.search(text)
    net = _money(net_m.group(1)) if net_m else None

    deductions: dict[str, float] = {}
    for name, amount in _DEDUCTION_RE.findall(text):
        value = _money(amount)
        if value is None:
            continue
        key = re.sub(r"\s+", " ", name.strip().lower())
        if key.startswith("total") or re.search(r"ref|number|\bno\b", key):
            continue  # "Tax Reference No: ..." is an identifier, not a deduction
        deductions[key] = abs(value)

    if net is None and gross is not None:
        total_m = _TOTAL_DED_RE.search(text)
        total = _money(total_m.group(1)) if total_m else None
        if total is None and deductions:
            total = sum(deductions.values())
        if total is not None:
            net = round(gross - abs(total), 2)
    if net is None:
        raise PayslipParseError("no net pay figure found on payslip")
    return Payslip(employer=employer, pay_date=pay_date, gross=gross, net=abs(net), deductions=deductions)


def load_payslip(path: str | Path) -> Payslip:
    return parse_payslip(Path(path).read_text(encoding="utf-8-sig"))
