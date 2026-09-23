"""Shared fixtures: hand-built statements in every layout and the demo personas."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from affordability.parse.statement import Transaction, classify, extract_reference

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "demo" / "data"


@pytest.fixture(scope="session")
def manifest() -> list[dict]:
    return json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def demo_dir() -> Path:
    return DATA


def _rows(salary: float = 20_000.0, months: int = 4, opening: float = 20_000.0):
    """A boring, regular account: one salary, one rent, fixed costs, some spend."""
    rows = []
    balance = opening
    year = 2026
    for i in range(months):
        month = 3 + i
        for day, desc, amount in (
            (1, "MONTHLY ACCOUNT FEE", -109.0),
            (1, "RENT PAYMENT FLAT 2", -5_000.0),
            (2, "DEBIT ORDER VODACOM CONTRACT", -1_000.0),
            (3, "CAPFIN LOAN INSTALMENT", -2_000.0),
            (10, "CHECKERS HYPER", -1_812.37),
            (12, "FNB ATM CASH WITHDRAWAL", -500.0),
            (15, "BETWAY DEPOSIT", -487.25),
            (18, "PICK N PAY", -1_187.63),
            (25, f"ACME TRADING SALARY REF 0120{year}{month:02d}", salary),
        ):
            balance = round(balance + amount, 2)
            rows.append((date(year, month, day), desc, amount, balance))
    return rows


def make_transactions(salary: float = 20_000.0, months: int = 4, opening: float = 20_000.0):
    return [
        Transaction(
            date=d,
            description=desc,
            amount=amount,
            balance=balance,
            category=classify(desc),
            txn_id=extract_reference(desc),
            line_no=i + 2,
        )
        for i, (d, desc, amount, balance) in enumerate(_rows(salary, months, opening))
    ]


def _plain(v: float) -> str:
    return f"{v:.2f}"


def _sa(v: float, prefix: str = "") -> str:
    s = f"{abs(v):,.2f}".replace(",", " ").replace(".", ",")
    return f"{'-' if v < 0 else ''}{prefix}{s}"


def render(layout: str, rows) -> str:
    if layout == "fnb":
        lines = ["FNB Cheque Account Statement", "Account Number: 62012345678", "",
                 "Date,Description,Amount,Balance"]
        lines += [f"{d:%d/%m/%Y},{desc},{_plain(a)},{_plain(b)}" for d, desc, a, b in rows]
    elif layout == "standard":
        lines = ["Standard Bank - Transaction History", "", "Date,Details,Debit,Credit,Balance"]
        for d, desc, a, b in rows:
            debit = f"{abs(a):,.2f}" if a < 0 else ""
            credit = f"{a:,.2f}" if a > 0 else ""
            bal = ("-" if b < 0 else "") + f"{abs(b):,.2f}"
            cells = [f"{d:%d/%m/%Y}", desc, debit, credit, bal]
            lines.append(",".join(f'"{c}"' if "," in c else c for c in cells))
    elif layout == "capitec":
        lines = ["Capitec Bank Statement", "", "Date | Description | Amount | Balance"]
        lines += [f"{d:%d/%m/%Y} | {desc} | {_sa(a, 'R')} | {_sa(b, 'R')}" for d, desc, a, b in rows]
    elif layout == "absa":
        lines = ["Absa Bank Limited - Statement of Account", "",
                 "Transaction Date,Description,Amount (R),Balance (R)"]
        lines += [f'{d:%d/%m/%Y},{desc},"{_sa(a)}","{_sa(b)}"' for d, desc, a, b in rows]
    else:
        raise ValueError(layout)
    return "\n".join(lines) + "\n"


@pytest.fixture
def statement_rows():
    return _rows()


@pytest.fixture
def fnb_text(statement_rows) -> str:
    return render("fnb", statement_rows)


@pytest.fixture
def transactions():
    return make_transactions()


PAYSLIP = """ACME TRADING
PAYSLIP
Employer: Acme Trading
Employee No: EMP-0001
ID Number: 9205244136087
Pay Date: 25/06/2026
Period: 01/06/2026 - 30/06/2026

Gross Pay: R 26,500.00

Deductions:
  PAYE: R 4,323.00
  UIF: R 177.00
  Pension Fund: R 2,000.00
Total Deductions: R 6,500.00

Net Pay: R 20,000.00
"""


@pytest.fixture
def payslip_text() -> str:
    return PAYSLIP
