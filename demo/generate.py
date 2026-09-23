"""Deterministic generator of synthetic SA bank statements and payslips.

Twelve personas span domestic worker to doctor, four bank export layouts, three
to six months of history, labelled per-transaction categories and a labelled
12-month outcome. Two tampered variants are derived from clean personas: one
with an edited salary line (breaks the running balance and the payslip match),
one with a duplicated salary line (same bank reference twice).

Everything is fictional. Employers, references and ID numbers are invented;
the ID numbers are Luhn-valid only so the redaction path can be exercised.

    python demo/generate.py            # writes demo/data/
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from affordability.privacy import luhn_valid  # noqa: E402

DATA_DIR = ROOT / "demo" / "data"
END_YEAR, END_MONTH = 2026, 8
SEED = 2026


@dataclass(frozen=True)
class Persona:
    id: str
    index: int
    bank: str  # fnb | standard | capitec | absa
    employer: str
    gross: float
    net: float
    pay_day: int
    months: int
    opening: float
    current_rent: float
    rent_desc: str
    proposed_rent: float
    outcome: bool  # True = 12 months without a missed payment
    debit_orders: tuple[tuple[str, float], ...] = ()
    loans: tuple[tuple[str, float], ...] = ()
    retail_share: float = 0.30
    cash_share: float = 0.08
    gambling_share: float = 0.0
    savings: float = 0.0
    proxies: tuple[tuple[str, float, str], ...] = ()  # (description, amount, label)
    variable_income: float = 0.0
    pay_jitter: int = 0
    pension: bool = True
    note: str = ""


PERSONAS: tuple[Persona, ...] = (
    Persona(
        "p01_domestic_worker", 1, "capitec", "Bayview Household Services", 6800, 6200, 25, 6,
        5300, 2500, "RENT PAYMENT TO MRS D", 2800, False,
        debit_orders=(("DEBIT ORDER 1LIFE FUNERAL POLICY", 89.00),),
        retail_share=0.38, cash_share=0.16, pension=False,
        note="rent at 45% of income",
    ),
    Persona(
        "p02_security_guard", 2, "fnb", "Umkhonto Protection Services", 10500, 8900, 30, 6,
        7600, 3200, "RENT LANDLORD MTHEMBU", 3500, True,
        debit_orders=(("DEBIT ORDER MTN CONTRACT", 299.00), ("DEBIT ORDER OUTSURANCE", 410.00)),
        loans=(("CAPFIN LOAN INSTALMENT", 780.00),),
        retail_share=0.30, cash_share=0.12, pension=False,
    ),
    Persona(
        "p03_cashier_gambling", 3, "standard", "Coastal Retail Holdings", 11900, 9800, 25, 4,
        4300, 3000, "RENT PAYMENT FLAT 4B", 3800, False,
        debit_orders=(("DEBIT ORDER CELL C", 249.00),),
        loans=(("WONGA LOAN REPAYMENT", 650.00),),
        retail_share=0.28, cash_share=0.10, gambling_share=0.18, pension=False,
        note="gambling and payday loans",
    ),
    Persona(
        "p04_admin_clerk", 4, "absa", "Dolphin Coast Logistics", 15200, 12400, 25, 6,
        10500, 4500, "RENT SEAVIEW FLATS", 5200, True,
        debit_orders=(
            ("DEBIT ORDER TELKOM FIBRE", 599.00),
            ("DEBIT ORDER SANLAM POLICY", 320.00),
            ("DEBIT ORDER DISCOVERY MEDICAL AID", 1650.00),
        ),
        retail_share=0.22, cash_share=0.08,
        proxies=(("TITHE - GRACE FAMILY CHURCH", 1240.00, "other"),),
        note="marginal band but paid every month",
    ),
    Persona(
        "p05_call_centre_agent", 5, "fnb", "Nexa Contact Solutions", 18500, 14800, 27, 3,
        9000, 5500, "RENTAL PAYMENT BEREA APARTMENTS", 6500, False,
        debit_orders=(
            ("DEBIT ORDER VIRGIN ACTIVE", 649.00),
            ("DEBIT ORDER VODACOM CONTRACT", 899.00),
            ("DEBIT ORDER MIWAY INSURANCE", 1150.00),
        ),
        loans=(("AFRICAN BANK PERSONAL LOAN", 2350.00),),
        retail_share=0.26, cash_share=0.06, pay_jitter=2,
        note="only three months of history",
    ),
    Persona(
        "p06_nurse", 6, "capitec", "Coastal Care Hospital Group", 24000, 18500, 25, 6,
        16000, 6500, "RENT PAYMENT GLENWOOD", 7000, True,
        debit_orders=(
            ("DEBIT ORDER OUTSURANCE POLICY", 649.00),
            ("DEBIT ORDER VODACOM CONTRACT", 499.00),
            ("DEBIT ORDER DISCOVERY MEDICAL AID", 2100.00),
        ),
        loans=(("PERSONAL LOAN REPAYMENT AFRICAN BANK", 1450.00),),
        retail_share=0.20, cash_share=0.05,
        proxies=(("MAINTENANCE PAYMENT S DLAMINI", 1500.00, "transfer"),),
    ),
    Persona(
        "p07_teacher", 7, "standard", "Provincial Education Department", 29500, 22300, 25, 6,
        19000, 7500, "RENT PAYMENT WESTVILLE", 7800, True,
        debit_orders=(
            ("DEBIT ORDER OLD MUTUAL POLICY", 780.00),
            ("DEBIT ORDER MTN CONTRACT", 649.00),
            ("SCHOOL FEES ST JOHNS PREP", 3200.00),
        ),
        loans=(("WESBANK VEHICLE FINANCE", 3890.00),),
        retail_share=0.15, cash_share=0.05, savings=1000.00,
    ),
    Persona(
        "p08_sales_rep_commission", 8, "absa", "Kingfisher Pharma Distribution", 21000, 16000,
        25, 6, 12000, 7000, "RENT MUSGRAVE COURT", 8500, False,
        debit_orders=(
            ("DEBIT ORDER SANTAM CAR INSURANCE", 1290.00),
            ("DEBIT ORDER RAIN 5G", 599.00),
        ),
        loans=(("MFC VEHICLE FINANCE", 4200.00),),
        retail_share=0.18, cash_share=0.06, variable_income=0.32,
        note="commission income swings month to month",
    ),
    Persona(
        "p09_it_support", 9, "fnb", "Harbour Systems (Pty) Ltd", 37000, 28000, 25, 6,
        24000, 8500, "RENTAL PAYMENT UMHLANGA RIDGE", 9500, True,
        debit_orders=(
            ("DEBIT ORDER AFRIHOST FIBRE", 899.00),
            ("DEBIT ORDER DISCOVERY LIFE POLICY", 1100.00),
            ("PLANET FITNESS DEBIT ORDER", 429.00),
        ),
        loans=(("CREDIT CARD PAYMENT", 3500.00),),
        retail_share=0.22, cash_share=0.04, savings=2000.00,
        proxies=(("WESTERN UNION REMITTANCE", 2500.00, "transfer"),),
    ),
    Persona(
        "p10_accountant", 10, "standard", "Marlin Chartered Accountants", 47000, 35000, 25, 6,
        30000, 11000, "RENT PAYMENT LA LUCIA", 12000, True,
        debit_orders=(
            ("DEBIT ORDER OUTSURANCE", 1450.00),
            ("DEBIT ORDER MOMENTUM RETIREMENT ANNUITY", 2500.00),
            ("DEBIT ORDER DSTV PREMIUM", 929.00),
        ),
        loans=(("BMW FINANCE INSTALMENT", 6900.00),),
        retail_share=0.15, cash_share=0.03, savings=4000.00,
    ),
    Persona(
        "p11_engineer_overdraft", 11, "absa", "Trident Consulting Engineers", 62000, 45000,
        25, 6, -8000, 14000, "RENT PAYMENT HILLCREST ESTATE", 16000, False,
        debit_orders=(
            ("DEBIT ORDER LIBERTY POLICY", 1800.00),
            ("DEBIT ORDER VODACOM", 1299.00),
            ("DEBIT ORDER VIRGIN ACTIVE", 999.00),
        ),
        loans=(
            ("WESBANK VEHICLE FINANCE", 9800.00),
            ("PERSONAL LOAN REPAYMENT NEDBANK", 4200.00),
        ),
        retail_share=0.28, cash_share=0.05,
        note="high earner living in overdraft",
    ),
    Persona(
        "p12_doctor", 12, "capitec", "Ridge Medical Partners", 125000, 85000, 25, 6,
        60000, 22000, "RENT PAYMENT MOUNT EDGECOMBE", 25000, True,
        debit_orders=(
            ("DEBIT ORDER DISCOVERY MEDICAL AID", 4800.00),
            ("DEBIT ORDER SANTAM", 2100.00),
            ("DEBIT ORDER ALLAN GRAY UNIT TRUST", 8000.00),
        ),
        loans=(("AUDI FINANCE INSTALMENT", 12500.00),),
        retail_share=0.20, cash_share=0.02,
    ),
)

# (description, label). Some are deliberately hard for a keyword classifier.
MERCHANTS: tuple[tuple[str, str], ...] = (
    ("CHECKERS HYPER", "retail"),
    ("PICK N PAY", "retail"),
    ("WOOLWORTHS FOOD", "retail"),
    ("SPAR SUPERSPAR", "retail"),
    ("SHOPRITE", "retail"),
    ("BOXER SUPERSTORES", "retail"),
    ("CLICKS", "retail"),
    ("DIS-CHEM", "retail"),
    ("TAKEALOT.COM", "retail"),
    ("UBER TRIP", "retail"),
    ("BOLT TRIP", "retail"),
    ("ENGEN FUEL", "retail"),
    ("SHELL ULTRA CITY", "retail"),
    ("KFC", "retail"),
    ("NANDOS", "retail"),
    ("MR PRICE", "retail"),
    ("PEP STORES", "retail"),
    ("GAME STORES", "retail"),
    ("TOTAL SPORTS", "retail"),
    ("NETFLIX.COM", "retail"),
    ("POS PURCHASE BETHLEHEM SPAR", "retail"),
    ("CARD PURCHASE ROMANS PIZZA", "retail"),
    ("VODACOM PREPAID AIRTIME", "debit_order"),
    ("STEERS DRIVE THRU", "retail"),
    ("MAKRO", "retail"),
    ("COTTON ON", "retail"),
    ("SPUR STEAK RANCH", "retail"),
    ("AVIS RENT-A-CAR", "retail"),
)
SUBURB_MERCHANTS: tuple[tuple[str, str], ...] = (
    ("CHECKERS UMHLANGA", "retail"),
    ("PNP UMLAZI MEGA CITY", "retail"),
    ("WOOLWORTHS SANDTON", "retail"),
    ("SPAR KLOOF", "retail"),
)
GAMBLING: tuple[str, ...] = ("BETWAY DEPOSIT", "HOLLYWOODBETS", "SUPABETS ONLINE", "LOTTO ONLINE")
FEES = {"fnb": 109.00, "standard": 119.00, "capitec": 7.50, "absa": 99.00}
ATM = {
    "fnb": "FNB ATM CASH WITHDRAWAL",
    "standard": "ATM CASH WITHDRAWAL",
    "capitec": "CASH WITHDRAWAL ATM",
    "absa": "ABSA ATM CASH WITHDRAWAL",
}


@dataclass(frozen=True)
class Row:
    date: date
    description: str
    amount: float
    label: str
    balance: float = 0.0
    order: int = 0


@dataclass
class Generated:
    persona: Persona
    rows: list[Row]
    statement: str
    payslip: str
    labels: list[tuple[int, str]] = field(default_factory=list)
    id_number: str = ""


# --------------------------------------------------------------------------- helpers
def _months(count: int) -> list[tuple[int, int]]:
    out = []
    y, m = END_YEAR, END_MONTH
    for _ in range(count):
        out.append((y, m))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return list(reversed(out))


def _days_in(y: int, m: int) -> int:
    nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return (nxt - date(y, m, 1)).days


def _weekday_shift(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def synthetic_sa_id(rng: np.random.Generator) -> str:
    """Luhn-valid 13-digit ID with a plausible date prefix. Fictional."""
    yy = int(rng.integers(70, 99))
    mm = int(rng.integers(1, 13))
    dd = int(rng.integers(1, 29))
    body = f"{yy:02d}{mm:02d}{dd:02d}{int(rng.integers(0, 9999)):04d}08"
    for check in range(10):
        if luhn_valid(body + str(check)):
            return body + str(check)
    raise RuntimeError("unreachable")


def _scaled(rng: np.random.Generator, n: int, total: float, sigma: float = 0.8) -> list[float]:
    weights = rng.lognormal(0.0, sigma, n)
    amounts = weights / weights.sum() * total
    return [max(12.0, round(float(a) + float(rng.integers(0, 100)) / 100, 2)) for a in amounts]


def _ref(persona: Persona, y: int, m: int) -> str:
    return f"REF {persona.index:02d}{y}{m:02d}"


# --------------------------------------------------------------------------- generation
def generate_rows(persona: Persona, seed: int = SEED) -> list[Row]:
    rng = np.random.default_rng(seed * 1000 + persona.index)
    rows: list[Row] = []
    order = 0

    def add(d: date, desc: str, amount: float, label: str) -> None:
        nonlocal order
        rows.append(Row(d, desc, round(amount, 2), label, order=order))
        order += 1

    for y, m in _months(persona.months):
        dim = _days_in(y, m)
        add(date(y, m, 1), "MONTHLY ACCOUNT FEE", -FEES[persona.bank], "fees")
        if persona.current_rent:
            add(date(y, m, 1), persona.rent_desc, -persona.current_rent, "rent")
        for i, (desc, amt) in enumerate(persona.debit_orders):
            add(date(y, m, 2 + i % 4), desc, -amt, "debit_order")
        for i, (desc, amt) in enumerate(persona.loans):
            add(date(y, m, 3 + i), desc, -amt, "loan")
        for desc, amt, label in persona.proxies:
            add(date(y, m, 5), desc, -amt, label)
        if persona.savings:
            add(date(y, m, 26), "PAYMENT TO SAVINGS POCKET", -persona.savings, "transfer")

        pay = date(y, m, min(persona.pay_day, dim))
        if persona.pay_jitter:
            pay += timedelta(days=int(rng.integers(-persona.pay_jitter, persona.pay_jitter + 1)))
            pay = min(pay, date(y, m, dim))
        pay = _weekday_shift(pay)
        salary = persona.net
        if persona.variable_income:
            # Commission months swing around the base net; clip to a believable
            # 0.6x-1.5x so no single month looks like a typo. The payslip for the
            # pay month is rendered from this figure, and its gross scales with it
            # (see render_payslip), so a legitimate swing never trips the 2 %
            # salary-vs-payslip tamper check.
            swing = float(np.clip(1 + rng.normal(0, persona.variable_income), 0.6, 1.5))
            salary = round(persona.net * swing, 2)
        add(pay, f"{persona.employer.upper()} SALARY {_ref(persona, y, m)}", salary, "salary")

        n_retail = int(rng.integers(14, 28))
        for amt in _scaled(rng, n_retail, persona.net * persona.retail_share):
            pool = MERCHANTS + (SUBURB_MERCHANTS if persona.index % 3 == 0 else ())
            desc, label = pool[int(rng.integers(0, len(pool)))]
            add(date(y, m, int(rng.integers(1, dim + 1))), desc, -amt, label)
        n_cash = int(rng.integers(1, 4))
        for amt in _scaled(rng, n_cash, persona.net * persona.cash_share, sigma=0.3):
            rounded = max(50.0, round(amt / 50) * 50)
            add(date(y, m, int(rng.integers(1, dim + 1))), ATM[persona.bank], -rounded, "cash")
        if persona.gambling_share:
            n_bets = int(rng.integers(4, 11))
            for amt in _scaled(rng, n_bets, persona.net * persona.gambling_share):
                desc = GAMBLING[int(rng.integers(0, len(GAMBLING)))]
                add(date(y, m, int(rng.integers(1, dim + 1))), desc, -amt, "gambling")
        if rng.random() < 0.5:
            add(date(y, m, int(rng.integers(6, 20))), "PAYMENT FROM T MOKOENA", 350.00, "transfer")

    rows.sort(key=lambda r: (r.date, r.order))
    return _with_balances(rows, persona.opening)


def _with_balances(rows: list[Row], opening: float) -> list[Row]:
    out: list[Row] = []
    bal = opening
    for r in rows:
        bal = round(bal + r.amount, 2)
        out.append(replace(r, balance=bal))
    return out


# --------------------------------------------------------------------------- rendering
def _plain(v: float) -> str:
    return f"{v:.2f}"


def _comma(v: float) -> str:
    return f"{abs(v):,.2f}"


def _sa(v: float, prefix: str = "") -> str:
    s = f"{abs(v):,.2f}".replace(",", " ").replace(".", ",")
    return f"{'-' if v < 0 else ''}{prefix}{s}"


def render_statement(rows: list[Row], bank: str, account: str) -> tuple[str, list[tuple[int, str]]]:
    """Return statement text and ``(line_no, label)`` pairs for every transaction line."""
    lines: list[str] = []
    labels: list[tuple[int, str]] = []
    if bank == "fnb":
        lines += ["FNB Cheque Account Statement", f"Account Number: {account}", ""]
        lines.append("Date,Description,Amount,Balance")
        for r in rows:
            labels.append((len(lines) + 1, r.label))
            lines.append(f"{r.date:%d/%m/%Y},{r.description},{_plain(r.amount)},{_plain(r.balance)}")
    elif bank == "standard":
        lines += ["Standard Bank - Transaction History", f"Account: {account}", ""]
        lines.append("Date,Details,Debit,Credit,Balance")
        for r in rows:
            debit = _comma(r.amount) if r.amount < 0 else ""
            credit = _comma(r.amount) if r.amount > 0 else ""
            bal = ("-" if r.balance < 0 else "") + _comma(r.balance)
            cells = [f"{r.date:%d/%m/%Y}", r.description, debit, credit, bal]
            labels.append((len(lines) + 1, r.label))
            lines.append(",".join(f'"{c}"' if "," in c else c for c in cells))
    elif bank == "capitec":
        lines += ["Capitec Bank Statement", f"Account No: {account}", ""]
        lines.append("Date | Description | Amount | Balance")
        for r in rows:
            labels.append((len(lines) + 1, r.label))
            lines.append(
                f"{r.date:%d/%m/%Y} | {r.description} | {_sa(r.amount, 'R')} | {_sa(r.balance, 'R')}"
            )
    elif bank == "absa":
        lines += ["Absa Bank Limited - Statement of Account", f"Account number {account}", ""]
        lines.append("Transaction Date,Description,Amount (R),Balance (R)")
        for r in rows:
            labels.append((len(lines) + 1, r.label))
            lines.append(f'{r.date:%d/%m/%Y},{r.description},"{_sa(r.amount)}","{_sa(r.balance)}"')
    else:
        raise ValueError(f"unknown bank layout {bank!r}")
    return "\n".join(lines) + "\n", labels


def render_payslip(persona: Persona, pay_date: date, net: float, id_number: str) -> str:
    # Gross scales with the month's net so commission months stay internally consistent.
    gross = round(persona.gross * (net / persona.net), 2)
    uif = round(min(gross * 0.01, 177.12), 2)
    pension = round(gross * 0.075, 2) if persona.pension else 0.0
    paye = round(gross - net - uif - pension, 2)
    if paye < 0:
        raise ValueError(f"{persona.id}: gross too low for net")
    y, m = pay_date.year, pay_date.month
    lines = [
        persona.employer.upper(),
        "PAYSLIP",
        f"Employer: {persona.employer}",
        f"Employee No: EMP-{persona.index:04d}",
        f"ID Number: {id_number}",
        f"Pay Date: {pay_date:%d/%m/%Y}",
        f"Period: 01/{m:02d}/{y} - {_days_in(y, m):02d}/{m:02d}/{y}",
        "",
        f"Gross Pay: R {gross:,.2f}",
        "",
        "Deductions:",
        f"  PAYE: R {paye:,.2f}",
        f"  UIF: R {uif:,.2f}",
    ]
    if pension:
        lines.append(f"  Pension Fund: R {pension:,.2f}")
    lines += [
        f"Total Deductions: R {paye + uif + pension:,.2f}",
        "",
        f"Net Pay: R {net:,.2f}",
        "",
    ]
    return "\n".join(lines)


def generate_persona(persona: Persona, seed: int = SEED) -> Generated:
    rng = np.random.default_rng(seed * 7919 + persona.index)
    rows = generate_rows(persona, seed)
    account = f"{persona.index:02d}{int(rng.integers(10**8, 10**9))}"
    statement, labels = render_statement(rows, persona.bank, account)
    salaries = [r for r in rows if r.label == "salary"]
    last = salaries[-1]
    id_number = synthetic_sa_id(rng)
    payslip = render_payslip(persona, last.date, last.amount, id_number)
    return Generated(persona, rows, statement, payslip, labels, id_number)


# --------------------------------------------------------------------------- tampering
def tamper_edit_salary(gen: Generated, bump: float = 4000.0) -> Generated:
    """Raise every salary amount but leave balances untouched: reconciliation breaks."""
    rows = [
        replace(r, amount=round(r.amount + bump, 2)) if r.label == "salary" else r
        for r in gen.rows
    ]
    statement, labels = render_statement(rows, gen.persona.bank, "00000000000")
    return Generated(gen.persona, rows, statement, gen.payslip, labels, gen.id_number)


def tamper_duplicate_salary(gen: Generated) -> Generated:
    """Insert a second copy of one salary credit (same reference) and re-chain balances."""
    salaries = [i for i, r in enumerate(gen.rows) if r.label == "salary"]
    idx = salaries[1] if len(salaries) > 1 else salaries[0]
    src = gen.rows[idx]
    dup = replace(src, date=src.date + timedelta(days=1), order=src.order + 1000)
    rows = list(gen.rows)
    rows.insert(idx + 1, dup)
    rows.sort(key=lambda r: (r.date, r.order))
    rows = _with_balances(rows, gen.persona.opening)
    statement, labels = render_statement(rows, gen.persona.bank, "00000000000")
    return Generated(gen.persona, rows, statement, gen.payslip, labels, gen.id_number)


TAMPERED: tuple[tuple[str, str, str], ...] = (
    ("t01_edited_salary", "p05_call_centre_agent", "edited_salary"),
    ("t02_duplicate_salary", "p07_teacher", "duplicate_salary"),
)


# --------------------------------------------------------------------------- output
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _write_labels(path: Path, labels: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["line_no", "category"])
        w.writerows(labels)


def write_all(out_dir: Path = DATA_DIR, seed: int = SEED) -> list[dict]:
    manifest: list[dict] = []
    generated: dict[str, Generated] = {}
    for persona in PERSONAS:
        gen = generate_persona(persona, seed)
        generated[persona.id] = gen
        manifest.append(_emit(out_dir, persona.id, gen, tampered=None, source=None))
    for tid, source, kind in TAMPERED:
        base = generated[source]
        gen = tamper_edit_salary(base) if kind == "edited_salary" else tamper_duplicate_salary(base)
        manifest.append(_emit(out_dir, tid, gen, tampered=kind, source=source))
    _write(out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def _emit(out_dir: Path, doc_id: str, gen: Generated, *, tampered: str | None, source: str | None) -> dict:
    ext = "txt" if gen.persona.bank == "capitec" else "csv"
    folder = out_dir / doc_id
    _write(folder / f"statement.{ext}", gen.statement)
    _write(folder / "payslip.txt", gen.payslip)
    _write_labels(folder / "labels.csv", gen.labels)
    return {
        "id": doc_id,
        "persona": gen.persona.id,
        "bank": gen.persona.bank,
        "statement": f"{doc_id}/statement.{ext}",
        "payslip": f"{doc_id}/payslip.txt",
        "labels": f"{doc_id}/labels.csv",
        "months": gen.persona.months,
        "proposed_rent": gen.persona.proposed_rent,
        "outcome_no_default_12m": gen.persona.outcome,
        "tampered": tampered is not None,
        "tamper_kind": tampered,
        "source": source,
        "note": gen.persona.note,
    }


def main() -> None:
    manifest = write_all()
    print(f"wrote {len(manifest)} documents to {DATA_DIR}")
    for entry in manifest:
        flag = f"  TAMPERED ({entry['tamper_kind']})" if entry["tampered"] else ""
        print(f"  {entry['id']:<28} {entry['bank']:<9} {entry['months']}m  rent R{entry['proposed_rent']:>8,.0f}{flag}")


if __name__ == "__main__":
    main()
