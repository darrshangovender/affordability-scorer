"""Bank statement parsing for the common South African export formats.

Four layouts are handled without the caller naming the bank:

* ``Date,Description,Amount,Balance`` (FNB-style CSV, signed amounts)
* ``Date,Details,Debit,Credit,Balance`` (Standard Bank-style CSV, unsigned columns)
* ``Date | Description | Amount | Balance`` (Capitec-style pipe text, ``R`` prefix, comma decimals)
* ``Transaction Date,Description,Amount (R),Balance (R)`` (Absa-style CSV, space thousands)

Dates are ``dd/mm/yyyy`` (also ``dd-mm-yyyy``, ISO and ``dd Mon yyyy``). Money accepts
``R``, ``ZAR``, thousands separators (space, comma or point), comma or point decimals,
bracketed negatives, trailing ``-``, and ``Cr``/``Dr`` suffixes.

Every row becomes a :class:`Transaction` with a keyword-classified ``category``.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path

CATEGORIES: tuple[str, ...] = (
    "salary",
    "rent",
    "debit_order",
    "loan",
    "gambling",
    "transfer",
    "retail",
    "cash",
    "fees",
    "other",
)


class StatementParseError(ValueError):
    """Raised when the text does not look like a bank statement we understand."""


@dataclass(frozen=True)
class Transaction:
    """One statement line. Credits are positive, debits negative."""

    date: date
    description: str
    amount: float
    balance: float | None
    category: str = "other"
    txn_id: str | None = None
    line_no: int = 0
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_credit(self) -> bool:
        return self.amount > 0

    @property
    def fingerprint(self) -> str:
        """Stable identity for duplicate detection when the bank gives no reference."""
        raw = f"{self.date.isoformat()}|{self.description.strip().upper()}|{self.amount:.2f}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def with_description(self, description: str) -> Transaction:
        return replace(self, description=description)


# --------------------------------------------------------------------------- classification
# Order matters: the first matching rule wins. Salary is checked first so that
# "SALARY PAYMENT FROM ACME" is income, not a transfer.
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("salary", re.compile(r"\b(salary|sal\b|wages|payroll|remuneration|stipend|commission)", re.I)),
    # "RENT-A-CAR" / "CAR RENTAL" is retail, not housing.
    ("rent", re.compile(r"(?<!car )\b(rent|rental|lease)\b(?!-a-car)", re.I)),
    (
        "loan",
        re.compile(
            r"\b(loan|capfin|wonga|bayport|african bank|finchoice|vehicle finance|"
            r"wesbank|mfc\b|bond repayment|home loan|instal?lment|credit card (?:payment|repay))",
            re.I,
        ),
    ),
    (
        "gambling",
        re.compile(
            r"\b(betway|hollywood ?bets|sportingbet|supabets|gbets|world sports betting|"
            r"lotto|powerball|casino|sun international|tsogo|\bbet\b)",
            re.I,
        ),
    ),
    (
        "cash",
        re.compile(r"\b(atm|cash withdrawal|cash dep(?:osit)?|cashsend|e-?wallet|cash send)", re.I),
    ),
    (
        "debit_order",
        re.compile(
            r"\b(debit order|d/o\b|insurance|assurance|policy|premium|medical aid|vodacom|mtn\b|"
            r"telkom|cell c|rain\b|dstv|multichoice|virgin active|planet fitness|old mutual|"
            r"sanlam|discovery|momentum|liberty|outsurance|miway|santam|1life|levy|"
            r"school fees|municipal|eskom|prepaid elec|fibre|afrihost|webafrica|gym)",
            re.I,
        ),
    ),
    (
        "transfer",
        re.compile(
            r"\b(transfer|tfr|eft|payment to|payment from|pay to|immediate pay|instant pay|"
            r"internal tfr|savings|unit trust|investment|maintenance payment|remit(?:tance)?|"
            r"western union|mukuru|mama money|hello paisa)",
            re.I,
        ),
    ),
    (
        "fees",
        re.compile(r"\b(fee|fees|charge|charges|interest|service fee|admin fee)\b", re.I),
    ),
    (
        "retail",
        re.compile(
            r"\b(checkers|pick n pay|pnp|woolworths|woolies|spar|shoprite|boxer|clicks|dis-?chem|"
            r"takealot|uber|bolt|engen|shell|sasol|caltex|bp\b|total\b|mcdonald|kfc|nando|"
            r"steers|wimpy|debonairs|mr price|pep\b|ackermans|edgars|game\b|makro|builders|"
            r"food lover|purchase|pos\b|card|netflix|spotify|showmax|cotton on|sportscene|"
            r"h&m|zara|ocean basket|spur|fishaways|chicken licken|galito|romans|pharmacy|"
            r"avis|hertz|rent-a-car|car hire|car rental)",
            re.I,
        ),
    ),
)


def classify(description: str) -> str:
    """Map a transaction description to one of :data:`CATEGORIES` by keyword rule."""
    text = description or ""
    for category, pattern in _RULES:
        if pattern.search(text):
            return category
    return "other"


_REF_RE = re.compile(r"\b(?:REF|REFERENCE|TXN|TRN|ID)[:#\s]*([A-Z0-9][A-Z0-9-]{5,})", re.I)


def extract_reference(description: str) -> str | None:
    """Pull a bank reference (``REF 00012345``) out of a description if present."""
    m = _REF_RE.search(description or "")
    return m.group(1).upper() if m else None


# --------------------------------------------------------------------------- money / dates
_CR_DR_RE = re.compile(r"\s*(cr|dr)\.?\s*$", re.I)


def parse_amount(text: str | None) -> float | None:
    """Parse ``R 1 234,56``, ``-1,234.56``, ``(1234.56)``, ``1234.56-``, ``12.34 Cr`` ... to float.

    Returns ``None`` for an empty cell. Raises ``ValueError`` for garbage.
    """
    if text is None:
        return None
    s = str(text).strip().replace(" ", " ")
    if not s or s in {"-", "--"}:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    m = _CR_DR_RE.search(s)
    if m:
        if m.group(1).lower() == "dr":
            negative = True
        s = s[: m.start()].strip()
    s = re.sub(r"(?i)\bzar\b", "", s)
    s = s.replace("R", "").replace("r", "").strip()
    if s.endswith("-"):
        negative = True
        s = s[:-1].strip()
    if s.startswith("-"):
        negative = True
        s = s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()
    s = s.replace(" ", "").replace("'", "")
    if "," in s and "." in s:
        decimal_sep = "," if s.rfind(",") > s.rfind(".") else "."
        other = "." if decimal_sep == "," else ","
        s = s.replace(other, "").replace(decimal_sep, ".")
    elif "," in s:
        s = s.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+", s) else s.replace(",", ".")
    elif "." in s and re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    if not re.fullmatch(r"\d+(\.\d+)?", s):
        raise ValueError(f"unparseable amount: {text!r}")
    value = float(s)
    return -value if negative else value


_DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d", "%d %b %Y", "%d %B %Y", "%d/%m/%y")


def parse_date(text: str) -> date:
    s = (text or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {text!r}")


# --------------------------------------------------------------------------- layout detection
_HEADER_KEYS = {
    "date": ("date",),
    "description": ("description", "details", "narrative", "transaction description", "memo"),
    "amount": ("amount",),
    "debit": ("debit", "debits", "withdrawal"),
    "credit": ("credit", "credits", "deposit"),
    "balance": ("balance",),
}


def _norm(h: str) -> str:
    return re.sub(r"[^a-z]+", " ", h.lower()).strip()


def _map_headers(headers: list[str]) -> dict[str, int]:
    cols: dict[str, int] = {}
    normed = [_norm(h) for h in headers]
    for key, aliases in _HEADER_KEYS.items():
        for i, h in enumerate(normed):
            if i in cols.values():
                continue
            if any(h == a or h.startswith(a + " ") or h.endswith(" " + a) for a in aliases):
                cols[key] = i
                break
    if "date" not in cols or ("amount" not in cols and "debit" not in cols):
        raise StatementParseError(f"could not recognise statement header: {headers!r}")
    return cols


def _is_header(line: str) -> bool:
    low = line.lower()
    return "date" in low and any(k in low for k in ("amount", "debit", "credit", "balance"))


def _split(line: str, delimiter: str) -> list[str]:
    if delimiter == "|":
        return [c.strip() for c in line.split("|")]
    return [c.strip() for c in next(csv.reader([line], delimiter=delimiter))]


def parse_statement(text: str) -> list[Transaction]:
    """Parse statement text (any supported layout) into chronological transactions."""
    lines = text.splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if _is_header(ln)), None)
    if header_idx is None:
        raise StatementParseError("no header row found (need Date and Amount/Debit columns)")
    header_line = lines[header_idx]
    if "|" in header_line:
        delimiter = "|"
    elif header_line.count(";") > header_line.count(","):
        delimiter = ";"
    elif header_line.count(",") > 0:
        delimiter = ","
    else:
        delimiter = "\t"
    cols = _map_headers(_split(header_line, delimiter))

    txns: list[Transaction] = []
    for offset, raw in enumerate(lines[header_idx + 1 :], start=header_idx + 2):
        if not raw.strip() or set(raw.strip()) <= {"-", "=", "|", " "}:
            continue
        cells = _split(raw, delimiter)
        if len(cells) <= max(cols.values()):
            continue
        try:
            when = parse_date(cells[cols["date"]])
        except ValueError:
            continue  # footer / opening-balance rows
        description = cells[cols["description"]] if "description" in cols else ""
        if "amount" in cols:
            amount = parse_amount(cells[cols["amount"]])
        else:
            debit = parse_amount(cells[cols["debit"]]) or 0.0
            credit = parse_amount(cells[cols["credit"]]) if "credit" in cols else 0.0
            amount = (credit or 0.0) - abs(debit)
        if amount is None:
            continue
        balance = parse_amount(cells[cols["balance"]]) if "balance" in cols else None
        txns.append(
            Transaction(
                date=when,
                description=description.strip(),
                amount=round(amount, 2),
                balance=None if balance is None else round(balance, 2),
                category=classify(description),
                txn_id=extract_reference(description),
                line_no=offset,
            )
        )
    if not txns:
        raise StatementParseError("header found but no transaction rows parsed")
    if txns[0].date > txns[-1].date:
        txns.reverse()
    return txns


def load_statement(path: str | Path) -> list[Transaction]:
    return parse_statement(Path(path).read_text(encoding="utf-8-sig"))
