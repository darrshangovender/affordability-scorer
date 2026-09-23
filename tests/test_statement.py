from datetime import date

import pytest

from affordability.parse.statement import (
    CATEGORIES,
    StatementParseError,
    classify,
    extract_reference,
    load_statement,
    parse_amount,
    parse_date,
    parse_statement,
)
from tests.conftest import render


# --------------------------------------------------------------------------- layouts
@pytest.mark.parametrize("layout", ["fnb", "standard", "capitec", "absa"])
def test_each_bank_layout_parses_to_same_ledger(layout, statement_rows):
    txns = parse_statement(render(layout, statement_rows))
    assert len(txns) == len(statement_rows)
    for t, (d, desc, amount, balance) in zip(txns, statement_rows, strict=True):
        assert t.date == d
        assert t.description == desc
        assert t.amount == pytest.approx(amount)
        assert t.balance == pytest.approx(balance)


def test_line_numbers_point_at_source_lines(fnb_text):
    txns = parse_statement(fnb_text)
    lines = fnb_text.splitlines()
    for t in txns:
        assert t.description in lines[t.line_no - 1]


def test_reverse_chronological_input_is_flipped(statement_rows):
    text = render("fnb", list(reversed(statement_rows)))
    txns = parse_statement(text)
    assert txns[0].date < txns[-1].date


def test_no_header_raises():
    with pytest.raises(StatementParseError):
        parse_statement("hello\nworld\n")


def test_header_without_rows_raises():
    with pytest.raises(StatementParseError):
        parse_statement("Date,Description,Amount,Balance\n")


def test_footer_rows_without_dates_are_skipped(fnb_text):
    text = fnb_text + "Closing balance,,,1234.00\n"
    assert len(parse_statement(text)) == len(parse_statement(fnb_text))


def test_load_statement_handles_bom(tmp_path, fnb_text):
    p = tmp_path / "s.csv"
    p.write_bytes(b"\xef\xbb\xbf" + fnb_text.encode("utf-8"))
    assert len(load_statement(p)) > 0


def test_demo_statements_all_parse(manifest, demo_dir):
    for e in manifest:
        txns = load_statement(demo_dir / e["statement"])
        assert len(txns) > 50
        assert all(t.balance is not None for t in txns)


# --------------------------------------------------------------------------- money
@pytest.mark.parametrize(
    "text,expected",
    [
        ("1234.56", 1234.56),
        ("-1234.56", -1234.56),
        ("1,234.56", 1234.56),
        ("R 1 234,56", 1234.56),
        ("-R1 234,56", -1234.56),
        ("R1 234 567,89", 1234567.89),
        ("1234,56", 1234.56),
        ("(1234.56)", -1234.56),
        ("1234.56-", -1234.56),
        ("12.34 Cr", 12.34),
        ("12.34 Dr", -12.34),
        ("ZAR 500.00", 500.0),
        ("1.234,56", 1234.56),
        ("1.234.567", 1234567.0),
        ("+99", 99.0),
        ("", None),
        (None, None),
        ("-", None),
    ],
)
def test_parse_amount(text, expected):
    got = parse_amount(text)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


def test_parse_amount_rejects_garbage():
    with pytest.raises(ValueError):
        parse_amount("twelve rand")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("25/08/2026", date(2026, 8, 25)),
        ("25-08-2026", date(2026, 8, 25)),
        ("2026-08-25", date(2026, 8, 25)),
        ("25 Aug 2026", date(2026, 8, 25)),
        ("25 August 2026", date(2026, 8, 25)),
        ("25/08/26", date(2026, 8, 25)),
    ],
)
def test_parse_date(text, expected):
    assert parse_date(text) == expected


def test_parse_date_rejects_us_order():
    with pytest.raises(ValueError):
        parse_date("13/25/2026")


# --------------------------------------------------------------------------- classification
@pytest.mark.parametrize(
    "description,category",
    [
        ("ACME TRADING SALARY REF 0120260825", "salary"),
        ("WAGES WEEK 34", "salary"),
        ("COMMISSION PAYOUT AUG", "salary"),
        ("RENT PAYMENT FLAT 4B", "rent"),
        ("RENTAL PAYMENT BEREA APARTMENTS", "rent"),
        ("AVIS RENT-A-CAR", "retail"),
        ("CAPFIN LOAN INSTALMENT", "loan"),
        ("WESBANK VEHICLE FINANCE", "loan"),
        ("BETWAY DEPOSIT", "gambling"),
        ("HOLLYWOODBETS", "gambling"),
        ("LOTTO ONLINE", "gambling"),
        ("FNB ATM CASH WITHDRAWAL", "cash"),
        ("CASHSEND TO 0821234567", "cash"),
        ("DEBIT ORDER DISCOVERY MEDICAL AID", "debit_order"),
        ("DEBIT ORDER MTN CONTRACT", "debit_order"),
        ("SCHOOL FEES ST JOHNS PREP", "debit_order"),
        ("PAYMENT TO SAVINGS POCKET", "transfer"),
        ("PAYMENT FROM T MOKOENA", "transfer"),
        ("WESTERN UNION REMITTANCE", "transfer"),
        ("MAINTENANCE PAYMENT S DLAMINI", "transfer"),
        ("MONTHLY ACCOUNT FEE", "fees"),
        ("CHECKERS HYPER", "retail"),
        ("UBER TRIP", "retail"),
        ("NETFLIX.COM", "retail"),
        ("TITHE - GRACE FAMILY CHURCH", "other"),
        ("", "other"),
    ],
)
def test_classify(description, category):
    assert classify(description) == category


def test_salary_rule_wins_over_transfer():
    assert classify("SALARY PAYMENT FROM ACME") == "salary"


def test_every_category_is_known():
    for desc in ("SALARY", "RENT", "LOAN", "BETWAY", "ATM", "DEBIT ORDER", "EFT", "FEE", "SPAR", "zzz"):
        assert classify(desc) in CATEGORIES


def test_extract_reference():
    assert extract_reference("ACME SALARY REF 0120260825") == "0120260825"
    assert extract_reference("ACME SALARY ref: ab-123456") == "AB-123456"
    assert extract_reference("CHECKERS HYPER") is None


def test_fingerprint_is_stable_and_distinct(transactions):
    a, b = transactions[0], transactions[1]
    assert a.fingerprint == a.fingerprint
    assert a.fingerprint != b.fingerprint


def test_demo_labels_match_classifier(manifest, demo_dir):
    """Every generator label in the demo set is reproduced by the keyword rules."""
    import csv

    for e in manifest:
        if e["tampered"]:
            continue
        txns = load_statement(demo_dir / e["statement"])
        with (demo_dir / e["labels"]).open(encoding="utf-8", newline="") as fh:
            labels = {int(r["line_no"]): r["category"] for r in csv.DictReader(fh)}
        mismatches = [(t.line_no, t.description, t.category, labels[t.line_no]) for t in txns
                      if t.category != labels[t.line_no]]
        assert not mismatches, mismatches[:5]
