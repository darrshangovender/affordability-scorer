from dataclasses import replace
from datetime import date, timedelta

from affordability.parse.payslip import Payslip, load_payslip, parse_payslip
from affordability.parse.statement import Transaction, load_statement, parse_statement
from affordability.parse.tamper import (
    check_date_gaps,
    check_document,
    check_duplicates,
    check_payslip,
    check_round_numbers,
    check_running_balance,
)
from tests.conftest import render


def test_clean_ledger_reconciles(transactions):
    assert check_running_balance(transactions) == []


def test_edited_amount_breaks_running_balance(statement_rows):
    d, desc, amount, balance = statement_rows[8]  # the first salary line
    rows = list(statement_rows)
    rows[8] = (d, desc, amount + 4_000.0, balance)  # amount edited, balance left as is
    flags = check_running_balance(parse_statement(render("fnb", rows)))
    assert len(flags) == 1
    assert flags[0].severity == "high"
    assert flags[0].line_no == 8 + 5  # 4 header lines + 1-based


def test_edited_line_in_every_layout(statement_rows):
    rows = list(statement_rows)
    d, desc, amount, balance = rows[8]
    rows[8] = (d, desc, amount + 1_000.0, balance)
    for layout in ("fnb", "standard", "capitec", "absa"):
        assert check_running_balance(parse_statement(render(layout, rows)))


def test_payslip_within_two_percent_passes(transactions, payslip_text):
    slip = parse_payslip(payslip_text)
    assert check_payslip(transactions, slip) == []
    close = Payslip(employer="Acme Trading", pay_date=date(2026, 6, 25), gross=None, net=20_300.0)
    assert check_payslip(transactions, close) == []


def test_payslip_mismatch_flags(transactions):
    slip = Payslip(employer="Acme Trading", pay_date=date(2026, 6, 25), gross=None, net=24_000.0)
    flags = check_payslip(transactions, slip)
    assert any(f.check == "payslip_match" and f.severity == "high" for f in flags)


def test_payslip_employer_not_on_statement_is_medium(transactions):
    slip = Payslip(employer="Totally Different Corp", pay_date=date(2026, 6, 25), gross=None, net=20_000.0)
    flags = check_payslip(transactions, slip)
    assert [f.severity for f in flags] == ["medium"]


def test_payslip_without_salary_credit(transactions):
    no_salary = [t for t in transactions if t.category != "salary"]
    flags = check_payslip(no_salary, Payslip(None, None, None, 1.0))
    assert flags and flags[0].severity == "medium"


def test_no_payslip_means_no_flags(transactions):
    assert check_payslip(transactions, None) == []


def test_date_gap_flagged(transactions):
    shifted = [
        replace(t, date=t.date + timedelta(days=60)) if t.date >= date(2026, 5, 1) else t
        for t in transactions
    ]
    flags = check_date_gaps(shifted)
    assert flags and flags[0].check == "date_gap"


def test_dates_running_backwards_is_high(transactions):
    swapped = [transactions[-1], *transactions[1:-1], transactions[0]]
    assert any(f.severity == "high" for f in check_date_gaps(swapped))


def test_duplicate_reference_flagged(transactions):
    salary = next(t for t in transactions if t.category == "salary")
    dup = replace(salary, date=salary.date + timedelta(days=1), line_no=999)
    flags = check_duplicates([*transactions, dup])
    assert any(f.check == "duplicate_id" and f.severity == "high" for f in flags)


def test_duplicate_credit_without_reference_is_medium():
    t = Transaction(date(2026, 3, 25), "BIG CREDIT", 5_000.0, 5_000.0, "other", None, 2)
    flags = check_duplicates([t, replace(t, balance=10_000.0, line_no=3)])
    assert [f.severity for f in flags] == ["medium"]


def test_round_numbers_density():
    rows = [
        Transaction(date(2026, 3, 1) + timedelta(days=i), "X", -100.0 * (i + 1), 0.0, "retail", None, i)
        for i in range(25)
    ]
    assert check_round_numbers(rows)
    real = [replace(t, amount=t.amount - 0.37) for t in rows]
    assert check_round_numbers(real) == []


def test_round_numbers_needs_enough_lines():
    rows = [Transaction(date(2026, 3, 1), "X", -100.0, 0.0, "retail", None, i) for i in range(5)]
    assert check_round_numbers(rows) == []


def test_short_statement_is_low_risk_coverage_note(transactions):
    one_month = [t for t in transactions if t.date.month == 3]
    report = check_document(one_month)
    assert report.risk == "low"
    assert report.by_check("coverage")


def test_report_summary_and_dict(transactions):
    report = check_document(transactions)
    assert report.ok
    assert report.summary() == "no consistency issues found"
    assert report.to_dict() == {"risk": "low", "flags": []}


def test_demo_clean_personas_are_low_risk(manifest, demo_dir):
    for e in manifest:
        if e["tampered"]:
            continue
        report = check_document(load_statement(demo_dir / e["statement"]), load_payslip(demo_dir / e["payslip"]))
        assert report.risk == "low", (e["id"], report.summary())


def test_demo_tampered_personas_are_high_risk(manifest, demo_dir):
    kinds = {}
    for e in manifest:
        if not e["tampered"]:
            continue
        report = check_document(load_statement(demo_dir / e["statement"]), load_payslip(demo_dir / e["payslip"]))
        assert report.risk == "high", e["id"]
        kinds[e["tamper_kind"]] = {f.check for f in report.flags}
    assert "running_balance" in kinds["edited_salary"]
    assert "payslip_match" in kinds["edited_salary"]
    assert "duplicate_id" in kinds["duplicate_salary"]


def test_commission_persona_payslip_does_not_false_positive(manifest, demo_dir):
    """Variable income: the payslip is for the pay month, gross scales with net."""
    e = next(x for x in manifest if x["id"] == "p08_sales_rep_commission")
    txns = load_statement(demo_dir / e["statement"])
    slip = load_payslip(demo_dir / e["payslip"])
    salaries = [t.amount for t in txns if t.category == "salary"]
    assert max(salaries) / min(salaries) > 1.3  # income really does swing
    assert check_payslip(txns, slip) == []
