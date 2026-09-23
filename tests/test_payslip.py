from datetime import date

import pytest

from affordability.parse.payslip import PayslipParseError, load_payslip, parse_payslip


def test_parse_fields(payslip_text):
    slip = parse_payslip(payslip_text)
    assert slip.employer == "Acme Trading"
    assert slip.pay_date == date(2026, 6, 25)
    assert slip.gross == pytest.approx(26_500.0)
    assert slip.net == pytest.approx(20_000.0)
    assert slip.deductions == {"paye": 4323.0, "uif": 177.0, "pension fund": 2000.0}
    assert slip.total_deductions == pytest.approx(6_500.0)


def test_net_derived_from_gross_and_total_deductions():
    text = "Employer: X\nGross Pay: R 10,000.00\nTotal Deductions: R 2,500.00\n"
    assert parse_payslip(text).net == pytest.approx(7_500.0)


def test_net_derived_from_itemised_deductions():
    text = "Gross Salary: R 10 000,00\nPAYE: R 1 500,00\nUIF: R 100,00\n"
    assert parse_payslip(text).net == pytest.approx(8_400.0)


def test_take_home_label_and_comma_decimals():
    assert parse_payslip("Take-home pay: R 12 345,67").net == pytest.approx(12_345.67)


def test_tax_reference_number_is_not_a_deduction():
    text = "Tax Reference No: 1234567890\nNet Pay: R 5,000.00\n"
    slip = parse_payslip(text)
    assert "tax reference no" not in slip.deductions
    assert slip.net == 5_000.0


def test_missing_net_raises():
    with pytest.raises(PayslipParseError):
        parse_payslip("Employer: Nobody\nHello\n")


def test_load_payslip_from_demo(demo_dir, manifest):
    for e in manifest:
        slip = load_payslip(demo_dir / e["payslip"])
        assert slip.net > 0
        assert slip.gross is not None and slip.gross > slip.net
        assert slip.pay_date is not None
