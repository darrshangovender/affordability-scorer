"""Parsers for bank statements and payslips, plus document-consistency checks."""

from affordability.parse.payslip import Payslip, PayslipParseError, load_payslip, parse_payslip
from affordability.parse.statement import (
    CATEGORIES,
    StatementParseError,
    Transaction,
    classify,
    load_statement,
    parse_amount,
    parse_date,
    parse_statement,
)
from affordability.parse.tamper import TamperFlag, TamperReport, check_document

__all__ = [
    "CATEGORIES",
    "Payslip",
    "PayslipParseError",
    "StatementParseError",
    "TamperFlag",
    "TamperReport",
    "Transaction",
    "check_document",
    "classify",
    "load_payslip",
    "load_statement",
    "parse_amount",
    "parse_date",
    "parse_payslip",
    "parse_statement",
]
