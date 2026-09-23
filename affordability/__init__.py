"""affordability-scorer: explainable, POPIA-aware tenant affordability scoring for SA rentals."""

from affordability.calibrate import Calibrator, brier_score, default_calibrator
from affordability.fairness import (
    AuditReport,
    DisparateImpactResult,
    ProtectedAttributeError,
    assert_no_protected_fields,
    disparate_impact_test,
    proxy_audit,
)
from affordability.features import FeatureVector, InsufficientHistoryError, build_features
from affordability.parse import (
    Payslip,
    TamperReport,
    Transaction,
    check_document,
    load_payslip,
    load_statement,
    parse_payslip,
    parse_statement,
)
from affordability.pipeline import Assessment, score_application
from affordability.privacy import AccessLog, ScoringPurpose, redact
from affordability.rationale import MockModel, Rationale, build_rationale, polish
from affordability.score import ScoreResult, band_for, explain, score_features

__version__ = "0.1.0"

__all__ = [
    "AccessLog",
    "Assessment",
    "AuditReport",
    "Calibrator",
    "DisparateImpactResult",
    "FeatureVector",
    "InsufficientHistoryError",
    "MockModel",
    "Payslip",
    "ProtectedAttributeError",
    "Rationale",
    "ScoreResult",
    "ScoringPurpose",
    "TamperReport",
    "Transaction",
    "assert_no_protected_fields",
    "band_for",
    "brier_score",
    "build_features",
    "build_rationale",
    "check_document",
    "default_calibrator",
    "disparate_impact_test",
    "explain",
    "load_payslip",
    "load_statement",
    "parse_payslip",
    "parse_statement",
    "polish",
    "proxy_audit",
    "redact",
    "score_application",
    "score_features",
]
