"""End-to-end: documents -> tamper check -> proxy audit -> features -> score -> rationale."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from affordability.calibrate import Calibrator, default_calibrator
from affordability.fairness import AuditReport, proxy_audit
from affordability.features import FeatureVector, build_features
from affordability.parse.payslip import Payslip, load_payslip
from affordability.parse.statement import Transaction, load_statement
from affordability.parse.tamper import TamperReport, check_document
from affordability.privacy import AccessLog, ScoringPurpose, get_logger
from affordability.rationale import ModelClient, Rationale, build_rationale, polish
from affordability.score import ScoreResult, score_features

log = get_logger("affordability.pipeline")


@dataclass
class Assessment:
    purpose: ScoringPurpose
    proposed_rent: float
    tamper: TamperReport
    audit: AuditReport
    features: FeatureVector
    score: ScoreResult
    rationale: Rationale

    def to_dict(self) -> dict:
        return {
            "purpose": self.purpose.model_dump(mode="json"),
            "proposed_rent": self.proposed_rent,
            "score": self.score.to_dict(),
            "features": self.features.to_dict(),
            "tamper": self.tamper.to_dict(),
            "audit": self.audit.to_dict(),
            "rationale": self.rationale.to_dict(),
        }


def score_application(
    *,
    statement: str | Path | Sequence[Transaction],
    rent: float,
    purpose: ScoringPurpose,
    payslip: str | Path | Payslip | None = None,
    subject: str = "applicant",
    access_log: AccessLog | None = None,
    calibrator: Calibrator | None = None,
    model: ModelClient | None = None,
) -> Assessment:
    """Score one application. No network access; ``model`` is optional polish only."""
    if access_log is not None:
        access_log.record("score", purpose, subject, detail=f"rent={rent}")
    txns = (
        list(statement)
        if isinstance(statement, (list, tuple))
        else load_statement(statement)
    )
    slip = load_payslip(payslip) if isinstance(payslip, (str, Path)) else payslip

    tamper = check_document(txns, slip)
    audit = proxy_audit(txns)
    features = build_features(txns, rent, audit=audit)
    result = score_features(features)
    cal = calibrator or default_calibrator()
    result.probability = round(cal.predict(result.score), 4)
    rationale = build_rationale(result, features, tamper=tamper, audit=audit)
    if model is not None:
        rationale = polish(rationale, model)
    log.info(
        "scored subject=%s rent=%s score=%s band=%s tamper=%s",
        subject,
        rent,
        result.score,
        result.band,
        tamper.risk,
    )
    return Assessment(
        purpose=purpose,
        proposed_rent=float(rent),
        tamper=tamper,
        audit=audit,
        features=features,
        score=result,
        rationale=rationale,
    )
