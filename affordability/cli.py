"""``affordability`` command line: ``score``, ``audit``, ``tamper``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from affordability.fairness import proxy_audit
from affordability.parse.payslip import load_payslip
from affordability.parse.statement import load_statement
from affordability.parse.tamper import check_document
from affordability.pipeline import score_application
from affordability.privacy import AccessLog, ScoringPurpose
from affordability.score import explain


def _purpose_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--purpose", default="tenant_affordability_assessment")
    p.add_argument("--consent-ref", default="cli-consent", help="reference to the applicant's consent")
    p.add_argument("--requested-by", default="cli", help="who is asking (agent / landlord id)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="affordability",
        description="Explainable, POPIA-aware tenant affordability scoring (offline).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("score", help="score a statement (+ optional payslip) against a rent")
    s.add_argument("--statement", required=True, type=Path)
    s.add_argument("--payslip", type=Path, default=None)
    s.add_argument("--rent", required=True, type=float, help="proposed monthly rent in rand")
    s.add_argument("--json", action="store_true", help="emit the full assessment as JSON")
    s.add_argument("--tenant", action="store_true", help="print the tenant-facing rationale")
    _purpose_args(s)

    a = sub.add_parser("audit", help="show which transactions the proxy audit excludes and why")
    a.add_argument("--statement", required=True, type=Path)
    a.add_argument("--json", action="store_true")

    t = sub.add_parser("tamper", help="run document-consistency checks")
    t.add_argument("--statement", required=True, type=Path)
    t.add_argument("--payslip", type=Path, default=None)
    t.add_argument("--json", action="store_true")
    return parser


def cmd_score(args: argparse.Namespace) -> int:
    purpose = ScoringPurpose(
        purpose=args.purpose, consent_ref=args.consent_ref, requested_by=args.requested_by
    )
    assessment = score_application(
        statement=args.statement,
        payslip=args.payslip,
        rent=args.rent,
        purpose=purpose,
        subject=str(args.statement),
        access_log=AccessLog(),
    )
    if args.json:
        print(json.dumps(assessment.to_dict(), indent=2))
        return 0
    r = assessment.score
    print(f"score: {r.score}/100  band: {r.band}  tamper: {assessment.tamper.risk}")
    if r.probability is not None:
        print(f"P(no default, 12m): {r.probability:.0%}")
    print()
    for feature, value, points, reason in explain(r):
        print(f"  {points:+4d}  {feature:<18} {value:>10.3f}  {reason}")
    print()
    print(assessment.rationale.tenant if args.tenant else assessment.rationale.landlord)
    return 0 if assessment.tamper.risk != "high" else 2


def cmd_audit(args: argparse.Namespace) -> int:
    txns = load_statement(args.statement)
    report = proxy_audit(txns)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0
    print(report.summary())
    for item in report.excluded:
        print(f"  EXCLUDED line {item.line_no} {item.date} {item.amount:>10.2f}  [{item.ground}] "
              f"matched '{item.matched}': {item.why}")
    for item in report.stripped:
        print(f"  STRIPPED line {item.line_no} {item.date} {item.amount:>10.2f}  [{item.ground}] "
              f"matched '{item.matched}'")
    return 0


def cmd_tamper(args: argparse.Namespace) -> int:
    txns = load_statement(args.statement)
    slip = load_payslip(args.payslip) if args.payslip else None
    report = check_document(txns, slip)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())
        for f in report.flags:
            print(f"  [{f.severity:<6}] {f.check:<16} {f.message}")
    return 0 if report.risk != "high" else 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"score": cmd_score, "audit": cmd_audit, "tamper": cmd_tamper}
    try:
        return handlers[args.command](args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
