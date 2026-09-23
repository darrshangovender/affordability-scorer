import json
import logging

import pytest

from affordability import __version__
from affordability.cli import main
from affordability.pipeline import score_application
from affordability.privacy import AccessLog, ScoringPurpose
from affordability.rationale import MockModel

SA_ID = "9205244136087"


@pytest.fixture
def purpose():
    return ScoringPurpose(purpose="tenant_affordability_assessment", consent_ref="c-1", requested_by="agent")


def _paths(manifest, demo_dir, pid):
    e = next(x for x in manifest if x["id"] == pid)
    return demo_dir / e["statement"], demo_dir / e["payslip"], e["proposed_rent"]


def test_version():
    assert __version__ == "0.1.0"


def test_score_application_end_to_end(manifest, demo_dir, purpose):
    st, ps, rent = _paths(manifest, demo_dir, "p09_it_support")
    log = AccessLog()
    a = score_application(statement=st, payslip=ps, rent=rent, purpose=purpose, subject=SA_ID, access_log=log)
    assert a.score.band in {"strong", "adequate"}
    assert a.tamper.ok
    assert 0.0 <= a.score.probability <= 1.0
    assert a.features.months_observed == 6
    assert len(log) == 1 and SA_ID not in log.to_json()
    d = a.to_dict()
    assert json.dumps(d)  # serialisable
    assert d["purpose"]["purpose"] == purpose.purpose
    assert "landlord" in d["rationale"] and "tenant" in d["rationale"]


def test_score_application_accepts_transactions(transactions, purpose):
    a = score_application(statement=transactions, rent=6_000.0, purpose=purpose)
    assert a.score.score == 100


def test_score_application_with_mock_model(transactions, purpose):
    model = MockModel()
    a = score_application(statement=transactions, rent=6_000.0, purpose=purpose, model=model)
    assert "friendlier" in a.rationale.tenant
    assert model.prompts


def test_pipeline_log_never_contains_pii(transactions, purpose, caplog):
    with caplog.at_level(logging.INFO, logger="affordability.pipeline"):
        score_application(statement=transactions, rent=6_000.0, purpose=purpose, subject=SA_ID)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "scored subject=" in joined
    assert SA_ID not in joined and "[SA_ID]" in joined


def test_pipeline_needs_no_network(monkeypatch, transactions, purpose):
    import socket

    def boom(*a, **k):  # pragma: no cover - only hit on failure
        raise AssertionError("network call attempted")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    assert score_application(statement=transactions, rent=6_000.0, purpose=purpose).score.score >= 0


# --------------------------------------------------------------------------- CLI
def test_cli_score_text(manifest, demo_dir, capsys):
    st, ps, rent = _paths(manifest, demo_dir, "p07_teacher")
    rc = main(["score", "--statement", str(st), "--payslip", str(ps), "--rent", str(rent)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "score:" in out and "band:" in out and "P(no default" in out
    assert "What helped" in out


def test_cli_score_tenant_json(manifest, demo_dir, capsys):
    st, _, rent = _paths(manifest, demo_dir, "p04_admin_clerk")
    rc = main(["score", "--statement", str(st), "--rent", str(rent), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["score"]["band"] in {"strong", "adequate", "marginal", "insufficient"}
    assert payload["audit"]["excluded"]  # the tithe line
    rc = main(["score", "--statement", str(st), "--rent", str(rent), "--tenant"])
    assert rc == 0 and "Your affordability score" in capsys.readouterr().out


def test_cli_score_tampered_exits_2(manifest, demo_dir, capsys):
    st, ps, rent = _paths(manifest, demo_dir, "t01_edited_salary")
    assert main(["score", "--statement", str(st), "--payslip", str(ps), "--rent", str(rent)]) == 2


def test_cli_audit(manifest, demo_dir, capsys):
    st, _, _ = _paths(manifest, demo_dir, "p06_nurse")
    assert main(["audit", "--statement", str(st)]) == 0
    out = capsys.readouterr().out
    assert "EXCLUDED" in out and "family_responsibility" in out
    assert main(["audit", "--statement", str(st), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["by_ground"]


def test_cli_tamper(manifest, demo_dir, capsys):
    st, ps, _ = _paths(manifest, demo_dir, "t02_duplicate_salary")
    assert main(["tamper", "--statement", str(st), "--payslip", str(ps)]) == 2
    assert "duplicate_id" in capsys.readouterr().out
    st, ps, _ = _paths(manifest, demo_dir, "p02_security_guard")
    assert main(["tamper", "--statement", str(st), "--payslip", str(ps), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["risk"] == "low"


def test_cli_errors_are_exit_1(tmp_path, capsys):
    assert main(["score", "--statement", str(tmp_path / "missing.csv"), "--rent", "1000"]) == 1
    assert "error:" in capsys.readouterr().err
    bad = tmp_path / "bad.csv"
    bad.write_text("nothing here\n", encoding="utf-8")
    assert main(["tamper", "--statement", str(bad)]) == 1


def test_cli_bad_purpose_is_exit_1(manifest, demo_dir, capsys):
    st, _, rent = _paths(manifest, demo_dir, "p02_security_guard")
    assert main(["score", "--statement", str(st), "--rent", str(rent), "--purpose", "marketing"]) == 1
