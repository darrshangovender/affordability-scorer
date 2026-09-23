import logging
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from affordability.privacy import (
    AccessLog,
    RedactingFilter,
    ScoringPurpose,
    contains_pii,
    get_logger,
    is_sa_id,
    luhn_valid,
    redact,
    subject_token,
)

SA_ID = "9205244136087"  # fictional, Luhn-valid


def test_luhn():
    assert luhn_valid("79927398713")
    assert not luhn_valid("79927398710")
    assert not luhn_valid("abc")


def test_is_sa_id():
    assert is_sa_id(SA_ID)
    assert not is_sa_id("9213444136087")  # month 13
    assert not is_sa_id("9205244136287")  # citizenship digit 2
    assert not is_sa_id("9205244136080")  # bad checksum


@pytest.mark.parametrize(
    "text,token",
    [
        (f"ID Number: {SA_ID}", "[SA_ID]"),
        ("Account Number: 62012345678", "[ACCOUNT]"),
        ("acc no 1234 5678 9012", "[ACCOUNT]"),
        ("call 082 123 4567", "[PHONE]"),
        ("call +27 82 123 4567", "[PHONE]"),
        ("mail tenant@example.co.za now", "[EMAIL]"),
    ],
)
def test_redact_each_kind(text, token):
    out = redact(text)
    assert token in out
    assert not contains_pii(out)


def test_redact_leaves_ordinary_numbers_alone():
    text = "score 57/100 rent R5,000.00 on 25/08/2026 ref 1234567"
    assert redact(text) == text


def test_non_luhn_13_digits_are_not_an_id():
    assert "[SA_ID]" not in redact("ref 1234567890123")
    assert "[SA_ID]" not in redact("total 9999999999999")


def test_redact_empty():
    assert redact("") == ""


def test_logging_filter_redacts_msg_and_args(caplog):
    logger = get_logger("affordability.test")
    assert any(isinstance(f, RedactingFilter) for f in logger.filters)
    with caplog.at_level(logging.INFO, logger="affordability.test"):
        logger.info("subject %s id %s", "tenant@example.com", SA_ID)
        logger.info(f"inline {SA_ID}")
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert SA_ID not in joined and "tenant@example.com" not in joined
    assert "[SA_ID]" in joined and "[EMAIL]" in joined


def test_get_logger_attaches_filter_once():
    a = get_logger("affordability.once")
    b = get_logger("affordability.once")
    assert a is b and sum(isinstance(f, RedactingFilter) for f in a.filters) == 1


def test_scoring_purpose_requires_known_purpose():
    with pytest.raises(ValidationError):
        ScoringPurpose(purpose="marketing", consent_ref="abc", requested_by="agent")
    p = ScoringPurpose(purpose="tenant_affordability_assessment", consent_ref="abc", requested_by="agent")
    assert p.lawful_basis == "consent" and p.recorded_at.tzinfo is not None


def test_scoring_purpose_redacts_its_own_fields():
    p = ScoringPurpose(purpose="lease_renewal_review", consent_ref=f"id {SA_ID}", requested_by="a@b.co")
    assert SA_ID not in p.consent_ref and "[EMAIL]" == p.requested_by


def test_scoring_purpose_is_frozen_and_strict():
    p = ScoringPurpose(purpose="internal_model_audit", consent_ref="abc", requested_by="x")
    with pytest.raises(ValidationError):
        p.purpose = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ScoringPurpose(purpose="internal_model_audit", consent_ref="abc", requested_by="x", race="y")


def test_subject_token_is_pseudonymous():
    tok = subject_token(SA_ID)
    assert tok == subject_token(f" {SA_ID} ") and len(tok) == 16 and SA_ID not in tok


def test_access_log_records_redacted_events():
    log = AccessLog()
    p = ScoringPurpose(purpose="tenant_affordability_assessment", consent_ref="c-01", requested_by="agent-7")
    ev = log.record("score", p, subject=SA_ID, detail=f"phone 082 123 4567 id {SA_ID}")
    assert ev.subject == subject_token(SA_ID)
    assert SA_ID not in log.to_json() and "082 123 4567" not in log.to_json()
    assert len(log) == 1 and log.events[0].action == "score"


def test_purge_older_than_days():
    log = AccessLog(retention_days=90)
    p = ScoringPurpose(purpose="tenant_affordability_assessment", consent_ref="c-01", requested_by="agent")
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    log.record("score", p, "a", now=now - timedelta(days=100))
    log.record("score", p, "b", now=now - timedelta(days=10))
    log.record("score", p, "c", now=now)
    assert log.purge(older_than_days=30, now=now) == 1
    assert len(log) == 2
    assert log.purge(now=now) == 0  # default retention keeps the rest
    assert log.purge(older_than_days=0, now=now + timedelta(seconds=1)) == 2
