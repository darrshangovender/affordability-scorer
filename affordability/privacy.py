"""POPIA plumbing: lawful-purpose records, PII redaction, access logging, retention.

Nothing in this module talks to a network. The scoring pipeline never needs an
LLM; if one is used for rationale polish, the prompt passes through
:func:`redact` first, and every log line emitted by the ``affordability``
loggers passes through :class:`RedactingFilter`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- redaction
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ID13_RE = re.compile(r"(?<!\d)\d{13}(?!\d)")
_PHONE_RE = re.compile(r"(?<![\d\w])(?:\+27|0)[\s-]?[1-9]\d[\s-]?\d{3}[\s-]?\d{4}(?!\d)")
_ACCOUNT_RE = re.compile(r"(?<![\d.,])\d{8,12}(?![\d.,])")
_LABELLED_ACC_RE = re.compile(r"(?i)(acc(?:ount)?\.?\s*(?:no\.?|number|#)?\s*[:#]?\s*)(\d[\d\s-]{5,}\d)")


def luhn_valid(digits: str) -> bool:
    """Standard Luhn checksum (used by SA ID numbers and card numbers)."""
    if not digits.isdigit() or len(digits) < 2:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def is_sa_id(candidate: str) -> bool:
    """13 digits, plausible YYMMDD prefix, citizenship digit 0/1, Luhn-valid."""
    if not re.fullmatch(r"\d{13}", candidate):
        return False
    mm, dd = int(candidate[2:4]), int(candidate[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return False
    if candidate[10] not in "01":
        return False
    return luhn_valid(candidate)


def redact(text: str) -> str:
    """Replace emails, SA ID numbers, phone numbers and account numbers with tokens."""
    if not text:
        return text
    out = _EMAIL_RE.sub("[EMAIL]", text)
    out = _ID13_RE.sub(lambda m: "[SA_ID]" if is_sa_id(m.group(0)) else m.group(0), out)
    out = _LABELLED_ACC_RE.sub(lambda m: f"{m.group(1)}[ACCOUNT]", out)
    out = _PHONE_RE.sub("[PHONE]", out)
    out = _ACCOUNT_RE.sub("[ACCOUNT]", out)
    return out


def contains_pii(text: str) -> bool:
    return redact(text) != text


class RedactingFilter(logging.Filter):
    """Rewrites ``record.msg``/``record.args`` in place so no handler ever sees raw PII."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: redact(str(v)) for k, v in record.args.items()}
            else:
                record.args = tuple(redact(str(a)) for a in record.args)
        return True


def get_logger(name: str) -> logging.Logger:
    """Package loggers must be obtained here so the redacting filter is always attached."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, RedactingFilter) for f in logger.filters):
        logger.addFilter(RedactingFilter())
    return logger


# --------------------------------------------------------------------------- lawful purpose
LawfulBasis = Literal["consent", "contract", "legitimate_interest"]

ALLOWED_PURPOSES: frozenset[str] = frozenset(
    {
        "tenant_affordability_assessment",
        "lease_renewal_review",
        "internal_model_audit",
    }
)


class ScoringPurpose(BaseModel):
    """POPIA s.13: personal information may only be processed for a specific, lawful purpose."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    purpose: str
    consent_ref: str = Field(min_length=3)
    requested_by: str = Field(min_length=1)
    lawful_basis: LawfulBasis = "consent"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("purpose")
    @classmethod
    def _known_purpose(cls, v: str) -> str:
        if v not in ALLOWED_PURPOSES:
            raise ValueError(f"purpose must be one of {sorted(ALLOWED_PURPOSES)}, got {v!r}")
        return v

    @field_validator("requested_by", "consent_ref")
    @classmethod
    def _no_pii(cls, v: str) -> str:
        return redact(v)


def subject_token(identifier: str) -> str:
    """Pseudonymise a data-subject identifier (ID number, email) for logs."""
    return hashlib.sha256(identifier.strip().lower().encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- access log
class AccessEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    at: datetime
    action: str
    purpose: str
    consent_ref: str
    requested_by: str
    subject: str
    detail: str = ""


class AccessLog:
    """In-memory, redacting access log with a retention policy."""

    def __init__(self, retention_days: int = 90) -> None:
        self.retention_days = retention_days
        self._events: list[AccessEvent] = []

    def record(
        self,
        action: str,
        purpose: ScoringPurpose,
        subject: str,
        detail: str = "",
        *,
        now: datetime | None = None,
    ) -> AccessEvent:
        event = AccessEvent(
            at=now or datetime.now(timezone.utc),
            action=action,
            purpose=purpose.purpose,
            consent_ref=purpose.consent_ref,
            requested_by=purpose.requested_by,
            subject=subject_token(subject),
            detail=redact(detail),
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> list[AccessEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def purge(self, older_than_days: int | None = None, *, now: datetime | None = None) -> int:
        """Drop events older than the retention window. Returns how many were removed."""
        days = self.retention_days if older_than_days is None else older_than_days
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
        before = len(self._events)
        self._events = [e for e in self._events if e.at >= cutoff]
        return before - len(self._events)

    def to_json(self) -> str:
        return json.dumps([e.model_dump(mode="json") for e in self._events], indent=2)
