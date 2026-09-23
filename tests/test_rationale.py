import pytest

from affordability.fairness import proxy_audit
from affordability.features import build_features
from affordability.parse.tamper import check_document
from affordability.rationale import (
    MockModel,
    Rationale,
    RationaleIntegrityError,
    _prompt,
    build_rationale,
    numbers_in,
    polish,
    verify_preserved,
)
from affordability.score import explain, score_features

SA_ID = "9205244136087"


@pytest.fixture
def scored(transactions):
    fv = build_features(transactions, 6_000.0)
    return fv, score_features(fv)


def test_rationale_is_deterministic(scored):
    fv, res = scored
    assert build_rationale(res, fv) == build_rationale(res, fv)


def test_both_parties_get_same_reasons_and_score(scored):
    fv, res = scored
    r = build_rationale(res, fv)
    for _, _, _, reason in explain(res):
        assert reason in r.landlord and reason in r.tenant
    assert f"{res.score}/100" in r.landlord and f"{res.score}/100" in r.tenant
    assert set(r.reasons) == {reason for _, _, _, reason in explain(res)}
    assert "R6,000" in r.landlord and "R6,000" in r.tenant


def test_rationale_mentions_tamper_and_audit_when_relevant(transactions, scored):
    fv, res = scored
    tamper = check_document(transactions)
    audit = proxy_audit(transactions)
    r = build_rationale(res, fv, tamper=tamper, audit=audit)
    assert "Document check" not in r.landlord  # clean ledger
    assert "Fairness audit" not in r.landlord

    from dataclasses import replace

    edited = [replace(t, balance=t.balance + 1) if i == 3 else t for i, t in enumerate(transactions)]
    r2 = build_rationale(res, fv, tamper=check_document(edited))
    assert "Document check" in r2.landlord and "Document check" not in r2.tenant


def test_probability_is_shown_when_present(scored):
    fv, res = scored
    res.probability = 0.83
    assert "83%" in build_rationale(res, fv).tenant


def test_to_dict(scored):
    fv, res = scored
    d = build_rationale(res, fv).to_dict()
    assert set(d) == {"landlord", "tenant", "facts", "reasons"}


def test_polish_with_mock_preserves_everything(scored):
    fv, res = scored
    r = build_rationale(res, fv)
    model = MockModel()
    out = polish(r, model)
    assert out.reasons == r.reasons and out.facts == r.facts
    assert out.landlord != r.landlord and "friendlier" in out.landlord
    assert len(model.prompts) == 2


def test_polish_raises_when_numbers_dropped(scored):
    fv, res = scored
    with pytest.raises(RationaleIntegrityError):
        polish(build_rationale(res, fv), MockModel(drop_numbers=True))


def test_verify_preserved_catches_missing_reason():
    with pytest.raises(RationaleIntegrityError):
        verify_preserved("score 57", "score 57", ("a reason",))
    verify_preserved("score 57 and 12.5%", "12.5% ... score 57 a reason", ("a reason",))


def test_numbers_in():
    assert numbers_in("R5,000.00 is 25.6% of 20000") == ["5,000.00", "25.6%", "20000"]


def test_prompt_is_redacted_before_leaving():
    prompt = _prompt(f"tenant id {SA_ID} mail a@b.com phone 082 123 4567")
    assert SA_ID not in prompt and "a@b.com" not in prompt and "082 123 4567" not in prompt
    assert prompt.startswith("Rewrite")


def test_mock_model_never_sees_pii():
    r = Rationale(landlord="ok mail a@b.com", tenant="ok mail a@b.com", facts=(), reasons=())
    model = MockModel()
    polish(r, model)
    assert all("a@b.com" not in p for p in model.prompts)
    assert all("[EMAIL]" in p for p in model.prompts)
