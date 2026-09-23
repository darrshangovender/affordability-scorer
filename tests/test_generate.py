"""The demo generator is deterministic and its commission persona is internally consistent."""

import json
from pathlib import Path

import pytest

from affordability.parse.payslip import parse_payslip
from affordability.parse.statement import parse_statement
from affordability.parse.tamper import check_document
from affordability.privacy import is_sa_id
from demo.generate import PERSONAS, generate_persona, write_all


def _snapshot(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in root.rglob("*") if p.is_file()}


def test_write_all_is_deterministic(tmp_path):
    write_all(tmp_path / "a")
    write_all(tmp_path / "b")
    assert _snapshot(tmp_path / "a") == _snapshot(tmp_path / "b")


def test_committed_data_matches_generator(tmp_path, demo_dir):
    write_all(tmp_path / "fresh")
    assert _snapshot(tmp_path / "fresh") == _snapshot(demo_dir)


def test_manifest_shape(tmp_path):
    manifest = write_all(tmp_path)
    assert len(manifest) == 14
    assert sum(e["tampered"] for e in manifest) == 2
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_commission_persona_gross_scales_with_net():
    persona = next(p for p in PERSONAS if p.variable_income)
    gen = generate_persona(persona)
    slip = parse_payslip(gen.payslip)
    assert slip.gross / slip.net == pytest.approx(persona.gross / persona.net, rel=1e-3)
    assert slip.net != persona.net  # the pay month really was a variable one
    txns = parse_statement(gen.statement)
    salaries = [t.amount for t in txns if t.category == "salary"]
    assert 0.6 * persona.net <= min(salaries) and max(salaries) <= 1.5 * persona.net
    assert check_document(txns, slip).ok


def test_generated_ids_are_luhn_valid_fictions():
    for persona in PERSONAS:
        assert is_sa_id(generate_persona(persona).id_number)
