"""Offline evaluation. No API keys, no network, ~seconds.

    python eval/run.py

Reports:
  1. transaction classifier precision/recall per category vs generator labels
  2. score band vs labelled 12-month outcome (2x2 confusion, pass = strong|adequate)
  3. calibration Brier score on the personas and on a synthetic hold-out
  4. tamper detection precision/recall on 2 tampered + 12 clean documents
  5. disparate-impact ratio on random (no-signal) group splits, must be ~1.0
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from affordability.calibrate import (  # noqa: E402
    brier_score,
    default_calibrator,
    synthetic_labelled_set,
)
from affordability.fairness import disparate_impact_test, proxy_audit  # noqa: E402
from affordability.features import build_features  # noqa: E402
from affordability.parse.payslip import parse_payslip  # noqa: E402
from affordability.parse.statement import CATEGORIES, parse_statement  # noqa: E402
from affordability.parse.tamper import check_document  # noqa: E402
from affordability.score import score_features  # noqa: E402
from demo.generate import PERSONAS, generate_persona  # noqa: E402

DATA = ROOT / "demo" / "data"
PASS_BANDS = {"strong", "adequate"}


def load_manifest() -> list[dict]:
    return json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


def _labels(path: Path) -> dict[int, str]:
    with path.open(encoding="utf-8", newline="") as fh:
        return {int(r["line_no"]): r["category"] for r in csv.DictReader(fh)}


def eval_classifier(entries: list[dict]) -> dict:
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    total = correct = 0
    for e in entries:
        txns = parse_statement((DATA / e["statement"]).read_text(encoding="utf-8"))
        labels = _labels(DATA / e["labels"])
        for t in txns:
            truth = labels.get(t.line_no)
            if truth is None:
                continue
            total += 1
            if t.category == truth:
                tp[truth] += 1
                correct += 1
            else:
                fp[t.category] += 1
                fn[truth] += 1
    per_cat = {}
    for c in CATEGORIES:
        support = tp[c] + fn[c]
        if support == 0 and tp[c] + fp[c] == 0:
            continue
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / support if support else 0.0
        per_cat[c] = {"precision": p, "recall": r, "support": support}
    return {"accuracy": correct / total if total else 0.0, "n": total, "per_category": per_cat}


def eval_scores(entries: list[dict]) -> dict:
    rows = []
    for e in entries:
        txns = parse_statement((DATA / e["statement"]).read_text(encoding="utf-8"))
        fv = build_features(txns, e["proposed_rent"])
        res = score_features(fv)
        rows.append(
            {
                "id": e["id"],
                "score": res.score,
                "band": res.band,
                "outcome": bool(e["outcome_no_default_12m"]),
                "rent_to_income": fv.rent_to_income,
            }
        )
    cm = Counter()
    for r in rows:
        pred = r["band"] in PASS_BANDS
        cm[(pred, r["outcome"])] += 1
    n = len(rows)
    acc = (cm[(True, True)] + cm[(False, False)]) / n if n else 0.0
    return {"rows": rows, "confusion": dict(cm), "accuracy": acc}


def eval_calibration(score_rows: list[dict]) -> dict:
    cal = default_calibrator()
    persona_probs = [cal.predict(r["score"]) for r in score_rows]
    persona_outcomes = [int(r["outcome"]) for r in score_rows]
    _, hold_scores, hold_outcomes = synthetic_labelled_set(n=2000, seed=99)
    hold_probs = cal.predict_many(hold_scores)
    base_rate = float(np.mean(hold_outcomes))
    return {
        "brier_personas": brier_score(persona_probs, persona_outcomes),
        "brier_holdout": brier_score(hold_probs, hold_outcomes),
        "brier_holdout_baseline": brier_score([base_rate] * len(hold_outcomes), hold_outcomes),
        "n_fit": cal.n_samples,
        "n_holdout": len(hold_outcomes),
        "persona_probs": {r["id"]: round(p, 3) for r, p in zip(score_rows, persona_probs, strict=True)},
    }


def eval_tamper(entries: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    detail = []
    for e in entries:
        txns = parse_statement((DATA / e["statement"]).read_text(encoding="utf-8"))
        slip = parse_payslip((DATA / e["payslip"]).read_text(encoding="utf-8"))
        rep = check_document(txns, slip)
        flagged = rep.risk == "high"
        truth = bool(e["tampered"])
        detail.append({"id": e["id"], "risk": rep.risk, "tampered": truth, "summary": rep.summary()})
        if flagged and truth:
            tp += 1
        elif flagged and not truth:
            fp += 1
        elif not flagged and truth:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "detail": detail}


def _random_split_ratios(scores: list[int], n_splits: int, rng: np.random.Generator) -> list[float]:
    ratios = []
    for _ in range(n_splits):
        groups = rng.choice(["A", "B"], size=len(scores))
        ratios.append(disparate_impact_test(scores, list(groups)).ratio)
    return ratios


def eval_fairness(
    n_variants: int = 100, n_splits: int = 50, n_synthetic: int = 5000, seed: int = 11
) -> dict:
    """Disparate impact on groups that carry no signal.

    Two populations are split at random into "A" and "B": re-seeded variants of the
    twelve personas (real parsing path, proxy audit included) and a large synthetic
    feature-vector set. Because group membership is a coin flip, the four-fifths
    ratio should sit at ~1.0; anything else would mean the test itself is biased.
    """
    scores: list[int] = []
    excluded_total = 0
    for persona in PERSONAS:
        for k in range(n_variants):
            gen = generate_persona(persona, seed=100 + k)
            txns = parse_statement(gen.statement)
            audit = proxy_audit(txns)
            excluded_total += len(audit.excluded)
            fv = build_features(txns, persona.proposed_rent, audit=audit)
            scores.append(score_features(fv).score)
    rng = np.random.default_rng(seed)
    persona_ratios = _random_split_ratios(scores, n_splits, rng)
    _, synth_scores, _ = synthetic_labelled_set(n=n_synthetic, seed=seed)
    synth_ratios = _random_split_ratios([int(s) for s in synth_scores], n_splits, rng)
    return {
        "n_scored": len(scores),
        "mean_ratio": float(np.mean(persona_ratios)),
        "min_ratio": float(np.min(persona_ratios)),
        "pass_rate": float(np.mean([s >= 60 for s in scores])),
        "proxy_transactions_excluded": excluded_total,
        "n_synthetic": n_synthetic,
        "synthetic_mean_ratio": float(np.mean(synth_ratios)),
        "synthetic_min_ratio": float(np.min(synth_ratios)),
        "n_splits": n_splits,
    }


def main() -> int:
    manifest = load_manifest()
    clean = [e for e in manifest if not e["tampered"]]

    cls = eval_classifier(clean)
    sc = eval_scores(clean)
    cal = eval_calibration(sc["rows"])
    tam = eval_tamper(manifest)
    fair = eval_fairness()

    print("affordability-scorer eval (offline, synthetic personas)")
    print("=" * 66)
    print(f"\n1. Transaction classifier  ({cls['n']} labelled lines, accuracy {cls['accuracy']:.1%})")
    print(f"   {'category':<13}{'precision':>10}{'recall':>9}{'support':>9}")
    for c, m in cls["per_category"].items():
        print(f"   {c:<13}{m['precision']:>10.1%}{m['recall']:>9.1%}{m['support']:>9}")

    print(f"\n2. Score band vs labelled outcome  (n={len(sc['rows'])}, accuracy {sc['accuracy']:.1%})")
    cm = sc["confusion"]
    print("                       no default   default")
    print(f"   predicted pass     {cm.get((True, True), 0):>10}{cm.get((True, False), 0):>10}")
    print(f"   predicted fail     {cm.get((False, True), 0):>10}{cm.get((False, False), 0):>10}")
    for r in sc["rows"]:
        print(
            f"   {r['id']:<27} score {r['score']:>3}  {r['band']:<12} "
            f"rent/income {r['rent_to_income']:.0%}  outcome={'paid' if r['outcome'] else 'DEFAULT'}"
        )

    print(f"\n3. Calibration  (isotonic, fitted on {cal['n_fit']} synthetic labels)")
    print(f"   Brier on 12 personas          {cal['brier_personas']:.3f}")
    print(f"   Brier on {cal['n_holdout']} synthetic hold-out {cal['brier_holdout']:.3f}  "
          f"(base-rate baseline {cal['brier_holdout_baseline']:.3f})")

    print(f"\n4. Tamper detection  (risk=high on {tam['tp'] + tam['fn']} tampered + {tam['tn'] + tam['fp']} clean)")
    print(f"   precision {tam['precision']:.0%}   recall {tam['recall']:.0%}   "
          f"tp={tam['tp']} fp={tam['fp']} fn={tam['fn']} tn={tam['tn']}")
    for d in tam["detail"]:
        if d["tampered"] or d["risk"] != "low":
            print(f"   {d['id']:<27} risk={d['risk']:<7} {d['summary']}")

    print(f"\n5. Disparate impact on random (no-signal) groups  ({fair['n_splits']} splits each)")
    print(f"   {fair['n_scored']} persona variants   mean ratio {fair['mean_ratio']:.3f}   "
          f"min ratio {fair['min_ratio']:.3f}   pass rate {fair['pass_rate']:.0%}   "
          f"proxy lines excluded {fair['proxy_transactions_excluded']}")
    print(f"   {fair['n_synthetic']} synthetic vectors  mean ratio {fair['synthetic_mean_ratio']:.3f}   "
          f"min ratio {fair['synthetic_min_ratio']:.3f}")

    results = {"classifier": cls, "scores": sc, "calibration": cal, "tamper": tam, "fairness": fair}
    results["scores"]["confusion"] = {f"{k[0]}_{k[1]}": v for k, v in sc["confusion"].items()}
    (ROOT / "eval" / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    failures = []
    if tam["recall"] < 1.0:
        failures.append("tamper recall below 100%")
    if tam["fp"] > 0:
        failures.append("tamper false positive on a clean persona")
    if fair["mean_ratio"] < 0.9 or fair["synthetic_mean_ratio"] < 0.95:
        failures.append("disparate-impact ratio drifted from 1.0 on no-signal groups")
    if cls["accuracy"] < 0.9:
        failures.append("classifier accuracy below 90%")
    if failures:
        print("\nFAIL: " + "; ".join(failures))
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
