# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] - 2026-10-14

### Fixed
- A clipped score was not explained: the points range from -135 to +70 around a base of 50,
  so a strong applicant's listed points added up to 120 under a stated score of 100. Both
  rationales now carry a fact line with the uncapped total whenever the 0-100 clip applies.
- Synthetic SA ID numbers (generator and tests) used random plausible birth dates and
  sequence numbers, so some were likely real people's ID numbers. They now use 30/31
  February, which passes `is_sa_id` (so redaction is still exercised) but is no one's birth
  date. The generator keeps its RNG draws, so only the ID line of each payslip changed and
  every eval number is identical.
- README Python example pointed at `statement.csv` / `payslip.txt` that do not exist; it now
  uses the teacher persona (output `(74, 'adequate', 'low')` verified).
- README said the `explain()` rows "sum to the score"; they sum to the raw points.

### Changed
- README Results section states what each number measures (generator vocabulary,
  author-assigned outcome labels, synthetic calibration, random no-signal groups) and that
  none is a validation on real applicants. Standard footer.
- Ruff rule set widened to `E, F, W, I, B, UP, RUF` (E501 ignored, with the reason in
  `pyproject.toml`); 16 automatic fixes (typing imports, `datetime.UTC`). Lint now runs first
  in CI.
- Added `demo/README.md`, `CITATION.cff`, this changelog, `tests/test_regressions.py`.

## [0.1.0] - 2026-10-09

### Added
- Initial release: statement and payslip parsers for four SA layouts, tamper checks,
  money-only `FeatureVector`, proxy audit, points-table scorer with `explain()`, isotonic
  calibration on synthetic labels, POPIA redaction and access log, landlord and tenant
  rationales, CLI, seeded synthetic demo set and offline eval.
