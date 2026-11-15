# Implementation record

## Stage A — 2026-11-15
- Created an independent src-layout package; no historical code or research files modified.
- Added unit-aware, encoding-aware read-only instrument parser, canonical schema,
  versioned QC/candidate detection, reviewed boundary gate and two immutable feature sets.
- Tests: `python -m unittest tests.test_stage_a -v`: 7 passed in the reference environment.
- Initial invocation used the wrong relative PYTHONPATH and failed to import; corrected
  the command to resolve `src` from the new project root. No software test failure remained.
- Final suite strengthens A to 9 tests, including ceiling rejection and unapproved candidates.

## Stage B
- Added one seven-metric evaluator, explicit validity/reason metadata, strict comparison
  alignment and separate per-fold, fold-mean/SD and pooled-OOF outputs.
- Tests: `python -m unittest tests.test_stage_b -v`: 7 passed.

## Stage C
- Added five-model registry, fold-local preprocessing/search, nested grouped OOF,
  separate grouped final refit, explicit raw manifests, hashed artifacts and shared inference.
- Tests: `python -m unittest tests.test_stage_c -v`: 6 passed, including five-family
  artificial-data nested CV, scaler fit isolation, raw/feature inference equality,
  save/reload equality, channel/version rejection and pre-load checksum rejection.

## Stage D
- Added CLI/config files, explicit run/export status, read-only saved-result rendering,
  new-project compatibility wrappers, pinned public dependencies, README and five method docs.
- Stage D tests: 5 passed. First attempt detected ~4e-16 CSV round-trip differences;
  corrected the reader to round_trip and used explicit 1e-12 numerical comparison tolerance.
- Final suite: 27 tests passed (A=9, B=7, C=6, D=5); zero remaining failures/errors.
- Built a wheel with existing setuptools; installed only this package under tests/_build/
  with --no-index --no-deps. Verified installed-package prediction and legacy-alias rendering.
  Dependencies and system site-packages were unchanged; clean-environment installation unverified.
- All generated waveform fixtures, test-trained models, metrics and renderings remain in tests/.
