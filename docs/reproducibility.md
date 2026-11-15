# Reproducibility and release boundaries

Reference environment: see requirements.txt and REFACTORING_REPORT.md. CPU traditional
models do not require CUDA. No dependency was installed or upgraded during this refactor.
The standard-library unittest suite exercises only artificial software fixtures under tests.
An isolated wheel/package-target check does not demonstrate a fresh dependency installation.

Artifact directory:
- pipeline.joblib: fitted Imputer/Scaler/regressor.
- metadata.json: schema, model, ordered feature version/fingerprint/full definition,
  QC config/fingerprint, channel, canonical units, training config, complete estimator parameters,
  Python/package versions, package/code hash, UTC time and model SHA-256.

Only explicitly trusted artifacts may be loaded. Metadata, units, feature contract,
environment and checksum are checked before joblib loading. Hashes do not protect
against an attacker replacing both metadata and model; establish origin separately.
Exact environment equality is intentionally conservative; cross-version loading is
not silently attempted. No repository commit ID is asserted when unavailable; a source
hash and package version are recorded instead.
Nonfinite estimator defaults (for example XGBoost's missing=NaN sentinel) are serialized
as an explicit nonfinite_float object, not silently changed to a numerical default.

All public output helpers use exclusive file/directory creation. A failed run leaves a
failed status rather than overwriting old results. Rendering consumes saved predictions,
recalculates metrics only for consistency, and verifies source bytes are unchanged.
It never executes training or changes predictions.

Legacy code remains outside this project, unchanged. Old result feature versions are
not relabeled as new method versions. No original data/model/figure is redistributed.
Before actual research use confirm identities, provenance, groups, approved boundaries,
reference concentrations, supported time windows, permitted license and data sharing.

License and manuscript DOI remain pending. GitHub upload and formal Wiley checklist
sign-off require owner review; software tests cannot establish experimental validity.
