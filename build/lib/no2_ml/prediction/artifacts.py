"""Joblib is executable serialization: explicit trust required before loading."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import platform
import numpy as np
import joblib
from no2_ml import __version__
from no2_ml.features import feature_spec
from no2_ml.preprocessing.quality import QCConfig


def jsonable(value):
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [jsonable(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else {"nonfinite_float":str(float(value))}
    if value is None or isinstance(value,(str,bool,int)):return value
    return str(value)


def environment():
    return dict(python=platform.python_version(),packages={p:metadata.version(p) for p in
        ['numpy','pandas','scipy','scikit-learn','xgboost','matplotlib','joblib','PyYAML']})


def source_code_hash():
    base=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for path in sorted(base.rglob('*.py')):
        digest.update(path.relative_to(base).as_posix().encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def save_artifact(pipeline, directory, *, model_name, feature_version, channel, training_config, qc_config=None):
    spec=feature_spec(feature_version)
    if list(getattr(pipeline,'feature_names_in_',[]))!=list(spec.columns):
        raise ValueError("Fitted model feature order does not match declared version")
    if model_name not in {'Linear','Ridge','SVR','RandomForest','XGBoost'}:
        raise ValueError("Unregistered artifact model")
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    model=directory/'pipeline.joblib'
    joblib.dump(pipeline,model)
    info=dict(artifact_schema='no2_pipeline@1',model_name=model_name,feature_version=spec.version,
        feature_fingerprint=spec.fingerprint,feature_spec=asdict(spec),feature_order=list(spec.columns),
        channel=int(channel),time_unit='s',resistance_unit='ohm',concentration_unit='ppm',
        qc_config=asdict(QCConfig(**(qc_config or {}))),qc_fingerprint=QCConfig(**(qc_config or {})).fingerprint,training_config=training_config,
        environment=environment(),model_parameters=pipeline.named_steps['regressor'].get_params(deep=True),
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        created_utc=datetime.now(timezone.utc).isoformat(),package_version=__version__,code_sha256=source_code_hash(),
        prediction_mode='offline_reviewed_cycle',selection='full-data grouped inner-CV refit, separate from OOF evaluation')
    (directory/'metadata.json').write_text(json.dumps(jsonable(info),indent=2,allow_nan=False),encoding='utf8')
    return info


def load_artifact(directory, *, trusted=False, expected_feature_version=None):
    if trusted is not True:
        raise ValueError("Only load a trusted artifact; pass trusted=True explicitly")
    directory=Path(directory)
    info=json.loads((directory/'metadata.json').read_text(encoding='utf8'))
    spec=feature_spec(info['feature_version'])
    if expected_feature_version is not None and feature_spec(expected_feature_version).version!=spec.version:
        raise ValueError("Requested feature version differs from model artifact")
    if info.get('artifact_schema')!='no2_pipeline@1' or info.get('feature_fingerprint')!=spec.fingerprint or info.get('feature_order')!=list(spec.columns):
        raise ValueError("Artifact schema/feature contract mismatch")
    if (info.get('time_unit'),info.get('resistance_unit'),info.get('concentration_unit'))!=('s','ohm','ppm'):
        raise ValueError("Artifact units mismatch")
    if QCConfig(**info['qc_config']).fingerprint!=info.get('qc_fingerprint'):
        raise ValueError("Artifact QC configuration fingerprint mismatch")
    if info.get('environment')!=environment():
        raise ValueError("Artifact software environment differs; compatibility must be reviewed explicitly")
    model=directory/'pipeline.joblib'
    if hashlib.sha256(model.read_bytes()).hexdigest()!=info['model_sha256']:
        raise ValueError("Artifact checksum mismatch before deserialization")
    pipe=joblib.load(model)
    if list(getattr(pipe,'feature_names_in_',[]))!=list(spec.columns):
        raise ValueError("Serialized pipeline feature schema mismatch")
    return pipe,info
