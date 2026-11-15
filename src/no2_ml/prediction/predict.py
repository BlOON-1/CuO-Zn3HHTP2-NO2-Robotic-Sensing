import numpy as np
import pandas as pd
from no2_ml.features import feature_spec
from no2_ml.io.manifest import features_from_manifest
from no2_ml.evaluation.metrics import identifiers
from .artifacts import load_artifact


def predict_features(pipeline, info, features):
    spec=feature_spec(info['feature_version'])
    required={'sample_id','source_id','group_id','channel','feature_version','feature_fingerprint','qc_version','qc_fingerprint','quality_issues','prediction_mode',*spec.columns}
    if not required.issubset(features) or features.empty:
        raise ValueError("Prediction input fields missing or table empty")
    for key in ['sample_id','source_id','group_id']:
        identifiers(features[key],len(features),key,unique=key=='sample_id')
    if not features.feature_version.eq(spec.version).all() or not features.feature_fingerprint.eq(spec.fingerprint).all():
        raise ValueError("Prediction feature version/fingerprint mismatch")
    if not features.channel.eq(info['channel']).all():raise ValueError("Prediction channel differs from artifact")
    observed=[c for c in features if c in {'max_response','resp_60s','initial_slope','auc_0_120'}]
    if observed!=list(spec.columns):raise ValueError("Prediction feature column order differs from artifact")
    if not features.qc_version.eq(info['qc_config']['version']).all():raise ValueError("Prediction QC version mismatch")
    if not features.qc_fingerprint.eq(info['qc_fingerprint']).all():raise ValueError("Prediction QC configuration differs from training")
    X=features[list(spec.columns)].astype(float)
    if np.isinf(X.to_numpy()).any() or X.isna().all(axis=1).any():raise ValueError("Invalid or entirely missing features")
    predicted=pipeline.predict(X)
    if not np.isfinite(predicted).all():raise ValueError("Model produced nonfinite predictions")
    out=features[['sample_id','source_id','group_id','channel','feature_version','quality_issues','prediction_mode']].copy()
    out['predicted_concentration_ppm']=predicted
    out['model_name']=info['model_name']
    out['model_sha256']=info['model_sha256']
    return out


def predict_manifest(manifest, artifact, *, trusted=False, expected_feature_version=None):
    pipeline,info=load_artifact(artifact,trusted=trusted,expected_feature_version=expected_feature_version)
    features=features_from_manifest(manifest,feature_version=info['feature_version'],qc_config=info['qc_config'],require_reference=False)
    return predict_features(pipeline,info,features)
