"""Nested grouped evaluation and a separate full-data candidate refit."""
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from no2_ml.features.versions import feature_spec
from no2_ml.evaluation import evaluate_predictions
from no2_ml.evaluation.metrics import identifiers
from no2_ml.validation.splits import grouped_splits, ensure_group_disjoint
from .search import search_model


def validate_feature_table(table, version):
    spec=feature_spec(version)
    required={"sample_id","source_id","group_id","channel","reference_ppm","feature_version","feature_fingerprint","qc_version","qc_fingerprint",*spec.columns}
    if not required.issubset(table) or table.empty:
        raise ValueError(f"Feature table missing fields: {sorted(required-set(table))}")
    for name in ["sample_id","source_id","group_id"]:
        identifiers(table[name],len(table),name,unique=name=="sample_id")
    if not table.feature_version.eq(spec.version).all() or not table.feature_fingerprint.eq(spec.fingerprint).all():
        raise ValueError("Feature version/fingerprint mismatch")
    if table.qc_version.nunique()!=1 or table.qc_fingerprint.nunique()!=1 or table[['qc_version','qc_fingerprint']].isna().any().any():
        raise ValueError("Training rows must use one explicit QC definition/configuration")
    if table.channel.isna().any() or table.channel.nunique()!=1 or not (table.channel.astype(float)%1==0).all() or (table.channel.astype(float)<1).any():
        raise ValueError("One explicit positive integer channel per training run is required")
    if not table.groupby('source_id').group_id.nunique().eq(1).all():
        raise ValueError("One original source must belong to one group")
    if "source_sha256" in table and not table.groupby('source_sha256').group_id.nunique().eq(1).all():
        raise ValueError("Identical raw sources cross groups")
    X=table.loc[:,list(spec.columns)].astype(float)
    y=table.reference_ppm.to_numpy(float)
    if np.isinf(X.to_numpy()).any() or X.isna().all(axis=1).any() or X.isna().all(axis=0).any():
        raise ValueError("Infinite or entirely missing features; no fabricated signal defaults")
    if not np.isfinite(y).all() or (y<0).any():
        raise ValueError("Reference ppm must be finite and nonnegative")
    return spec,X,y,table.group_id.astype(str).to_numpy()


@dataclass
class NestedResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    splits: pd.DataFrame
    searches: list
    inner_splits: pd.DataFrame
    outer_estimators: dict


def nested_validate(table, *, feature_version, models, model_configs=None, validation=None, seed=42):
    spec,X,y,groups=validate_feature_table(table,feature_version)
    if not models or len(set(models))!=len(models) or 'Nested_selection' in models:
        raise ValueError("Choose a nonempty unique list of registered models")
    cfg=validation or {}
    allowed={"outer_strategy","outer_folds","inner_folds","scoring"}
    if set(cfg)-allowed:raise ValueError("Unknown validation configuration")
    inner=cfg.get('inner_folds',3)
    outer=grouped_splits(groups,strategy=cfg.get('outer_strategy','leave_one_group_out'),n_splits=cfg.get('outer_folds',3))
    # Validate all folds before any fit.
    for tr,te in outer:
        grouped_splits(groups[tr],strategy='group_kfold',n_splits=inner)
        ensure_group_disjoint(groups[tr],groups[te])
    predictions={name:np.full(len(y),np.nan) for name in models}
    predictions['Nested_selection']=np.full(len(y),np.nan)
    selected=np.empty(len(y),object)
    fold_ids=np.empty(len(y),object)
    logs=[];splitrows=[];innerrows=[];estimators={}
    for fold,(tr,te) in enumerate(outer):
        fold_ids[te]=str(fold)
        for role,idx in [('outer_train',tr),('outer_test',te)]:
            for i in idx:
                splitrows.append(dict(outer_fold=str(fold),role=role,sample_id=str(table.iloc[i].sample_id),source_id=table.iloc[i].source_id,group_id=groups[i]))
        for j,(itr,iva) in enumerate(grouped_splits(groups[tr],strategy='group_kfold',n_splits=inner)):
            ensure_group_disjoint(groups[tr[itr]],groups[tr[iva]])
            ensure_group_disjoint(np.r_[groups[tr[itr]],groups[tr[iva]]],groups[te])
            innerrows.append(dict(outer_fold=str(fold),inner_fold=str(j),train_groups=sorted(set(groups[tr[itr]])),
                                  validation_groups=sorted(set(groups[tr[iva]])),test_groups=sorted(set(groups[te]))))
        choices=[]
        for name in models:
            search=search_model(name,X.iloc[tr],y[tr],groups[tr],config=(model_configs or {}).get(name),seed=seed,
                                inner_folds=inner,scoring=cfg.get('scoring','neg_mean_absolute_error'))
            predictions[name][te]=search.predict(X.iloc[te])
            choices.append((search.best_score_,name))
            estimators[(fold,name)]=search.best_estimator_
            logs.append(dict(scope='outer_training_inner_cv',outer_fold=str(fold),model_name=name,
                best_score=float(search.best_score_),best_params=search.best_params_,
                full_estimator_params=search.best_estimator_.named_steps['regressor'].get_params(),cv_results=search.cv_results_))
        _,winner=max(choices,key=lambda v:v[0])
        selected[te]=winner
        predictions['Nested_selection'][te]=predictions[winner][te]
    rows=[]
    for name,p in predictions.items():
        if not np.isfinite(p).all():raise ValueError("Incomplete/nonfinite OOF predictions")
        for i in range(len(y)):
            row=dict(sample_id=str(table.iloc[i].sample_id),source_id=table.iloc[i].source_id,
                group_id=groups[i],channel=int(table.iloc[i].channel),fold_id=fold_ids[i],
                y_true=float(y[i]),y_pred=float(p[i]),error_ppm=float(p[i]-y[i]),model_name=name,
                feature_version=spec.version,evaluation_scope='pooled_oof',dataset_split='outer_test',
                selected_model=selected[i] if name=='Nested_selection' else name)
            rows.append(row)
    d=pd.DataFrame(rows)
    return NestedResult(d,evaluate_predictions(d),pd.DataFrame(splitrows),logs,pd.DataFrame(innerrows),estimators)


def refit_final(table, *, feature_version, models, model_configs=None, validation=None, seed=42):
    """Select on full-data grouped inner CV, then fit. No test metrics returned."""
    spec,X,y,groups=validate_feature_table(table,feature_version)
    cfg=validation or {}
    choices=[];logs=[]
    for name in models:
        search=search_model(name,X,y,groups,config=(model_configs or {}).get(name),seed=seed,
                            inner_folds=cfg.get('inner_folds',3),scoring=cfg.get('scoring','neg_mean_absolute_error'))
        choices.append((search.best_score_,name,search.best_estimator_))
        logs.append(dict(scope='full_data_grouped_cv_refit',model_name=name,best_score=float(search.best_score_),
                         best_params=search.best_params_,cv_results=search.cv_results_))
    _,name,pipe=max(choices,key=lambda v:v[0])
    return name,pipe,logs
