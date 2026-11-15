import numpy as np
import pandas as pd
from .metrics import evaluate_regression

REQUIRED = {"sample_id","model_name","feature_version","y_true","y_pred","fold_id","group_id","evaluation_scope"}


def compare_models(predictions, *, align_by_id=False):
    """Validate identical IDs/references/folds/groups/scopes before comparisons.

    Order differences are rejected unless alignment is explicitly requested.
    """
    d = predictions.copy()
    if not REQUIRED.issubset(d.columns) or d.empty:
        raise ValueError(f"Prediction schema requires {sorted(REQUIRED)}")
    if d[list(REQUIRED)].isna().any().any():
        raise ValueError("Missing prediction fields")
    baseline = None
    outputs = []
    for name,part in d.groupby("model_name",sort=False):
        if part.sample_id.duplicated().any() or part.feature_version.nunique()!=1 or part.evaluation_scope.nunique()!=1:
            raise ValueError("Duplicate samples or mixed feature version/evaluation scope")
        if align_by_id:
            part=part.sort_values("sample_id")
        key = part[["sample_id","y_true","fold_id","group_id","feature_version","evaluation_scope"]].reset_index(drop=True)
        if baseline is None:
            baseline=key
        elif not key.equals(baseline):
            raise ValueError("Models differ in test sample/order/reference/fold/group/feature version/scope")
        outputs.append(part)
    return pd.concat(outputs,ignore_index=True)


def evaluate_predictions(predictions, *, align_by_id=False):
    d = compare_models(predictions,align_by_id=align_by_id)
    rows = []
    for name,part in d.groupby("model_name",sort=False):
        scope = part.evaluation_scope.iloc[0]
        common = dict(model_name=name,feature_version=part.feature_version.iloc[0])
        if scope not in {"pooled_oof","single_test_set"}:
            raise ValueError("Prediction table must represent pooled_oof or single_test_set")
        result=evaluate_regression(part.y_true,part.y_pred,sample_id=part.sample_id,
                                  evaluation_scope=scope,fold_id=part.fold_id,group_id=part.group_id,**common)
        rows.extend(dict(r,fold_id="ALL",n_groups=part.group_id.nunique()) for r in result.rows())
        if scope=="single_test_set":
            continue
        foldrows=[]
        for fold,foldpart in part.groupby("fold_id",sort=False):
            result=evaluate_regression(foldpart.y_true,foldpart.y_pred,sample_id=foldpart.sample_id,
                evaluation_scope="per_fold",fold_id=foldpart.fold_id,group_id=foldpart.group_id,**common)
            foldrows.extend(dict(r,fold_id=str(fold),n_groups=foldpart.group_id.nunique()) for r in result.rows())
        rows.extend(foldrows)
        fd=pd.DataFrame(foldrows)
        for metric,g in fd.groupby("name",sort=False):
            values=g.loc[g.status.eq("valid"),"value"].astype(float)
            for aggregate in ["fold_mean","fold_sd"]:
                valid=len(values)>0 and (aggregate=="fold_mean" or len(values)>1)
                complete=len(values)==len(g)
                value=float(values.mean() if aggregate=="fold_mean" else values.std(ddof=1)) if valid else None
                rows.append(dict(name=metric,value=value,unit=g.unit.iloc[0],
                    status=("valid" if complete else "partial") if valid else "undefined",
                    reason="" if valid and complete else "Undefined/inapplicable folds excluded explicitly; see n_valid_folds",
                    n_samples=len(part),model_name=name,feature_version=common["feature_version"],
                    evaluation_scope=aggregate,fold_id="ALL",n_groups=part.group_id.nunique(),
                    n_folds=len(g),n_valid_folds=len(values),bias_definition="mean(y_pred - y_true)"))
    return pd.DataFrame(rows)
