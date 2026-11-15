"""One numerical definition per metric; no silent clipping or sample deletion."""
from dataclasses import dataclass, asdict
import numpy as np

SCOPES = {"single_test_set", "per_fold", "pooled_oof", "training_diagnostic"}


@dataclass(frozen=True)
class Metric:
    name: str
    value: object
    unit: str
    status: str = "valid"
    reason: str = ""


@dataclass
class Evaluation:
    metrics: list
    sample_id: tuple
    model_name: str
    feature_version: str
    evaluation_scope: str
    fold_id: object = None
    group_id: object = None

    def rows(self):
        return [dict(**asdict(metric), n_samples=len(self.sample_id), model_name=self.model_name,
                     feature_version=self.feature_version, evaluation_scope=self.evaluation_scope,
                     bias_definition="mean(y_pred - y_true)") for metric in self.metrics]

    def metric(self, name):
        return next(m for m in self.metrics if m.name == name)


def identifiers(values, n, name, *, unique=False):
    a = np.asarray(values, dtype=object)
    if a.ndim!=1 or len(a)!=n or any(v is None or str(v).strip()=="" or str(v).lower()=="nan" for v in a):
        raise ValueError(f"{name}: missing identifiers or wrong shape/length")
    out = tuple(str(v) for v in a)
    if unique and len(set(out)) != n:
        raise ValueError(f"{name}: duplicate identifiers")
    return out


def evaluate_regression(y_true, y_pred, *, sample_id, model_name, feature_version,
                        evaluation_scope, fold_id=None, group_id=None,
                        prediction_sample_id=None, adjusted_context=None):
    y, p = np.asarray(y_true,float), np.asarray(y_pred,float)
    if y.ndim!=1 or p.shape!=y.shape or not len(y) or not np.isfinite(np.r_[y,p]).all():
        raise ValueError("Targets/predictions must be nonempty finite 1-D arrays of identical length")
    if (y<0).any():
        raise ValueError("Reference NO2 concentration must be nonnegative (ppm)")
    if evaluation_scope not in SCOPES or not model_name or not feature_version:
        raise ValueError("Explicit supported evaluation_scope, model_name and feature_version required")
    n = len(y)
    ids = identifiers(sample_id,n,"sample_id",unique=True)
    if prediction_sample_id is not None and identifiers(prediction_sample_id,n,"prediction_sample_id",unique=True)!=ids:
        raise ValueError("Prediction sample order differs from reference sample order")
    folds = identifiers(fold_id,n,"fold_id") if fold_id is not None else None
    groups = identifiers(group_id,n,"group_id") if group_id is not None else None
    if evaluation_scope == "pooled_oof" and (folds is None or groups is None or len(set(folds))<2):
        raise ValueError("pooled_oof requires >=2 fold IDs and group IDs")
    if evaluation_scope == "per_fold" and (folds is None or len(set(folds))!=1):
        raise ValueError("per_fold requires exactly one fold")
    if groups is not None and folds is not None:
        for group in set(groups):
            if len({folds[i] for i,g in enumerate(groups) if g==group})!=1:
                raise ValueError("A source group crosses outer test folds")
    e = p-y
    mse = float(np.mean(e**2))
    if not np.isfinite(mse):
        raise ValueError("Squared-error overflow")
    metrics = [Metric("mse",mse,"ppm^2"), Metric("mae",float(np.mean(abs(e))),"ppm"),
               Metric("rmse",float(np.sqrt(mse)),"ppm"),
               Metric("mean_bias_error",float(np.mean(e)),"ppm")]
    if (p<0).any():
        metrics.append(Metric("rmsle",None,"1","not_applicable","Negative prediction; no clipping was applied"))
    else:
        metrics.append(Metric("rmsle",float(np.sqrt(np.mean((np.log1p(p)-np.log1p(y))**2))),"1"))
    sst = float(np.sum((y-y.mean())**2))
    if not np.isfinite(sst):raise ValueError("Target variation overflow")
    if n<2 or sst==0:
        r2 = Metric("r2",None,"1","undefined","n < 2 or constant reference target")
    else:
        r2 = Metric("r2",float(1-np.sum(e**2)/sst),"1")
    metrics.append(r2)
    ctx = adjusted_context or {}
    p_count = ctx.get("p")
    if model_name!="Linear" or evaluation_scope!="training_diagnostic" or ctx.get("fit_intercept") is not True or ctx.get("full_rank") is not True:
        adj = Metric("adjusted_r2",None,"1","not_applicable","Requires full-rank unpenalized Linear training diagnostic with intercept; not an OOF complexity correction")
    elif not isinstance(p_count,int) or isinstance(p_count,bool) or p_count<1:
        raise ValueError("Adjusted R2 context requires integer predictor count p >= 1")
    elif n<=p_count+1 or r2.status!="valid":
        adj = Metric("adjusted_r2",None,"1","undefined","n <= p + 1 or R2 undefined")
    else:
        adj = Metric("adjusted_r2",float(1-(1-r2.value)*(n-1)/(n-p_count-1)),"1")
    metrics.append(adj)
    return Evaluation(metrics,ids,model_name,feature_version,evaluation_scope,folds,groups)
