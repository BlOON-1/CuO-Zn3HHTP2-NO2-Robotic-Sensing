"""Audited legacy estimators; optional imports occur only on explicit model construction."""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

DEFAULTS = {
    "Linear": {"params":{}, "search":{}},
    "Ridge": {"params":{}, "search":{"alpha":[1,10]}},
    "SVR": {"params":{}, "search":{"C":[1,10]}},
    "RandomForest": {"params":{"n_estimators":60,"min_samples_leaf":2,"n_jobs":1},"search":{"max_depth":[2,None]}},
    "XGBoost": {"params":{"n_estimators":60,"max_depth":2,"learning_rate":.05,"objective":"reg:squarederror","n_jobs":1},"search":{"reg_lambda":[1,10]}},
}


def model_names():
    return tuple(DEFAULTS)


def build_pipeline(name, config=None, *, seed=42):
    if name not in DEFAULTS:
        raise ValueError(f"Unregistered model {name}; supported: {model_names()}")
    constructors = {"Linear":LinearRegression,"Ridge":Ridge,"SVR":SVR,"RandomForest":RandomForestRegressor}
    if name=="XGBoost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("XGBoost dependency missing; environment unchanged") from exc
        constructors[name]=XGBRegressor
    settings = config if config is not None else DEFAULTS[name]
    if set(settings)-{"params","search"}:
        raise ValueError("Unknown model configuration keys")
    params=dict(DEFAULTS[name]["params"])
    if name in {"RandomForest","XGBoost"}:
        params["random_state"]=seed
    params.update(settings.get("params",{}))
    estimator=constructors[name](**params)
    pipeline=Pipeline([("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),
                       ("scaler",StandardScaler()),("regressor",estimator)])
    grid={}
    for param,values in settings.get("search",DEFAULTS[name]["search"]).items():
        if param not in estimator.get_params() or not isinstance(values,list) or not values:
            raise ValueError(f"Invalid search parameter/values: {param}")
        grid[f"regressor__{param}"]=values
    return pipeline,grid
