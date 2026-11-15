from sklearn.model_selection import GridSearchCV
from no2_ml.models.registry import build_pipeline
from no2_ml.validation.splits import grouped_splits


def search_model(name, X, y, groups, *, config=None, seed=42, inner_folds=3, scoring="neg_mean_absolute_error"):
    if scoring not in {"neg_mean_absolute_error","neg_mean_squared_error","neg_root_mean_squared_error"}:
        raise ValueError("Search scoring must be a supported regression error score")
    pipe,grid=build_pipeline(name,config,seed=seed)
    splits=grouped_splits(groups,strategy="group_kfold",n_splits=inner_folds)
    search=GridSearchCV(pipe,grid or [{}],cv=splits,scoring=scoring,n_jobs=1,error_score="raise",return_train_score=False)
    search.fit(X,y)
    return search
