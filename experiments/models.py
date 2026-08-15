"""Model factories shared by the experiment scripts."""
import json
import os

DEFAULT_PARAMS = {
    "xgb":      {"n_estimators": 1000, "learning_rate": 0.03, "max_depth": 6,
                 "subsample": 0.8, "colsample_bytree": 0.8},
    "lgbm":     {"n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 63},
    "catboost": {"iterations": 1000, "learning_rate": 0.05, "depth": 6},
}


def tuned_params(name: str, path: str = "models/best_params.json") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get(name, DEFAULT_PARAMS[name])
    return DEFAULT_PARAMS[name]


def factory(name: str, params: dict | None = None):
    """Return a zero-arg callable producing a fresh classifier."""
    p = dict(params if params is not None else DEFAULT_PARAMS.get(name, {}))

    if name == "xgb":
        import xgboost as xgb
        return lambda: xgb.XGBClassifier(**p, random_state=42, eval_metric="auc",
                                         tree_method="hist", verbosity=0)
    if name == "lgbm":
        import lightgbm as lgb
        return lambda: lgb.LGBMClassifier(**p, random_state=42, verbose=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return lambda: CatBoostClassifier(**p, random_seed=42, eval_metric="AUC",
                                          verbose=0, allow_writing_files=False)
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        p.setdefault("max_iter", 2000)
        p.setdefault("C", 1.0)
        return lambda: LogisticRegression(**p)
    raise ValueError(f"unknown model: {name}")
