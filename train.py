"""Train one or more model versions from feature_config.py.

Time-based split (never shuffle time series!):
  train: seasons <= test-2   |  validation: test-1 (early stopping)  |  test: last full season
Then refit on ALL labeled rows with the tree count early stopping picked, and save.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import model_registry
from config import SETTINGS
from feature_config import MODEL_CONFIGS
from features import build_features, load_raw

log = logging.getLogger(__name__)

DEFAULT_PARAMS = dict(
    n_estimators=3000,          # upper bound; early stopping picks the real number
    learning_rate=0.02,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    eval_metric="rmse",
    early_stopping_rounds=150,
    random_state=42,
)


def _metrics(y, p) -> dict:
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p))}


def train_version(raw: dict, version: str, test_season: int) -> dict:
    cfg = MODEL_CONFIGS[version]
    params = {**DEFAULT_PARAMS, **cfg.get("xgb_params", {})}
    df, features = build_features(raw, cfg, scoring=SETTINGS.scoring)
    data = df.dropna(subset=["target"])

    val_season = test_season - 1
    tr = data[data["season"] < val_season]
    va = data[data["season"] == val_season]
    te = data[data["season"] == test_season]
    log.info("%s: %d features | rows train=%d val=%d test=%d",
             version, len(features), len(tr), len(va), len(te))

    model = xgb.XGBRegressor(**params)
    model.fit(tr[features], tr["target"], eval_set=[(va[features], va["target"])], verbose=False)
    best_n = model.best_iteration + 1

    test = _metrics(te["target"], model.predict(te[features]))
    baseline = _metrics(te["target"], te["fpts_roll3"])  # "just use his last-3-game average"

    final = xgb.XGBRegressor(**{**params, "n_estimators": best_n, "early_stopping_rounds": None})
    final.fit(data[features], data["target"], verbose=False)

    importance = sorted(zip(features, final.feature_importances_.tolist()), key=lambda x: -x[1])
    meta = {
        "version": version, "model_key": cfg["model_key"], "features": features,
        "best_n_estimators": best_n, "params": {**params, "n_estimators": best_n},
        "test_season": test_season, "test": test, "baseline_roll3": baseline,
        "scoring": SETTINGS.scoring, "n_rows": int(len(data)),
        "top_features": importance[:25], "trained_at": datetime.now().isoformat(timespec="minutes"),
    }
    model_registry.save(version, final, meta)
    return meta


def main(current_season: int, versions: list[str], promote: bool = False):
    seasons = list(range(SETTINGS.first_season, current_season + 1))
    raw = load_raw(seasons, include_pbp=True)
    print(f"\n{'version':8s} {'RMSE':>6s} {'MAE':>6s} {'R2':>6s} | {'base RMSE':>9s} {'base R2':>7s}")
    for v in versions:
        m = train_version(raw, v, test_season=current_season - 1)
        t, b = m["test"], m["baseline_roll3"]
        print(f"{v:8s} {t['rmse']:6.2f} {t['mae']:6.2f} {t['r2']:6.3f} | "
              f"{b['rmse']:9.2f} {b['r2']:7.3f}   top: {[f for f, _ in m['top_features'][:5]]}")
        if promote:
            model_registry.promote(v)
