"""Tiny model registry: save versions side by side, point the live agent at one per slot.

models/
  active.json                 {"WR": "WR_v1", "RB": "RB_v2", "QB": "QB_v1"}
  WR_v1/model.json, meta.json
  RB_v2/model.json, meta.json
"""
from __future__ import annotations

import json

import xgboost as xgb

from config import MODEL_DIR
from feature_config import DEFAULT_ACTIVE, MODEL_CONFIGS

ACTIVE_FILE = MODEL_DIR / "active.json"


def save(version: str, model: xgb.XGBRegressor, meta: dict):
    d = MODEL_DIR / version
    d.mkdir(parents=True, exist_ok=True)
    model.save_model(d / "model.json")
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def load(version: str) -> tuple[xgb.XGBRegressor, dict]:
    d = MODEL_DIR / version
    model = xgb.XGBRegressor()
    model.load_model(d / "model.json")
    return model, json.loads((d / "meta.json").read_text())


def active() -> dict[str, str]:
    if ACTIVE_FILE.exists():
        return {**DEFAULT_ACTIVE, **json.loads(ACTIVE_FILE.read_text())}
    return dict(DEFAULT_ACTIVE)


def promote(version: str):
    if not (MODEL_DIR / version / "model.json").exists():
        raise FileNotFoundError(f"{version} has not been trained yet")
    slots = active()
    slots[MODEL_CONFIGS[version]["model_key"]] = version
    ACTIVE_FILE.write_text(json.dumps(slots, indent=2))
    return slots


def list_versions() -> list[dict]:
    live = set(active().values())
    rows = []
    for d in sorted(MODEL_DIR.iterdir()):
        if (d / "meta.json").exists():
            m = json.loads((d / "meta.json").read_text())
            rows.append({"version": d.name, "slot": m["model_key"], "active": d.name in live,
                         "test_rmse": m["test"]["rmse"], "test_r2": m["test"]["r2"],
                         "baseline_rmse": m["baseline_roll3"]["rmse"],
                         "n_features": len(m["features"]), "trained": m["trained_at"]})
    return rows
