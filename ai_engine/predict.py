from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


def load_model_and_metadata(
    artifacts_dir: str = "ai_engine/model_artifacts",
) -> tuple[object, dict]:
    artifacts_path = Path(artifacts_dir)
    model = joblib.load(artifacts_path / "fraud_rf_model.joblib")
    metadata = json.loads((artifacts_path / "fraud_rf_metadata.json").read_text(encoding="utf-8"))
    return model, metadata


def score_transactions(df: pd.DataFrame, artifacts_dir: str = "ai_engine/model_artifacts") -> pd.DataFrame:
    """
    Score transactions and return a copy with prediction columns.
    Expects DataFrame columns to match training feature names.
    """
    model, metadata = load_model_and_metadata(artifacts_dir=artifacts_dir)
    feature_names = metadata["feature_names"]

    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    X = df[feature_names].copy()
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    output = df.copy()
    output["fraud_probability"] = proba
    output["is_fraud_pred"] = pred.astype(int)
    return output
