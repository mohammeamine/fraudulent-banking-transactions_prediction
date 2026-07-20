from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class PreparedDataset:
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]


def load_training_csv(csv_path: str) -> pd.DataFrame:
    """Load the training CSV file into a DataFrame."""
    return pd.read_csv(csv_path)


def prepare_features(
    df: pd.DataFrame,
    target_column: str = "is_fraud",
    drop_columns: Iterable[str] | None = None,
) -> PreparedDataset:
    """
    Prepare model inputs.

    The clean dataset already contains numeric engineered features, so we only:
    - validate required columns
    - drop optional non-feature columns when provided
    - cast target to integer labels
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    drop_set = set(drop_columns or [])
    candidate_features = [c for c in df.columns if c != target_column and c not in drop_set]
    if not candidate_features:
        raise ValueError("No feature columns available after dropping columns.")

    X = df[candidate_features].copy()
    y = df[target_column].astype(int).copy()

    if X.isnull().any().any():
        raise ValueError("Input features contain missing values. Clean data before training.")

    return PreparedDataset(X=X, y=y, feature_names=candidate_features)
