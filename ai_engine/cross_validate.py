from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ai_engine.preprocessing import load_training_csv, prepare_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 5-fold cross-validation on training dataset.")
    parser.add_argument(
        "--csv-path",
        default="dataset/clean/fraudTrain_clean.csv",
        help="Path to training CSV file.",
    )
    parser.add_argument(
        "--target-column",
        default="is_fraud",
        help="Target label column name.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees for RandomForest.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Max tree depth. Use None for unlimited depth.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row sample for faster experiments.",
    )
    parser.add_argument(
        "--output-json",
        default="ai_engine/model_artifacts/cv_metrics.json",
        help="Where to save fold and average metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading dataset: {args.csv_path}")
    df = load_training_csv(args.csv_path)
    if args.sample_size is not None and args.sample_size > 0:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=args.random_state)
        print(f"Using sampled rows: {len(df)}")
    else:
        print(f"Using full rows: {len(df)}")

    prepared = prepare_features(df, target_column=args.target_column)
    X, y = prepared.X, prepared.y

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    fold_metrics: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        metrics = {
            "fold": fold_idx,
            "roc_auc": float(roc_auc_score(y_val, y_proba)),
            "precision_fraud": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall_fraud": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1_fraud": float(f1_score(y_val, y_pred, zero_division=0)),
        }
        fold_metrics.append(metrics)
        print(
            f"Fold {fold_idx}: "
            f"AUC={metrics['roc_auc']:.4f}, "
            f"P={metrics['precision_fraud']:.4f}, "
            f"R={metrics['recall_fraud']:.4f}, "
            f"F1={metrics['f1_fraud']:.4f}"
        )

    avg_metrics = {
        "roc_auc": float(np.mean([m["roc_auc"] for m in fold_metrics])),
        "precision_fraud": float(np.mean([m["precision_fraud"] for m in fold_metrics])),
        "recall_fraud": float(np.mean([m["recall_fraud"] for m in fold_metrics])),
        "f1_fraud": float(np.mean([m["f1_fraud"] for m in fold_metrics])),
    }

    print("\nAverage metrics across folds")
    print("----------------------------")
    print(f"ROC-AUC:        {avg_metrics['roc_auc']:.4f}")
    print(f"Precision(1):   {avg_metrics['precision_fraud']:.4f}")
    print(f"Recall(1):      {avg_metrics['recall_fraud']:.4f}")
    print(f"F1(1):          {avg_metrics['f1_fraud']:.4f}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "n_rows_used": int(len(df)),
                "n_splits": args.n_splits,
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "fold_metrics": fold_metrics,
                "average_metrics": avg_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved CV metrics: {output_path}")


if __name__ == "__main__":
    main()
