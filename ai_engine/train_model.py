from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from ai_engine.preprocessing import load_training_csv, prepare_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection model (RandomForest).")
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
        "--artifacts-dir",
        default="ai_engine/model_artifacts",
        help="Directory where model and metadata are saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split ratio.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading dataset: {csv_path}")
    df = load_training_csv(str(csv_path))
    if args.sample_size is not None and args.sample_size > 0:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=args.random_state)
        print(f"Using sampled rows: {len(df)}")
    else:
        print(f"Using full rows: {len(df)}")

    print("[2/5] Preparing features")
    prepared = prepare_features(df, target_column=args.target_column)
    X, y = prepared.X, prepared.y
    print(f"Features: {len(prepared.feature_names)}")
    print(f"Fraud rate: {y.mean():.4f}")

    print("[3/5] Splitting train/validation")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    print("[4/5] Training RandomForestClassifier")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("[5/5] Evaluating and saving artifacts")
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_proba)
    report_dict = classification_report(y_val, y_pred, output_dict=True)
    report_text = classification_report(y_val, y_pred, digits=4)

    model_path = artifacts_dir / "fraud_rf_model.joblib"
    metadata_path = artifacts_dir / "fraud_rf_metadata.json"

    joblib.dump(model, model_path)
    metadata = {
        "target_column": args.target_column,
        "feature_names": prepared.feature_names,
        "n_features": len(prepared.feature_names),
        "n_rows_used": int(len(df)),
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "roc_auc": float(auc),
        "classification_report": report_dict,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nValidation metrics")
    print("------------------")
    print(f"ROC-AUC: {auc:.4f}")
    print(report_text)
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
