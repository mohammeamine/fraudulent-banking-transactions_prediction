from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from ai_engine.predict import score_transactions
from ai_engine.preprocessing import load_training_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved model on test dataset.")
    parser.add_argument(
        "--csv-path",
        default="dataset/clean/fraudTest_clean.csv",
        help="Path to test CSV file.",
    )
    parser.add_argument(
        "--target-column",
        default="is_fraud",
        help="Target label column name in test CSV.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="ai_engine/model_artifacts",
        help="Directory containing saved model and metadata.",
    )
    parser.add_argument(
        "--output-json",
        default="ai_engine/model_artifacts/test_metrics.json",
        help="Where to save final test metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading test dataset: {args.csv_path}")
    df_test = load_training_csv(args.csv_path)

    if args.target_column not in df_test.columns:
        raise ValueError(f"Target column '{args.target_column}' not found in test dataset.")

    y_true = df_test[args.target_column].astype(int)
    scored = score_transactions(df_test, artifacts_dir=args.artifacts_dir)
    y_pred = scored["is_fraud_pred"].astype(int)
    y_proba = scored["fraud_probability"].astype(float)

    auc = roc_auc_score(y_true, y_proba)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_text = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred).tolist()

    print("\nFinal test results")
    print("------------------")
    print(f"ROC-AUC: {auc:.4f}")
    print(report_text)
    print(f"Confusion matrix: {cm}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "n_rows_test": int(len(df_test)),
                "roc_auc": float(auc),
                "classification_report": report_dict,
                "confusion_matrix": cm,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved test metrics: {output_path}")


if __name__ == "__main__":
    main()
