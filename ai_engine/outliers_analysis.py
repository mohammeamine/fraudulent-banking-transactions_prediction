from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect outliers using IQR and Z-score.")
    parser.add_argument(
        "--csv-path",
        default="dataset/clean/fraudTrain_clean.csv",
        help="Path to dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="ai_engine/eda_outputs/outliers",
        help="Directory for outlier outputs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200000,
        help="Optional sample size for faster plotting. Use <=0 for full data.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Absolute Z-score threshold for outlier detection.",
    )
    return parser.parse_args()


def maybe_sample(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if sample_size and sample_size > 0 and len(df) > sample_size:
        return df.sample(n=sample_size, random_state=random_state)
    return df


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    df = pd.read_csv(args.csv_path)
    df = maybe_sample(df, args.sample_size, args.random_state)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "is_fraud" in numeric_cols:
        numeric_cols.remove("is_fraud")
    # Exclude binary one-hot columns from outlier analysis: IQR/Z-score is not informative for 0/1 flags.
    numeric_cols = [c for c in numeric_cols if df[c].dropna().nunique() > 2]

    iqr_summary: dict[str, dict[str, float]] = {}
    zscore_summary: dict[str, dict[str, float]] = {}

    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_mask = (s < lower) | (s > upper)
        iqr_count = int(iqr_mask.sum())
        iqr_ratio = float(iqr_count / len(s))
        iqr_summary[col] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": iqr_count,
            "outlier_ratio": iqr_ratio,
        }

        std = float(s.std(ddof=0))
        if std == 0:
            z_count = 0
        else:
            z_scores = (s - float(s.mean())) / std
            z_count = int((z_scores.abs() > args.z_threshold).sum())
        z_ratio = float(z_count / len(s))
        zscore_summary[col] = {
            "z_threshold": args.z_threshold,
            "outlier_count": z_count,
            "outlier_ratio": z_ratio,
        }

    top_iqr = sorted(iqr_summary.items(), key=lambda kv: kv[1]["outlier_ratio"], reverse=True)[:8]
    top_cols = [c for c, _ in top_iqr]

    if top_cols:
        plt.figure(figsize=(11, 6))
        ratios = [iqr_summary[c]["outlier_ratio"] for c in top_cols]
        sns.barplot(x=top_cols, y=ratios, hue=top_cols, palette="Reds", legend=False)
        plt.title("Top columns by IQR outlier ratio")
        plt.xlabel("Feature")
        plt.ylabel("Outlier ratio")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "iqr_outlier_ratio_top.png", dpi=140)
        plt.close()

        ncols = 2
        nrows = int(np.ceil(min(len(top_cols), 6) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.8 * nrows))
        axes = np.array(axes).reshape(-1)
        for idx, col in enumerate(top_cols[:6]):
            sns.boxplot(y=df[col], ax=axes[idx], color="#93C5FD")
            axes[idx].set_title(f"Boxplot: {col}")
            axes[idx].set_ylabel(col)
        for j in range(idx + 1, len(axes)):  # type: ignore[name-defined]
            axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / "boxplots_top_outliers.png", dpi=140)
        plt.close()

    report = {
        "dataset_path": args.csv_path,
        "n_rows_used": int(len(df)),
        "n_numeric_features": int(len(numeric_cols)),
        "iqr_summary": iqr_summary,
        "zscore_summary": zscore_summary,
    }
    (output_dir / "outliers_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Outlier analysis done. Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
