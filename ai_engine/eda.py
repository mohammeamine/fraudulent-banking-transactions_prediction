from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA on fraud train dataset (raw/clean).")
    parser.add_argument("--raw-csv", default="dataset/raw/fraudTrain.csv", help="Path to raw train CSV.")
    parser.add_argument(
        "--clean-csv",
        default="dataset/clean/fraudTrain_clean.csv",
        help="Path to clean train CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="ai_engine/eda_outputs",
        help="Directory where EDA figures and summary are saved.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200000,
        help="Optional sample size for faster plotting. Use 0 or negative for full data.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling.")
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

    raw_df = pd.read_csv(args.raw_csv)
    clean_df = pd.read_csv(args.clean_csv)
    raw_df = maybe_sample(raw_df, args.sample_size, args.random_state)
    clean_df = maybe_sample(clean_df, args.sample_size, args.random_state)

    # Remove accidental index-like columns if they exist in raw files.
    raw_drop_candidates = [c for c in raw_df.columns if c.lower() in {"unnamed: 0", "look"}]
    if raw_drop_candidates:
        raw_df = raw_df.drop(columns=raw_drop_candidates)

    summary = {
        "raw_shape": list(raw_df.shape),
        "clean_shape": list(clean_df.shape),
        "raw_columns": raw_df.columns.tolist(),
        "clean_columns": clean_df.columns.tolist(),
        "raw_missing_values": raw_df.isna().sum().to_dict(),
        "clean_missing_values": clean_df.isna().sum().to_dict(),
        "raw_target_rate": float(raw_df["is_fraud"].mean()) if "is_fraud" in raw_df.columns else None,
        "clean_target_rate": float(clean_df["is_fraud"].mean()) if "is_fraud" in clean_df.columns else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 1) Class balance plot (clean dataset).
    if "is_fraud" in clean_df.columns:
        plt.figure(figsize=(7, 4.5))
        ax = sns.countplot(data=clean_df, x="is_fraud", hue="is_fraud", palette="Blues", legend=False)
        ax.set_title("Class distribution (clean train)")
        ax.set_xlabel("is_fraud")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "class_distribution_clean.png", dpi=140)
        plt.close()

    # 2) Missing values bar chart (raw dataset).
    raw_missing = raw_df.isna().sum().sort_values(ascending=False)
    raw_missing = raw_missing[raw_missing > 0]
    if not raw_missing.empty:
        plt.figure(figsize=(10, 5))
        raw_missing.head(20).plot(kind="bar", color="#2563EB")
        plt.title("Top missing-value columns (raw train)")
        plt.xlabel("Columns")
        plt.ylabel("Missing count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "raw_missing_top20.png", dpi=140)
        plt.close()

    # 3) Amount distribution by class (raw when available, fallback clean).
    source_df = raw_df if {"amt", "is_fraud"}.issubset(raw_df.columns) else clean_df
    if {"amt", "is_fraud"}.issubset(source_df.columns):
        plot_df = source_df[["amt", "is_fraud"]].copy()
        plot_df["amt_log1p"] = np.log1p(plot_df["amt"].clip(lower=0))

        plt.figure(figsize=(8, 5))
        sns.kdeplot(data=plot_df, x="amt_log1p", hue="is_fraud", common_norm=False, fill=True, alpha=0.25)
        plt.title("Log(amount) distribution by class")
        plt.xlabel("log1p(amt)")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(output_dir / "amount_log_distribution_by_class.png", dpi=140)
        plt.close()

    # 4) Correlation heatmap on selected clean features.
    selected_cols = [
        c
        for c in [
            "is_fraud",
            "amt",
            "age",
            "distance_km",
            "city_pop",
            "merchant_freq",
            "trans_hour",
            "trans_day_of_week",
            "trans_month",
        ]
        if c in clean_df.columns
    ]
    if len(selected_cols) >= 3:
        corr = clean_df[selected_cols].corr(numeric_only=True)
        plt.figure(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Correlation heatmap (selected clean features)")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap_clean.png", dpi=140)
        plt.close()

    print(f"EDA done. Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
