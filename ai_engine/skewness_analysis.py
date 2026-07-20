from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze skewness and log-transform candidates.")
    parser.add_argument(
        "--csv-path",
        default="dataset/clean/fraudTrain_clean.csv",
        help="Path to dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="ai_engine/eda_outputs/skewness",
        help="Directory for skewness outputs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200000,
        help="Optional sample size for faster plotting. Use <=0 for full data.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--skew-threshold",
        type=float,
        default=1.0,
        help="Absolute skew threshold to mark a feature as highly skewed.",
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
    # Exclude binary one-hot columns from skew/log recommendations.
    numeric_cols = [c for c in numeric_cols if df[c].dropna().nunique() > 2]

    skew_map: dict[str, float] = {}
    for col in numeric_cols:
        skew_map[col] = float(df[col].dropna().skew())

    ranked = sorted(skew_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
    candidates = [c for c, sk in ranked if abs(sk) >= args.skew_threshold]

    transformed_skew: dict[str, float] = {}
    valid_candidates: list[str] = []
    for col in candidates:
        s = df[col]
        if (s < 0).any():
            continue
        transformed_skew[col] = float(np.log1p(s).skew())
        valid_candidates.append(col)

    top_ranked = ranked[:10]
    if top_ranked:
        plt.figure(figsize=(11, 6))
        names = [c for c, _ in top_ranked]
        vals = [v for _, v in top_ranked]
        sns.barplot(x=names, y=vals, hue=names, palette="Purples", legend=False)
        plt.title("Top skewed features (original)")
        plt.xlabel("Feature")
        plt.ylabel("Skewness")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "top_skewed_features.png", dpi=140)
        plt.close()

    top_compare = valid_candidates[:4]
    if top_compare:
        fig, axes = plt.subplots(nrows=len(top_compare), ncols=2, figsize=(12, 3.2 * len(top_compare)))
        axes = np.array(axes).reshape(len(top_compare), 2)
        for i, col in enumerate(top_compare):
            sns.histplot(df[col], kde=True, ax=axes[i, 0], color="#60A5FA")
            axes[i, 0].set_title(f"{col} (original)")
            sns.histplot(np.log1p(df[col]), kde=True, ax=axes[i, 1], color="#34D399")
            axes[i, 1].set_title(f"{col} (log1p)")
        plt.tight_layout()
        plt.savefig(output_dir / "skew_before_after_log1p.png", dpi=140)
        plt.close()

    report = {
        "dataset_path": args.csv_path,
        "n_rows_used": int(len(df)),
        "skew_threshold": args.skew_threshold,
        "skewness_original": skew_map,
        "log_transform_candidates": valid_candidates,
        "skewness_after_log1p": transformed_skew,
    }
    (output_dir / "skewness_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Skewness analysis done. Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
