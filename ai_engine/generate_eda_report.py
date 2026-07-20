from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an automatic EDA markdown report.")
    parser.add_argument("--eda-dir", default="ai_engine/eda_outputs", help="Base EDA output directory.")
    parser.add_argument(
        "--report-path",
        default="ai_engine/eda_outputs/EDA_REPORT.md",
        help="Markdown report output path.",
    )
    return parser.parse_args()


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pct(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%"


def main() -> None:
    args = parse_args()
    eda_dir = Path(args.eda_dir)
    report_path = Path(args.report_path)

    summary = read_json_if_exists(eda_dir / "summary.json")
    outliers = read_json_if_exists(eda_dir / "outliers" / "outliers_summary.json")
    skewness = read_json_if_exists(eda_dir / "skewness" / "skewness_summary.json")

    lines: list[str] = []
    lines.append("# EDA Report")
    lines.append("")

    if summary:
        lines.append("## 1) Dataset Overview")
        lines.append(f"- Raw shape: `{summary.get('raw_shape')}`")
        lines.append(f"- Clean shape: `{summary.get('clean_shape')}`")
        lines.append(f"- Raw fraud rate: `{pct(summary.get('raw_target_rate'))}`")
        lines.append(f"- Clean fraud rate: `{pct(summary.get('clean_target_rate'))}`")
        lines.append("")
        lines.append("### Figures")
        for fig in [
            "class_distribution_clean.png",
            "amount_log_distribution_by_class.png",
            "correlation_heatmap_clean.png",
        ]:
            fig_path = eda_dir / fig
            if fig_path.exists():
                lines.append(f"- `{fig}`")
        lines.append("")

    if outliers:
        lines.append("## 2) Outlier Analysis (IQR + Z-score)")
        iqr_summary = outliers.get("iqr_summary", {})
        ranked_iqr = sorted(
            iqr_summary.items(),
            key=lambda kv: kv[1].get("outlier_ratio", 0.0),
            reverse=True,
        )[:5]
        if ranked_iqr:
            lines.append("Top 5 features by IQR outlier ratio:")
            for feature, stats in ranked_iqr:
                lines.append(
                    f"- `{feature}`: {stats.get('outlier_ratio', 0.0) * 100:.2f}% "
                    f"({stats.get('outlier_count', 0)} outliers)"
                )
        lines.append("")
        for fig in ["iqr_outlier_ratio_top.png", "boxplots_top_outliers.png"]:
            fig_path = eda_dir / "outliers" / fig
            if fig_path.exists():
                lines.append(f"- `{fig}`")
        lines.append("")

    if skewness:
        lines.append("## 3) Skewness Analysis")
        candidates = skewness.get("log_transform_candidates", [])
        lines.append(
            f"- Number of log-transform candidates (`|skew| >= {skewness.get('skew_threshold', 1.0)}`): "
            f"`{len(candidates)}`"
        )
        if candidates:
            lines.append("- Top candidates: " + ", ".join(f"`{c}`" for c in candidates[:8]))
        lines.append("")
        for fig in ["top_skewed_features.png", "skew_before_after_log1p.png"]:
            fig_path = eda_dir / "skewness" / fig
            if fig_path.exists():
                lines.append(f"- `{fig}`")
        lines.append("")

    lines.append("## 4) Quick Conclusion")
    lines.append("- Clean dataset appears ready for model training.")
    lines.append("- Dataset is highly imbalanced (fraud is a minority class).")
    lines.append("- Outlier-heavy and skewed features should be reviewed before non-tree models.")
    lines.append("- For RandomForest, current preprocessing remains acceptable.")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"EDA report generated: {report_path}")


if __name__ == "__main__":
    main()
