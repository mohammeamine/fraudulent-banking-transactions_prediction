# EDA Report

## 1) Dataset Overview

- Raw shape: `[150000, 22]`
- Clean shape: `[150000, 23]`
- Raw fraud rate: `0.58%`
- Clean fraud rate: `0.58%`

### Figures

- `class_distribution_clean.png`
- `amount_log_distribution_by_class.png`
- `correlation_heatmap_clean.png`

## 2) Outlier Analysis (IQR + Z-score)

Top 5 features by IQR outlier ratio:

- `city_pop`: 18.67% (28006 outliers)
- `amt`: 5.16% (7746 outliers)
- `merchant_freq`: 2.17% (3253 outliers)
- `age`: 0.02% (23 outliers)
- `trans_hour`: 0.00% (0 outliers)
- `iqr_outlier_ratio_top.png`
- `boxplots_top_outliers.png`

## 3) Skewness Analysis

- Number of log-transform candidates (`|skew| >= 1.0`): `2`
- Top candidates: `amt`, `city_pop`
- `top_skewed_features.png`
- `skew_before_after_log1p.png`

## 4) Quick Conclusion

- Clean dataset appears ready for model training.
- Dataset is highly imbalanced (fraud is a minority class).
- Outlier-heavy and skewed features should be reviewed before non-tree models.
- For RandomForest, current preprocessing remains acceptable.

