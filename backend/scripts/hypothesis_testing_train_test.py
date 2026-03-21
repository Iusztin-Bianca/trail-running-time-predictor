"""
Hypothesis testing to compare train vs test set distributions.

Tests:
1. Levene's test for equality of variances (homoscedasticity)
2. Two-sample t-test for equality of means
3. Kolmogorov-Smirnov test for distribution similarity
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from app.config.settings import settings

# Load data
parquet_path = settings.data_dir / "strava_training_features.parquet"
df = pd.read_parquet(parquet_path)

# Sort by date and split
df['start_date_dt'] = pd.to_datetime(df['start_date'])
df = df.sort_values('start_date_dt').reset_index(drop=True)

n_total = len(df)
n_train = int(n_total * 0.85)

train_df = df.iloc[:n_train]
test_df = df.iloc[n_train:]

print("=" * 100)
print("HYPOTHESIS TESTING: TRAIN vs TEST SET")
print("=" * 100)
print(f"\nSample sizes: Train={len(train_df)}, Test={len(test_df)}")

# Features to test
features = ['km_effort', 'elevation_gain_m', 'total_time_sec', 'average_elevation',
            'elevation_loss_m', 'downhill_ratio', 'flat_ratio', 'mean_positive_grade']

print("\n" + "=" * 100)
print("TEST 1: LEVENE'S TEST FOR EQUALITY OF VARIANCES")
print("=" * 100)
print("H0: Variances are equal (homoscedastic)")
print("H1: Variances are different\n")

variance_results = []
for feature in features:
    train_data = train_df[feature].dropna()
    test_data = test_df[feature].dropna()

    # Levene's test (robust to non-normality)
    statistic, p_value = stats.levene(train_data, test_data)

    # Variance ratio (test/train)
    variance_ratio = test_data.var() / train_data.var()

    significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

    variance_results.append({
        'feature': feature,
        'train_var': train_data.var(),
        'test_var': test_data.var(),
        'ratio': variance_ratio,
        'p_value': p_value,
        'significant': significant
    })

    print(f"{feature:25s} | Variance ratio: {variance_ratio:.3f} | "
          f"p-value: {p_value:.4f} {significant}")

print("\n" + "=" * 100)
print("TEST 2: TWO-SAMPLE T-TEST FOR EQUALITY OF MEANS")
print("=" * 100)
print("H0: Means are equal")
print("H1: Means are different\n")

mean_results = []
for feature in features:
    train_data = train_df[feature].dropna()
    test_data = test_df[feature].dropna()

    # Welch's t-test (doesn't assume equal variances)
    statistic, p_value = stats.ttest_ind(train_data, test_data, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((train_data.var() + test_data.var()) / 2)
    cohens_d = (test_data.mean() - train_data.mean()) / pooled_std

    significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

    mean_results.append({
        'feature': feature,
        'train_mean': train_data.mean(),
        'test_mean': test_data.mean(),
        'cohens_d': cohens_d,
        'p_value': p_value,
        'significant': significant
    })

    print(f"{feature:25s} | Cohen's d: {cohens_d:+.3f} | "
          f"p-value: {p_value:.4f} {significant}")

print("\n" + "=" * 100)
print("TEST 3: KOLMOGOROV-SMIRNOV TEST FOR DISTRIBUTION SIMILARITY")
print("=" * 100)
print("H0: Distributions are the same")
print("H1: Distributions are different\n")

ks_results = []
for feature in features:
    train_data = train_df[feature].dropna()
    test_data = test_df[feature].dropna()

    # KS test
    statistic, p_value = stats.ks_2samp(train_data, test_data)

    significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

    ks_results.append({
        'feature': feature,
        'ks_statistic': statistic,
        'p_value': p_value,
        'significant': significant
    })

    print(f"{feature:25s} | KS statistic: {statistic:.3f} | "
          f"p-value: {p_value:.4f} {significant}")

print("\n" + "=" * 100)
print("SUMMARY OF SIGNIFICANT DIFFERENCES")
print("=" * 100)
print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05\n")

# Variance differences
sig_variance = [r for r in variance_results if r['significant']]
if sig_variance:
    print(f"VARIANCE DIFFERENCES (Levene's test): {len(sig_variance)}/{len(features)} features")
    for r in sig_variance:
        direction = "LOWER" if r['ratio'] < 1 else "HIGHER"
        print(f"  - {r['feature']:25s}: Test variance {direction} ({r['ratio']:.2f}x) {r['significant']}")
else:
    print("VARIANCE: No significant differences found")

# Mean differences
sig_mean = [r for r in mean_results if r['significant']]
if sig_mean:
    print(f"\nMEAN DIFFERENCES (t-test): {len(sig_mean)}/{len(features)} features")
    for r in sig_mean:
        direction = "LOWER" if r['cohens_d'] < 0 else "HIGHER"
        effect = "LARGE" if abs(r['cohens_d']) >= 0.8 else "MEDIUM" if abs(r['cohens_d']) >= 0.5 else "SMALL"
        print(f"  - {r['feature']:25s}: Test mean {direction}, effect size: {effect} (d={r['cohens_d']:+.2f}) {r['significant']}")
else:
    print("\nMEAN: No significant differences found")

# Distribution differences
sig_ks = [r for r in ks_results if r['significant']]
if sig_ks:
    print(f"\nDISTRIBUTION DIFFERENCES (KS test): {len(sig_ks)}/{len(features)} features")
    for r in sig_ks:
        print(f"  - {r['feature']:25s}: Different distributions {r['significant']}")
else:
    print("\nDISTRIBUTION: No significant differences found")

print("\n" + "=" * 100)
print("CONCLUSION")
print("=" * 100)

# Count total significant differences
total_sig = len(set([r['feature'] for r in sig_variance] +
                    [r['feature'] for r in sig_mean] +
                    [r['feature'] for r in sig_ks]))

if total_sig > 0:
    print(f"\n{total_sig}/{len(features)} features show significant differences between train and test sets.")
    print("\nThis confirms that the test set is NOT representative of the training distribution.")
    print("The lower test MAE is likely due to this distribution shift, not model performance.")
    print("\nRecommendation: Use CV MAE as the more reliable performance estimate.")
else:
    print("\nNo significant differences found. Sets appear to be from the same distribution.")
    print("The CV vs test MAE difference may be due to random variation or sample size.")
