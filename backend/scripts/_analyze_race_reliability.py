"""Quick analysis: is race-level MAE reliable with only 9 test races?"""
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import numpy as np
from app.ml.data.blob_storage import BlobStorageManager
from app.ml.data.data_splitter import TemporalSplitter
from app.ml.evaluation.metrics import MetricsCalculator
from app.ml.services.trainer import ModelTrainer
from app.ml.models.model_factory import ModelFactory
from app.ml.config.xgboost_params import XGBoostParams
from app.constants import DEFAULT_EXCLUDE_COLUMNS, SEGMENT_TARGET_COLUMN

blob = BlobStorageManager(blob_name="segments_training_features.parquet")
df = blob.download_parquet()

splitter = TemporalSplitter(test_ratio=0.15)
metrics = MetricsCalculator()

# Best params from last run
params = XGBoostParams(
    learning_rate=0.03, max_depth=3, min_child_weight=1, n_estimators=300, subsample=0.8
)
model = ModelFactory.create("xgboost", params)
trainer = ModelTrainer(model=model, splitter=splitter, metrics=metrics)

# 1. Race-level cross-validation (5 folds on full dataset)
print("=== Race-level Cross-Validation (5 folds) ===")
cv = trainer.cross_validate(df, n_splits=5)
print(f"Avg Race MAE:  {cv['mae']:.0f}s ({cv['mae']/60:.1f} min)")
print(f"Avg Race MAPE: {cv['mape']:.1f}%")
print(f"Avg Race R2:   {cv['r2']:.3f}")

# 2. Naive baseline: predict mean train race time for every test race
train_df, test_df = splitter.split_train_test(df)
train_race_mean = train_df.groupby("activity_id")[SEGMENT_TARGET_COLUMN].sum().mean()
test_race_actual = test_df.groupby("activity_id")[SEGMENT_TARGET_COLUMN].sum().values
baseline_pred = np.full(len(test_race_actual), train_race_mean)
baseline_mae = np.abs(test_race_actual - baseline_pred).mean()
baseline_mape = (np.abs(test_race_actual - baseline_pred) / test_race_actual).mean() * 100
print()
print("=== Naive Baseline (predict mean train race time) ===")
print(f"Mean train race: {train_race_mean/60:.0f} min")
print(f"Baseline MAE:    {baseline_mae:.0f}s  (model: 780s)")
print(f"Baseline MAPE:   {baseline_mape:.1f}%  (model: 9.6%)")

# 3. Train model, get per-race errors
feature_cols = [
    c for c in df.columns
    if c not in DEFAULT_EXCLUDE_COLUMNS and c != SEGMENT_TARGET_COLUMN
]
X_train = train_df[feature_cols].fillna(0).values
y_train = train_df[SEGMENT_TARGET_COLUMN].values
model.fit(X_train, y_train)

test_df2 = test_df.copy()
test_df2["_pred"] = model.predict(test_df[feature_cols].fillna(0).values)
race = test_df2.groupby("activity_id").agg(
    actual=(SEGMENT_TARGET_COLUMN, "sum"),
    pred=("_pred", "sum"),
)
abs_errors = np.abs(race["actual"].values - race["pred"].values)

# 4. Bootstrap CI
np.random.seed(42)
n = len(abs_errors)
boot_maes = []
for _ in range(10000):
    idx = np.random.choice(n, n, replace=True)
    boot_maes.append(abs_errors[idx].mean())
boot_maes = np.array(boot_maes)
ci_low, ci_high = np.percentile(boot_maes, [2.5, 97.5])

print()
print("=== Bootstrap 95% CI for Race MAE (9 test races) ===")
print(f"Observed MAE: {abs_errors.mean():.0f}s ({abs_errors.mean()/60:.1f} min)")
print(f"95% CI:       [{ci_low:.0f}s, {ci_high:.0f}s]  =  [{ci_low/60:.1f} min, {ci_high/60:.1f} min]")
print(f"CI width:     {(ci_high-ci_low)/60:.1f} min  (wide = uncertain)")

print()
print("Per-race errors:")
for i, (idx, row) in enumerate(race.iterrows()):
    err = row["pred"] - row["actual"]
    pct = abs(err) / row["actual"] * 100
    print(f"  Race {i+1}: actual={row['actual']/60:.0f}min  pred={row['pred']/60:.0f}min  err={err/60:+.0f}min ({pct:.0f}%)")
