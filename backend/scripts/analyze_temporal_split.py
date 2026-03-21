"""
Analyze the temporal train/test split to understand CV vs test discrepancy.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from app.config.settings import settings

# Load data
parquet_path = settings.data_dir / "strava_training_features.parquet"
df = pd.read_parquet(parquet_path)

# Sort by date
df['start_date_dt'] = pd.to_datetime(df['start_date'])
df = df.sort_values('start_date_dt').reset_index(drop=True)

# Calculate split point (85% train, 15% test)
n_total = len(df)
n_train = int(n_total * 0.85)
n_test = n_total - n_train

# Split data
train_df = df.iloc[:n_train]
test_df = df.iloc[n_train:]

print("=" * 100)
print("TEMPORAL SPLIT ANALYSIS")
print("=" * 100)
print(f"\nTotal activities: {n_total}")
print(f"Train set: {n_train} activities (85%)")
print(f"Test set: {n_test} activities (15%)")

print("\n" + "=" * 100)
print("DATE RANGES")
print("=" * 100)
print(f"\nTrain set:")
print(f"  Start: {train_df['start_date_dt'].min()}")
print(f"  End:   {train_df['start_date_dt'].max()}")
print(f"\nTest set:")
print(f"  Start: {test_df['start_date_dt'].min()}")
print(f"  End:   {test_df['start_date_dt'].max()}")

print("\n" + "=" * 100)
print("TEST SET ACTIVITIES (newest 15%)")
print("=" * 100)
test_display = test_df[['start_date_dt', 'activity_name', 'intensity_level',
                         'km_effort', 'elevation_gain_m', 'total_time_sec']].copy()
test_display['total_time_min'] = (test_display['total_time_sec'] / 60).round(1)
test_display = test_display.drop(columns=['total_time_sec'])

for idx, row in test_display.iterrows():
    # Handle special characters in activity names
    activity_name = str(row['activity_name']).encode('ascii', 'replace').decode('ascii')
    print(f"\n{row['start_date_dt'].strftime('%Y-%m-%d')} | {activity_name}")
    print(f"  Intensity: {row['intensity_level']} | Km effort: {row['km_effort']:.1f} | "
          f"Elev: {row['elevation_gain_m']:.0f}m | Time: {row['total_time_min']:.1f}min")

# Statistics comparison
print("\n" + "=" * 100)
print("STATISTICS COMPARISON: TRAIN vs TEST")
print("=" * 100)

metrics = ['km_effort', 'elevation_gain_m', 'total_time_sec', 'intensity_level']
for metric in metrics:
    train_mean = train_df[metric].mean()
    test_mean = test_df[metric].mean()
    train_std = train_df[metric].std()
    test_std = test_df[metric].std()

    print(f"\n{metric}:")
    print(f"  Train: mean={train_mean:.2f}, std={train_std:.2f}")
    print(f"  Test:  mean={test_mean:.2f}, std={test_std:.2f}")

print("\n" + "=" * 100)
print("INTENSITY DISTRIBUTION")
print("=" * 100)
print("\nTrain set:")
print(train_df['intensity_level'].value_counts().sort_index())
print("\nTest set:")
print(test_df['intensity_level'].value_counts().sort_index())
