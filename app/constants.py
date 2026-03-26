"""
Constants and default configurations for the app.

Centralizes hardcoded values to make them easier to maintain and modify.
"""

# Target column for segment-level training
SEGMENT_TARGET_COLUMN = "segment_time_sec"

# Columns to exclude from features during data preparation
# These are either target variables, identifiers, or date columns
DEFAULT_EXCLUDE_COLUMNS = {
    "start_date",           # Date column used for temporal sorting
    "segment_time_sec",     # Target variable (segment-level)
    "segment_pace_mps",     # Derived from target, not a feature
    "total_time_sec",       # Legacy activity-level target (kept for safety)
    "activity_id",          # Identifier, not a feature
    "activity_name",        # Identifier, not a feature
    "start_date_dt",            # Datetime version of start_date
    "elevation_gain_last_30d",  # Historical training load — cannot be computed at inference time
}

# Default training configuration values
DEFAULT_CV_SPLITS = 5           # Number of cross-validation folds
DEFAULT_CV_GAP = 0              # Gap between train and validation sets
DEFAULT_TEST_SIZE = 0.20        # Holdout test set ratio (20%)

# Minimum sizes for data splitting
MIN_VALIDATION_SIZE = 5         # Minimum samples per validation fold
MIN_TRAIN_RATIO = 0.2           # Minimum ratio of data for initial training

# Strava workout_type constants
STRAVA_WORKOUT_TYPE_RECOVERY = 2
STRAVA_WORKOUT_TYPE_RACE = 1

# Activity type binary flags (used in segment features: is_race, is_easy)
IS_RACE = 1
IS_EASY = 1
IS_NOT = 0

# GPX downsampling thresholds (applied only for long races to reduce latency)
GPX_DOWNSAMPLE_THRESHOLD_M = 30_000   # Minimum race distance (m) to trigger downsampling
GPX_MAX_POINTS = 5000                  # Maximum GPS points after downsampling
