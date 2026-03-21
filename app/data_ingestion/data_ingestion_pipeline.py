"""
Pipeline for fetching Strava running activities and extracting features for training.

This pipeline:
1. Fetches all "Run" and "TrailRun" activities from Strava with elevation gain >= 100m
2. Extracts features from each activity using FeatureExtractor
3. Stores intensity_level feature (0 = recovery, 1 = training, 2 = race)
4. Computes historical features (e.g., elevation_gain_last_30d for training load)
5. Saves all features to a parquet file for ML training
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from app.data_ingestion.strava_client import StravaClient
from app.feature_engineering.point_extractor import PointExtractor
from app.feature_engineering.segment_features import SegmentFeatureExtractor
from app.config.settings import settings
from app.constants import INTENSITY_RACE, INTENSITY_RECOVERY, INTENSITY_TRAINING, STRAVA_WORKOUT_TYPE_RACE, STRAVA_WORKOUT_TYPE_RECOVERY
from app.ml.data.blob_storage import BlobStorageManager

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """
    Pipeline for fetching Strava activities and extracting features for ML training.

    This pipeline handles the complete flow from Strava API to parquet dataset:
    - Fetch all Run and TrailRun activities with elevation_gain >= 150m and distance >= 5km
    - Extract segment features using SegmentFeatureExtractor (one row per segment per activity)
    - Track intensity_level feature (0 = recovery, 1 = training, 2 = race)
    - Save to parquet file for model training
    """

    def __init__(
        self,
        strava_client: StravaClient,
        feature_extractor: SegmentFeatureExtractor,
        output_path: Optional[Path] = None,
        min_elevation_gain_m: float = 150.00,
        min_distance_m: float = 4000.00,
        blob_manager: Optional[BlobStorageManager] = None,
        save_raw_activities: bool = True
    ):
        """
        Initialize the pipeline.

        Args:
            strava_client: Configured StravaClient instance
            feature_extractor: SegmentFeatureExtractor instance for extracting features
            output_path: Path to save parquet file (default: data/strava_training_features.parquet)
            min_elevation_gain_m: Minimum elevation gain filter (default: 150m)
            min_distance_m: Minimum run distance (default: 5000m)
            blob_manager: Optional BlobStorageManager for saving raw activities
            save_raw_activities: Whether to save raw stream data to blob storage (default: True)
        """
        self.client = strava_client
        self.feature_extractor = feature_extractor
        self.output_path = output_path or (settings.data_dir / "strava_training_features.parquet")
        self.min_elevation_gain_m = min_elevation_gain_m
        self.min_distance_m = min_distance_m
        self.blob_manager = blob_manager
        self.save_raw_activities = save_raw_activities

    def fetch_all_run_activities(self, after: Optional[int] = None) -> List[Dict]:
        """
        Fetch all running activities from Strava.

        Args:
            after: timestamp to fetch activities after this date (for incremental updates)

        Returns:
            List of activity dictionaries filtered by:
            - type = "Run" or "TrailRun"
            - total_elevation_gain >= min_elevation_gain_m
            - total_distance >= min_distance_m
            - start_date > after (if provided)
        """

        if after:
            logger.info(f"Fetching Run and TrailRun activities from Strava after {datetime.fromtimestamp(after)}...")
        else:
            logger.info("Fetching all Run and TrailRun activities from Strava...")

        all_activities = []
        page = 1
        per_page = 200  # Max allowed by Strava API

        while True:
            activities = self.client.get_activities(page=page, per_page=per_page, after=after)

            if not activities:
                logger.info(f"No more activities found (page {page}). Stopping pagination.")
                break

            # Filter: Run and TrailRun activities with elevation gain >= elevation threshold and distance >= distance threshold
            filtered = [
                activity for activity in activities
                if activity.get("type") in ["Run", "TrailRun"]
                and activity.get("total_elevation_gain", 0) >= self.min_elevation_gain_m
                and activity.get("distance", 0) >= self.min_distance_m
            ]

            all_activities.extend(filtered)

            # Stop if we got fewer activities than requested (last page)
            if len(activities) < per_page:
                logger.info(f"Received {len(activities)} < {per_page}, reached last page.")
                break

            page += 1

        logger.info(
            f"Fetched {len(all_activities)} Run and TrailRun activities with "
            f"elevation gain >= {self.min_elevation_gain_m}m and "
            f"total distance >= {self.min_distance_m}m"
        )
        return all_activities

    
    def extract_features_from_activity(
        self,
        activity_id: int,
        start_time: datetime,
        intensity_level: int,
        activity_metadata: Optional[Dict] = None
    ) -> Optional[List[Dict[str, float]]]:
        """
        Extract features from a single Strava activity.
        
        Args:
            activity_id: Strava activity ID
            start_time: Activity start datetime
            activity_metadata: Optional activity metadata for saving raw data

        Returns:
            Dictionary of features, or None if extraction fails
        """
        try:
            # If raw activity already exists in blob storage, reuse its streams
            if self.blob_manager and self.blob_manager.raw_activity_exists(activity_id):
                logger.info(f"Raw activity {activity_id} already in blob storage, reusing streams")
                raw_activity = self.blob_manager.download_raw_activity(activity_id)
                streams = raw_activity["streams"]
            else:
                # Fetch activity streams from Strava API (latlng, altitude, time)
                streams = self.client.get_activity_streams(activity_id)

                # Save raw activity data to blob storage (if configured)
                if self.blob_manager and self.save_raw_activities and activity_metadata:
                    try:
                        self.blob_manager.upload_raw_activity(
                            activity_id=activity_id,
                            activity_metadata=activity_metadata,
                            streams=streams,
                            overwrite=False
                        )
                        logger.info(f"Saved raw activity {activity_id} to blob storage")
                    except Exception as e:
                        logger.warning(f"Failed to save raw activity {activity_id}: {e}")

            # Extract points and distance stream from Strava streams
            try:
                points = PointExtractor.extract_from_streams(streams, start_time)
            except ValueError as e:
                logger.warning(f"Missing required streams for activity {activity_id}: {e}")
                return None
            
            distance_stream = None
            if streams.get('distance'):
                distance_stream = streams['distance']['data'] if isinstance(streams['distance'], dict) else streams['distance']
            segment_features = self.feature_extractor.extract_features(points, intensity_level, distance_stream)

            return segment_features

        except Exception as e:
            logger.error(f"Failed to extract features from activity {activity_id}: {e}")
            return None

    def process_activities(self, activities: List[Dict]) -> pd.DataFrame:
        """
        Process list of activities and extract features for each.

        Args:
            activities: List of activity dictionaries from Strava API

        Returns:
            DataFrame with features and metadata for each activity

        Each row contains:
        - activity_id: Strava activity ID
        - activity_name: Activity name
        - intensity_level: 0 = recovery, 1 = training, 2 = race
        - start_date: Activity start datetime
        - All extracted features (distance, elevation, slopes, etc.)
        """
        logger.info(f"Processing {len(activities)} activities...")

        rows = []
        successful = 0
        failed = 0

        for i, activity in enumerate(activities, 1):
            activity_id = activity["id"]
            activity_name = activity.get("name", "Unknown")

            logger.info(
                f"[{i}/{len(activities)}] Processing activity {activity_id}: {activity_name}"
            )

            try:
                # Parse start time
                start_time = datetime.fromisoformat(
                    activity["start_date"].replace("Z", "+00:00")
                )

                # Determine intensity level from workout_type
                workout_type = activity.get("workout_type")
                if workout_type == STRAVA_WORKOUT_TYPE_RECOVERY:
                    intensity_level = INTENSITY_RECOVERY
                elif workout_type == STRAVA_WORKOUT_TYPE_RACE:
                    intensity_level = INTENSITY_RACE
                else:
                    intensity_level = INTENSITY_TRAINING

                # Extract features from streams (also saves raw data if configured)
                segment_features = self.extract_features_from_activity(
                    activity_id=activity_id,
                    start_time=start_time,
                    intensity_level = intensity_level,
                    activity_metadata=activity  # Pass full activity for raw data storage
                )

                if segment_features is None:
                    logger.warning(f"Skipping activity {activity_id} - feature extraction failed")
                    failed += 1
                    continue

                # Fetch detailed activity to get description for tehnic_terrain
                #detailed = self.client.get_activity(activity_id)
                #description = (detailed.get("description") or "").lower()
                #tehnic_terrain = 1 if "tehnic" in description else 0

                if not segment_features:
                    logger.warning(f"Skipping activity {activity_id}")
                    failed+=1
                    continue
                
                for segment in segment_features:
                    row = {
                        "activity_id": activity_id,
                        "activity_name": activity_name,
                        "start_date": start_time.isoformat(),
                        **segment  # Unpack all extracted features
                        }

                    rows.append(row)
                successful += 1

            except Exception as e:
                logger.error(
                    f"[{i}/{len(activities)}] Failed to process activity {activity_id}: {e}",
                    exc_info=True
                )
                failed += 1

        logger.info(f"Processing complete: {successful} successful, {failed} failed")

        if not rows:
            logger.warning("No activities were successfully processed!")
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def compute_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute historical/temporal features that depend on past activities.

        Args:
            df: DataFrame with activities and their features

        Returns:
            DataFrame with additional historical features:
            - elevation_gain_last_30d: Total elevation gained in 30 days before each activity
              (reflects training load / how trained the athlete was)

        Activities are sorted by date, and for each activity we look at
        all prior activities within the 30-day window (excluding current activity).
        """
        if df.empty:
            return df

        logger.info("Computing historical features...")

        # Ensure start_date is datetime
        df = df.copy()
        df["start_date_dt"] = pd.to_datetime(df["start_date"])

        # Sort by date (oldest first)
        df = df.sort_values("start_date_dt").reset_index(drop=True)

        # Compute elevation gained in last 30 days for each activity
        elevation_last_30d = []

        for i, row in df.iterrows():
            current_date = row["start_date_dt"]
            window_start = current_date - timedelta(days=30)

            # Get all activities in the 30-day window BEFORE current activity
            mask = (df["start_date_dt"] >= window_start) & (df["start_date_dt"] < current_date)
            prior_activities = df.loc[mask]

            # Sum elevation gain from prior activities
            total_elevation = prior_activities["elevation_gain_m"].sum()
            elevation_last_30d.append(round(total_elevation, 1))

        df["elevation_gain_last_30d"] = elevation_last_30d

        # Remove temporary column
        df = df.drop(columns=["start_date_dt"])

        return df

    def save_to_parquet(self, df: pd.DataFrame) -> None:
        """
        Save DataFrame to parquet file.

        Args:
            df: DataFrame with features and metadata

        Note: Creates parent directories if they don't exist
        """
        if df.empty:
            logger.warning("DataFrame is empty, not saving to parquet")
            return

        # Create parent directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to parquet
        df.to_parquet(self.output_path, index=False, engine='pyarrow')

        logger.info(f"Saved {len(df)} activities to {self.output_path}")
        logger.info(f"File size: {self.output_path.stat().st_size / 1024 / 1024:.2f} MB")

    def run(self, after: Optional[int] = None) -> pd.DataFrame:
        """
        Run the complete pipeline.

        Args:
            after: timestamp to fetch activities after this date (for incremental updates)

        Returns:
            DataFrame with all processed activities and their features

        Pipeline steps:
        1. Fetch all Run activities with elevation gain >= threshold (optionally after a date)
        2. Extract features from each activity
        3. Compute historical features (training load indicators)
        4. Save to parquet file
        """
        logger.info("=" * 80)
        logger.info("STRAVA TRAINING PIPELINE - START")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  - Minimum elevation gain: {self.min_elevation_gain_m}m")
        logger.info(f"  - Output path: {self.output_path}")
        logger.info(f"  - Save raw activities: {self.save_raw_activities and self.blob_manager is not None}")
        if after:
            logger.info(f"  - Incremental update: fetching activities after {datetime.fromtimestamp(after)}")
        logger.info("=" * 80)

        # Step 1: Fetch activities
        activities = self.fetch_all_run_activities(after=after)

        if not activities:
            logger.warning("No activities found matching criteria. Exiting.")
            return pd.DataFrame()

        # Step 2: Extract features
        df = self.process_activities(activities)

        # Step 3: Compute historical features (training load indicators)
        if not df.empty:
            df = self.compute_historical_features(df)

        # Step 4: Save to parquet
        if not df.empty:
            self.save_to_parquet(df)

            # Print summary statistics
            logger.info("=" * 80)
            logger.info("SUMMARY STATISTICS")
            logger.info("=" * 80)
            logger.info(f"Total segment activities: {len(df)}")
            logger.info(f"  - Recovery runs (intensity_level=0): {(df['intensity_level'] == 0).sum()}")
            logger.info(f"  - Training runs (intensity_level=1): {(df['intensity_level'] == 1).sum()}")
            logger.info(f"  - Race runs (intensity_level=2): {(df['intensity_level'] == 2).sum()}")
            logger.info(f"\nKm Effort statistics:")
            logger.info(f"  - Mean: {df['km_effort'].mean():.2f} km")
            logger.info(f"  - Median: {df['km_effort'].median():.2f} km")
            logger.info(f"  - Min: {df['km_effort'].min():.2f} km")
            logger.info(f"  - Max: {df['km_effort'].max():.2f} km")
            logger.info(f"\nElevation gain statistics:")
            logger.info(f"  - Mean: {df['elevation_gain_m'].mean():.1f} m")
            logger.info(f"  - Median: {df['elevation_gain_m'].median():.1f} m")
            logger.info(f"  - Min: {df['elevation_gain_m'].min():.1f} m")
            logger.info(f"  - Max: {df['elevation_gain_m'].max():.1f} m")
            logger.info(f"\nTraining load (elevation_gain_last_30d):")
            logger.info(f"  - Mean: {df['elevation_gain_last_30d'].mean():.1f} m")
            logger.info(f"  - Median: {df['elevation_gain_last_30d'].median():.1f} m")
            logger.info(f"  - Min: {df['elevation_gain_last_30d'].min():.1f} m")
            logger.info(f"  - Max: {df['elevation_gain_last_30d'].max():.1f} m")
            logger.info("=" * 80)

        logger.info("STRAVA TRAINING PIPELINE - COMPLETE")
        logger.info("=" * 80)

        return df
