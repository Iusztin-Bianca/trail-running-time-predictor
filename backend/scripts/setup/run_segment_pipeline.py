"""
Script to generate segment-based training dataset from Strava activities.

This script:
1. Fetches all Run/TrailRun activities from Strava
2. Breaks each activity into segments based on gradient changes
3. Each segment becomes a separate observation
4. Saves to segments_training_features.parquet

Expected outcome: ~69 activities → hundreds/thousands of segments
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
from datetime import datetime
from app.data_ingestion.strava_client import StravaClient
from app.feature_engineering.segment_features import SegmentFeatureExtractor
from app.config.settings import settings

# Strava workout_type constants
STRAVA_WORKOUT_TYPE_RECOVERY = 2
STRAVA_WORKOUT_TYPE_RACE = 1

# Our intensity_level mapping
INTENSITY_RECOVERY = 0
INTENSITY_TRAINING = 1
INTENSITY_RACE = 2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def fetch_activities(client: StravaClient, min_elevation_m: float = 150.0, min_distance_m: float = 4000.0) -> list:
    """Fetch all Run and TrailRun activities from Strava.

    Args:
        client: StravaClient instance
        min_elevation_m: Minimum elevation gain threshold (default: 150m)
        min_distance_m: Minimum distance threshold (default: 4000m = 4km)

    Returns:
        List of activities matching criteria
    """
    logger.info("Fetching Run and TrailRun activities from Strava...")

    all_activities = []
    page = 1
    per_page = 200

    while True:
        activities = client.get_activities(page=page, per_page=per_page)

        if not activities:
            logger.info(f"No more activities found (page {page})")
            break

        # Filter Run/TrailRun with elevation >= threshold AND distance >= threshold
        filtered = [
            activity for activity in activities
            if activity.get("type") in ["Run", "TrailRun"]
            and activity.get("total_elevation_gain", 0) >= min_elevation_m
            and activity.get("distance", 0) >= min_distance_m
        ]

        all_activities.extend(filtered)

        if len(activities) < per_page:
            break

        page += 1

    logger.info(f"Fetched {len(all_activities)} activities with elevation >= {min_elevation_m}m and distance >= {min_distance_m}m")
    return all_activities


def process_activity_to_segments(
    activity: dict,
    client: StravaClient,
    extractor: SegmentFeatureExtractor
) -> pd.DataFrame:
    """Process a single activity into segments.

    Args:
        activity: Activity dict from Strava API
        client: Strava client for fetching streams
        extractor: Segment feature extractor

    Returns:
        DataFrame with segments from this activity
    """
    activity_id = activity["id"]
    activity_name = activity.get("name", "Unknown")

    logger.info(f"Processing activity {activity_id}: {activity_name}")

    try:
        # Determine intensity level
        workout_type = activity.get("workout_type")
        if workout_type == STRAVA_WORKOUT_TYPE_RECOVERY:
            intensity_level = INTENSITY_RECOVERY
        elif workout_type == STRAVA_WORKOUT_TYPE_RACE:
            intensity_level = INTENSITY_RACE
        else:
            intensity_level = INTENSITY_TRAINING

        # Parse start time
        start_time = datetime.fromisoformat(
            activity["start_date"].replace("Z", "+00:00")
        )

        # Fetch streams
        streams = client.get_activity_streams(activity_id)

        # Extract segments
        segments = extractor.extract_from_streams(streams, start_time, intensity_level)

        if not segments:
            logger.warning(f"No segments extracted from activity {activity_id}")
            return pd.DataFrame()

        # Add activity metadata to each segment
        for segment in segments:
            segment['activity_id'] = activity_id
            segment['activity_name'] = activity_name
            segment['start_date'] = start_time.isoformat()

        logger.info(f"Extracted {len(segments)} segments from {activity_name}")
        return pd.DataFrame(segments)

    except Exception as e:
        logger.error(f"Failed to process activity {activity_id}: {e}", exc_info=True)
        return pd.DataFrame()


def main():
    """Run the segment-based training pipeline."""
    try:
        logger.info("=" * 100)
        logger.info("SEGMENT-BASED TRAINING PIPELINE - STARTING")
        logger.info("=" * 100)

        # Initialize Strava client
        logger.info("Initializing Strava client...")
        client = StravaClient(
            client_id=settings.strava_client_id,
            client_secret=settings.strava_client_secret,
            refresh_token=settings.strava_refresh_token
        )

        # Initialize segment extractor
        logger.info("Initializing segment feature extractor...")
        extractor = SegmentFeatureExtractor()

        # Fetch activities (same filters as per-activity pipeline)
        activities = fetch_activities(client, min_elevation_m=150.0, min_distance_m=4000.0)

        if not activities:
            logger.warning("No activities found!")
            return

        # Process all activities into segments
        logger.info(f"Processing {len(activities)} activities into segments...")
        all_segments = []

        for i, activity in enumerate(activities, 1):
            logger.info(f"[{i}/{len(activities)}] Processing activity...")
            segments_df = process_activity_to_segments(activity, client, extractor)

            if not segments_df.empty:
                all_segments.append(segments_df)

        # Combine all segments
        if not all_segments:
            logger.warning("No segments were created!")
            return

        final_df = pd.concat(all_segments, ignore_index=True)

        # Save to parquet
        output_path = settings.data_dir / "segments_training_features.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        final_df.to_parquet(output_path, index=False, engine='pyarrow')

        logger.info("=" * 100)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 100)
        logger.info(f"Total activities processed: {len(activities)}")
        logger.info(f"Total segments created: {len(final_df)}")
        logger.info(f"Average segments per activity: {len(final_df) / len(activities):.1f}")
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        # Display statistics
        logger.info("\n" + "=" * 100)
        logger.info("SEGMENT STATISTICS")
        logger.info("=" * 100)
        logger.info(f"\nGradient distribution:")
        logger.info(f"  - Flat (avg_gradient=0): {(final_df['avg_gradient'] == 0).sum()}")
        logger.info(f"  - Uphill (max_uphill_gradient>0): {(final_df['max_uphill_gradient'] > 0).sum()}")
        logger.info(f"  - Downhill (max_downhill_gradient>0): {(final_df['max_downhill_gradient'] > 0).sum()}")

        logger.info(f"\nIntensity distribution:")
        logger.info(f"  - Recovery: {(final_df['intensity_level'] == 0).sum()}")
        logger.info(f"  - Training: {(final_df['intensity_level'] == 1).sum()}")
        logger.info(f"  - Race: {(final_df['intensity_level'] == 2).sum()}")

        logger.info(f"\nSegment distance (m):")
        logger.info(f"  - Mean: {final_df['segment_distance_m'].mean():.1f}")
        logger.info(f"  - Median: {final_df['segment_distance_m'].median():.1f}")
        logger.info(f"  - Min: {final_df['segment_distance_m'].min():.1f}")
        logger.info(f"  - Max: {final_df['segment_distance_m'].max():.1f}")

        logger.info(f"\nSegment time (sec):")
        logger.info(f"  - Mean: {final_df['segment_time_sec'].mean():.1f}")
        logger.info(f"  - Median: {final_df['segment_time_sec'].median():.1f}")
        logger.info(f"  - Min: {final_df['segment_time_sec'].min():.1f}")
        logger.info(f"  - Max: {final_df['segment_time_sec'].max():.1f}")

        logger.info(f"\nAverage gradient:")
        logger.info(f"  - Mean: {final_df['avg_gradient'].mean():.4f} ({final_df['avg_gradient'].mean() * 100:.2f}%)")
        logger.info(f"  - Min: {final_df['avg_gradient'].min():.4f} ({final_df['avg_gradient'].min() * 100:.2f}%)")
        logger.info(f"  - Max: {final_df['avg_gradient'].max():.4f} ({final_df['avg_gradient'].max() * 100:.2f}%)")

        # Sample of first 5 segments
        logger.info("\n" + "=" * 100)
        logger.info("SAMPLE SEGMENTS (first 5):")
        logger.info("=" * 100)
        print("\n" + final_df[['activity_name', 'segment_distance_m', 'segment_time_sec',
                                 'segment_pace_mps', 'avg_gradient',
                                 'intensity_level']].head(5).to_string())

        logger.info("\n" + "=" * 100)
        logger.info("SEGMENT PIPELINE COMPLETE!")
        logger.info("=" * 100)

    except Exception as e:
        logger.error("=" * 100)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 100)
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
