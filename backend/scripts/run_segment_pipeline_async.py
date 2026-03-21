"""
Async version of segment pipeline - 3-5x faster than sync version.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

from app.data_ingestion.strava_client import StravaClient
from app.feature_engineering.segment_features import SegmentFeatureExtractor
from app.ml.data.blob_storage import BlobStorageManager
from app.config.settings import settings

# Strava workout_type constants
STRAVA_WORKOUT_TYPE_RECOVERY = 2
STRAVA_WORKOUT_TYPE_RACE = 1

# Our intensity_level mapping
INTENSITY_RECOVERY = 0
INTENSITY_TRAINING = 1
INTENSITY_RACE = 2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def get_streams(
    activity_id: int,
    activity: Dict,
    client: StravaClient,
    blob_manager: Optional[BlobStorageManager]
) -> Dict:
    """Get activity streams from blob storage cache or Strava API."""
    # Try blob storage cache first - download_raw_activity returns None if not found
    if blob_manager:
        raw = await asyncio.to_thread(blob_manager.download_raw_activity, activity_id)
        if raw is not None:
            logger.debug(f"Activity {activity_id} loaded from blob storage cache")
            return raw["streams"]

    # Fetch from Strava API
    streams = await client.get_activity_streams(activity_id)

    # Save to blob storage for future runs
    if blob_manager:
        try:
            await asyncio.to_thread(
                blob_manager.upload_raw_activity,
                activity_id, activity, streams, False  # overwrite=False
            )
            logger.debug(f"Activity {activity_id} saved to blob storage cache")
        except Exception as e:
            logger.warning(f"Failed to cache activity {activity_id} in blob storage: {e}")

    return streams


async def process_activity_async(
    activity: Dict,
    client: StravaClient,
    extractor: SegmentFeatureExtractor,
    blob_manager: Optional[BlobStorageManager] = None,
    max_retries: int = 3
) -> pd.DataFrame:
    """Process a single activity asynchronously with blob storage cache and retry logic."""
    activity_id = activity["id"]
    activity_name = activity.get("name", "Unknown")

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

    for attempt in range(max_retries):
        try:
            # Fetch streams from cache or Strava API
            streams = await get_streams(activity_id, activity, client, blob_manager)

            # Extract segments (sync - CPU bound, not worth parallelizing)
            segments = extractor.extract_from_streams(streams, start_time, intensity_level)

            if not segments:
                return pd.DataFrame()

            # Create DataFrame
            df = pd.DataFrame(segments)
            df["activity_id"] = activity_id
            df["activity_name"] = activity_name
            df["start_date"] = start_time.isoformat()

            return df

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for activity "
                    f"{activity_id}: {e}. Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed to process activity {activity_id} after {max_retries} attempts: {e}")

    return pd.DataFrame()


async def main_async():
    """Main async pipeline."""
    logger.info("=" * 100)
    logger.info("ASYNC SEGMENT-BASED TRAINING PIPELINE - STARTING")
    logger.info("=" * 100)

    # Initialize Blob Storage Manager (for caching raw activity data)
    blob_manager = None
    if settings.azure_storage_connection_string:
        logger.info("Initializing Blob Storage Manager...")
        blob_manager = BlobStorageManager()
        logger.info("Blob Storage Manager initialized - raw activity streams will be cached")
    else:
        logger.warning("Azure Storage not configured - streams won't be cached")

    # Initialize async client
    async with StravaClient(
        client_id=settings.strava_client_id,
        client_secret=settings.strava_client_secret,
        refresh_token=settings.strava_refresh_token,
        max_concurrent_requests=10  # Max 10 concurrent API calls
    ) as client:

        # Fetch all activities
        logger.info("Fetching activities from Strava...")
        activities = await client.fetch_all_activities(
            min_elevation_m=150.0,
            min_distance_m=4000.0
        )

        if not activities:
            logger.warning("No activities found!")
            return

        logger.info(f"Processing {len(activities)} activities concurrently...")

        # Initialize extractor
        extractor = SegmentFeatureExtractor()

        # Process all activities concurrently!
        tasks = [
            process_activity_async(activity, client, extractor, blob_manager)
            for activity in activities
        ]

        # Gather results with progress logging
        results = []
        cached = 0
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            df = await coro
            if not df.empty:
                results.append(df)
            logger.info(f"[{i}/{len(activities)}] Completed")

        # Combine results
        if not results:
            logger.warning("No segments extracted!")
            return

        final_df = pd.concat(results, ignore_index=True)

        # Upload parquet to blob storage
        segments_blob_manager = BlobStorageManager(blob_name="segments_training_features.parquet")
        await asyncio.to_thread(segments_blob_manager.upload_parquet, final_df, True)

        # Log statistics
        logger.info("=" * 100)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 100)
        logger.info(f"Total activities processed: {final_df['activity_id'].nunique()}")
        logger.info(f"Total segments created: {len(final_df)}")
        logger.info(f"Uploaded to blob storage: segments_training_features.parquet")

        # Display statistics
        logger.info("\nSEGMENT STATISTICS")
        logger.info("=" * 100)
        logger.info(f"Segments: {len(final_df)} total")

        logger.info(f"\nIntensity distribution:")
        logger.info(f"  - Recovery: {(final_df['intensity_level'] == 0).sum()}")
        logger.info(f"  - Training: {(final_df['intensity_level'] == 1).sum()}")
        logger.info(f"  - Race: {(final_df['intensity_level'] == 2).sum()}")


def main():
    """Entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
    except Exception as e:
        logger.error("=" * 100)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 100)
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
