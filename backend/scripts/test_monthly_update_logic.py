"""
Test the monthly update logic locally without Azure Functions runtime.

This script simulates what the Azure Function does, useful for testing.

Usage:
    python scripts/test_monthly_update_logic.py
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime
from app.data_ingestion import StravaClient, DataIngestionPipeline
from app.feature_engineering import FeatureExtractor
from app.ml.data import BlobStorageManager
from app.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_monthly_update():
    """Test the monthly update logic."""
    try:
        logger.info("=" * 100)
        logger.info("TESTING MONTHLY UPDATE LOGIC")
        logger.info("=" * 100)

        # Step 1: Initialize Blob Storage Manager
        logger.info("\n[STEP 1] Initializing Blob Storage Manager...")
        blob_manager = BlobStorageManager()
        logger.info("✓ Blob Storage Manager initialized")

        # Step 2: Check if blob exists
        logger.info("\n[STEP 2] Checking if parquet exists in Blob Storage...")
        if not blob_manager.blob_exists():
            logger.error("✗ Blob does not exist. Please run upload_initial_parquet_to_blob.py first.")
            return

        logger.info("✓ Blob exists")

        # Step 3: Get last activity timestamp
        logger.info("\n[STEP 3] Getting last activity timestamp...")
        last_activity_timestamp = blob_manager.get_last_activity_timestamp()

        if last_activity_timestamp is None:
            logger.error("✗ Could not get last activity timestamp")
            return

        logger.info(f"✓ Last activity: {datetime.fromtimestamp(last_activity_timestamp)}")
        logger.info(f"  Unix timestamp: {last_activity_timestamp}")

        # Step 4: Initialize Strava client
        logger.info("\n[STEP 4] Initializing Strava client...")
        strava_client = StravaClient(
            client_id=settings.strava_client_id,
            client_secret=settings.strava_client_secret,
            refresh_token=settings.strava_refresh_token
        )
        logger.info("✓ Strava client initialized")

        # Step 5: Initialize feature extractor
        logger.info("\n[STEP 5] Initializing feature extractor...")
        feature_extractor = FeatureExtractor()
        logger.info("✓ Feature extractor initialized")

        # Step 6: Initialize pipeline
        logger.info("\n[STEP 6] Initializing pipeline...")
        pipeline = DataIngestionPipeline(
            strava_client=strava_client,
            feature_extractor=feature_extractor,
            min_elevation_gain_m=100.0
        )
        logger.info("✓ Pipeline initialized")

        # Step 7: Fetch NEW activities only
        logger.info(f"\n[STEP 7] Fetching activities after {datetime.fromtimestamp(last_activity_timestamp)}...")
        new_df = pipeline.run(after=last_activity_timestamp)

        if new_df.empty:
            logger.info("✓ No new activities found (this is normal if you just uploaded initial data)")
            logger.info("\n" + "=" * 100)
            logger.info("TEST COMPLETE - No new activities to update")
            logger.info("=" * 100)
            return

        logger.info(f"✓ Found {len(new_df)} new activities")

        # Step 8: Test append (DRY RUN - don't actually upload)
        logger.info("\n[STEP 8] Simulating append to Blob Storage (DRY RUN)...")
        existing_df = blob_manager.download_parquet()
        logger.info(f"  Existing activities: {len(existing_df)}")
        logger.info(f"  New activities: {len(new_df)}")
        logger.info(f"  Total after merge: {len(existing_df) + len(new_df)}")

        # Ask user if they want to actually upload
        logger.info("\n" + "=" * 100)
        logger.info("TEST COMPLETE - All steps successful!")
        logger.info("=" * 100)
        logger.info(f"\nFound {len(new_df)} new activities ready to append.")

        response = input("\nDo you want to ACTUALLY append these to Blob Storage? (yes/no): ")
        if response.lower() == "yes":
            logger.info("\n[STEP 9] Appending and uploading to Blob Storage...")
            blob_manager.append_and_upload(new_df)
            logger.info("✓ Upload complete!")
        else:
            logger.info("\nSkipped upload. This was a dry run.")

    except Exception as e:
        logger.error("=" * 100)
        logger.error(f"TEST FAILED: {e}")
        logger.error("=" * 100)
        logger.error("Error details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    test_monthly_update()
