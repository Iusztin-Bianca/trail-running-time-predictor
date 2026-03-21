"""
Script to run the Strava Training Pipeline.

This script fetches all Run activities from Strava, extracts features, and saves to parquet.

Usage:
    python scripts/run_strava_training_pipeline.py
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from app.data_ingestion.strava_client import StravaClient
from app.data_ingestion.data_ingestion_pipeline import DataIngestionPipeline
from app.feature_engineering import FeatureExtractor
from app.ml.data.blob_storage import BlobStorageManager
from app.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run the Strava training pipeline."""
    try:
        logger.info("=" * 100)
        logger.info("STRAVA TRAINING PIPELINE - STARTING")
        logger.info("=" * 100)

        # Initialize Strava client
        logger.info("Initializing Strava client...")
        client = StravaClient(
            client_id=settings.strava_client_id,
            client_secret=settings.strava_client_secret,
            refresh_token=settings.strava_refresh_token
        )
        logger.info("Strava client initialized successfully")

        # Initialize feature extractor
        logger.info("Initializing feature extractor...")
        feature_extractor = FeatureExtractor()
        logger.info("Feature extractor initialized successfully")

        # Configure output path
        output_path = settings.data_dir / "strava_training_features.parquet"
        logger.info(f"Output file: {output_path}")

        # Initialize Blob Storage Manager (for caching raw activity data)
        blob_manager = None
        if settings.azure_storage_connection_string:
            logger.info("Initializing Blob Storage Manager...")
            blob_manager = BlobStorageManager()
            logger.info("Blob Storage Manager initialized - raw activity data will be cached")
        else:
            logger.warning("Azure Storage not configured - raw data won't be cached")

        # Initialize and run pipeline
        logger.info("Initializing Strava training pipeline...")
        pipeline = DataIngestionPipeline(
            strava_client=client,
            feature_extractor=feature_extractor,
            output_path=output_path,
            min_elevation_gain_m=100.0,
            blob_manager=blob_manager,
            save_raw_activities=True
        )

        logger.info("Running pipeline...")
        df = pipeline.run()

        # Upload parquet to blob storage
        if not df.empty and blob_manager:
            logger.info("Uploading parquet to Azure Blob Storage...")
            blob_manager.upload_parquet(df, overwrite=True)
            logger.info("Parquet uploaded to blob storage")

        # Display results
        if not df.empty:
            logger.info("=" * 100)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 100)
            logger.info(f"Total activities processed: {len(df)}")
            logger.info(f"Output saved to: {output_path}")
            logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

            # Display sample
            logger.info("\nSample of processed activities (first 5):")
            print("\n" + df[['activity_id', 'activity_name', 'intensity_level',
                            'km_effort', 'elevation_gain_m', 'total_time_sec']].head(5).to_string())
        else:
            logger.warning("=" * 100)
            logger.warning("PIPELINE COMPLETED WITH NO DATA")
            logger.warning("=" * 100)
            logger.warning("No activities were processed. Check your filters and Strava data.")

    except Exception as e:
        logger.error("=" * 100)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 100)
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
