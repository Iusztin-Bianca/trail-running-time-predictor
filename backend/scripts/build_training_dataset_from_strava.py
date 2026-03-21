"""
Script to build training dataset from Strava activities.

This script:
1. Fetches all Run activities from Strava with elevation gain >= 100m
2. Extracts features using FeatureExtractor (directly from streams - no GPX needed!)
3. Saves features + metadata to parquet file for ML training

Usage:
    python -m scripts.build_training_dataset_from_strava
"""
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data_ingestion.strava_client import StravaClient
from app.data_ingestion.data_ingestion_pipeline import DataIngestionPipeline
from app.feature_engineering import FeatureExtractor
from app.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('strava_training_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run the Strava training pipeline."""
    logger.info("=" * 100)
    logger.info("STRAVA TRAINING DATASET BUILDER")
    logger.info("=" * 100)

    try:
        # Initialize Strava client
        logger.info("Initializing Strava client...")
        client = StravaClient(
            client_id=settings.strava_client_id,
            client_secret=settings.strava_client_secret,
            refresh_token=settings.strava_refresh_token
        )

        # Initialize feature extractor
        logger.info("Initializing feature extractor...")
        feature_extractor = FeatureExtractor()

        # Initialize pipeline
        output_path = settings.data_dir / "strava_training_features.parquet"
        logger.info(f"Output path: {output_path}")

        pipeline = DataIngestionPipeline(
            strava_client=client,
            feature_extractor=feature_extractor,
            output_path=output_path,
            min_elevation_gain_m=100.0  # Only activities with >= 100m elevation gain
        )

        # Run pipeline
        df = pipeline.run()

        if not df.empty:
            logger.info(f"\n[SUCCESS] Dataset created with {len(df)} activities!")
            logger.info(f"Saved to: {output_path}")

            # Show sample
            logger.info("\nSample of first 3 activities:")
            print(df[['activity_id', 'activity_name', 'intensity_level', 'total_distance_km',
                      'elevation_gain_m', 'total_time_sec']].head(3).to_string())
        else:
            logger.warning("\n[WARNING] No activities were processed. Dataset is empty.")

    except Exception as e:
        logger.error(f"\n[ERROR] Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("\n" + "=" * 100)
    logger.info("DONE")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
