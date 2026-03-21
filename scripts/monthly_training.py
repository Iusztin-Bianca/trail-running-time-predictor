"""
Monthly Strava Training Data Update

Runs on the 1st of every month via GitHub Actions.
Fetches new Strava activities, updates the parquet file in Blob Storage,
retrains models, and saves the new model version.

This script mirrors the logic in azure_functions/strava_monthly_update/__init__.py.
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.data_ingestion.strava_client import StravaClient
from app.data_ingestion.data_ingestion_pipeline import DataIngestionPipeline
from app.feature_engineering.features import FeatureExtractor
from app.ml.data.blob_storage import BlobStorageManager
from app.ml.data.data_splitter import TemporalSplitter
from app.ml.evaluation.metrics import MetricsCalculator
from app.ml.services.model_comparisor import ModelComparisonService
from app.ml.services.model_persistence import ModelPersistenceService
from app.ml.services.trainer import ModelTrainer
from app.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _retrain_and_save(blob_manager: BlobStorageManager) -> None:
    """Train all models, run comparison, and persist new XGBoost version to Blob Storage."""
    logger.info("Starting model retraining...")
    splitter = TemporalSplitter()
    metrics = MetricsCalculator()

    segments_blob = BlobStorageManager(blob_name="segments_training_features.parquet")
    df = segments_blob.download_parquet()
    if df is None or df.empty:
        raise ValueError("No segment training data available in Blob Storage.")

    model_results = ModelTrainer.train_all(df, splitter, metrics)

    try:
        ModelComparisonService().run(df, model_results)
    except Exception as e:
        logger.warning("Model comparison/SHAP skipped (non-critical): %s", e)

    ModelPersistenceService(blob_manager).save(model_results)

    logger.info("Retraining complete!")


def main() -> None:
    utc_timestamp = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 100)
    logger.info("MONTHLY STRAVA UPDATE - Starting at %s", utc_timestamp)
    logger.info("=" * 100)

    try:
        # Initialize Blob Storage Manager
        logger.info("Initializing Blob Storage Manager...")
        blob_manager = BlobStorageManager(blob_name="segments_training_features.parquet")

        # Initialize Strava client
        logger.info("Initializing Strava client...")
        strava_client = StravaClient(
            client_id=settings.strava_client_id,
            client_secret=settings.strava_client_secret,
            refresh_token=settings.strava_refresh_token,
        )

        # Initialize feature extractor
        logger.info("Initializing feature extractor...")
        feature_extractor = FeatureExtractor()

        # Initialize pipeline
        logger.info("Initializing Strava training pipeline...")
        pipeline = DataIngestionPipeline(
            strava_client=strava_client,
            feature_extractor=feature_extractor,
            output_path=Path("/tmp/strava_training_features.parquet"),
            min_elevation_gain_m=100.0,
            blob_manager=blob_manager,
            save_raw_activities=True,
        )

        # Get last activity timestamp
        logger.info("Checking for existing data in Blob Storage...")
        last_activity_timestamp = blob_manager.get_last_activity_timestamp()

        if last_activity_timestamp is None:
            # No existing data - create initial dataset with ALL activities
            logger.info("=" * 100)
            logger.info("NO EXISTING DATA FOUND - CREATING INITIAL DATASET")
            logger.info("=" * 100)
            logger.info("Fetching ALL historical activities from Strava...")

            df = pipeline.run()

            if df.empty:
                logger.warning("No activities found matching criteria.")
                logger.info("=" * 100)
                logger.info("MONTHLY STRAVA UPDATE - COMPLETE (No activities to process)")
                logger.info("=" * 100)
                return

            logger.info("Uploading initial dataset with %d activities to Blob Storage...", len(df))
            blob_manager.upload_parquet(df, overwrite=True)

            logger.info("=" * 100)
            logger.info("INITIAL DATASET CREATED - %d activities uploaded", len(df))
            logger.info("=" * 100)

            _retrain_and_save(blob_manager)

            logger.info("=" * 100)
            logger.info("MONTHLY STRAVA UPDATE - COMPLETE (initial dataset + model trained)")
            logger.info("=" * 100)
            return

        # Existing data found - fetch only NEW activities
        logger.info(
            "Last activity timestamp: %s (%s)",
            last_activity_timestamp,
            datetime.fromtimestamp(last_activity_timestamp),
        )
        logger.info("Fetching new activities after %s...", datetime.fromtimestamp(last_activity_timestamp))
        new_df = pipeline.run(after=last_activity_timestamp)

        if new_df.empty:
            logger.info("No new activities found.")
            if not blob_manager.model_exists():
                logger.info("No model found either — running training on existing data...")
                _retrain_and_save(blob_manager)
                logger.info("=" * 100)
                logger.info("MONTHLY STRAVA UPDATE - COMPLETE (no new data, but model trained)")
                logger.info("=" * 100)
            else:
                logger.info("=" * 100)
                logger.info("MONTHLY STRAVA UPDATE - COMPLETE (No new data)")
                logger.info("=" * 100)
            return

        # Append new data to existing parquet in Blob Storage
        logger.info("Appending %d new activities to Blob Storage...", len(new_df))
        blob_manager.append_and_upload(new_df)

        logger.info("=" * 100)
        logger.info("DATA UPDATE COMPLETE - %d new activities added", len(new_df))
        logger.info("=" * 100)

        _retrain_and_save(blob_manager)

        logger.info("=" * 100)
        logger.info(
            "MONTHLY STRAVA UPDATE - COMPLETE (%d new activities + model retrained)",
            len(new_df),
        )
        logger.info("=" * 100)

    except Exception as e:
        logger.error("=" * 100)
        logger.error("MONTHLY STRAVA UPDATE - FAILED: %s", e)
        logger.error("=" * 100)
        logger.error("Error details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
