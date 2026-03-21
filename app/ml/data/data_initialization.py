"""
Data Initialization Module

Handles automatic initialization of training data in Blob Storage at application startup.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.ml.data.blob_storage import BlobStorageManager

if TYPE_CHECKING:
    from app.data_ingestion.data_ingestion_pipeline import DataIngestionPipeline

logger = logging.getLogger(__name__)


class DataInitializer:
    """Handles initialization and verification of training data in Blob Storage."""

    def __init__(self, blob_manager: BlobStorageManager, pipeline: DataIngestionPipeline):
        self.blob_manager = blob_manager
        self.pipeline = pipeline

    def data_exists(self) -> bool:
        """Check if training data already exists in Blob Storage."""
        return self.blob_manager.get_last_activity_timestamp() is not None

    def create_initial_dataset(self) -> None:
        """Fetch all historical Strava activities and upload as initial dataset."""
        logger.info("Fetching ALL historical activities from Strava...")
        df = self.pipeline.run()

        if df.empty:
            logger.warning("No activities found matching criteria (Run type, elevation gain >= 100m)")
            return

        logger.info(f"Uploading initial dataset with {len(df)} activities to Blob Storage...")
        self.blob_manager.upload_parquet(df, overwrite=True)
        logger.info(f"Initial dataset created - {len(df)} activities uploaded")

    def initialize(self) -> None:
        """
        Initialize training data if it doesn't exist.

        Called at application startup.
        """
        try:
            logger.info("Checking training data in Blob Storage...")

            if self.data_exists():
                logger.info("Training data found in Blob Storage")
                return

            logger.info("No training data found - creating initial dataset...")
            self.create_initial_dataset()

        except Exception as e:
            logger.error(f"Failed to initialize training data: {e}", exc_info=True)
            logger.warning("Application will continue running, but predictions may not work without training data.")
