"""
Script to upload initial training dataset to Azure Blob Storage.

This should be run ONCE after creating the initial parquet file locally.
After this, the Azure Function will handle monthly incremental updates.

Usage:
    python scripts/upload_initial_parquet_to_blob.py
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
from app.ml.data import BlobStorageManager
from app.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Upload initial parquet file to Azure Blob Storage."""
    try:
        logger.info("=" * 100)
        logger.info("UPLOADING INITIAL PARQUET TO AZURE BLOB STORAGE")
        logger.info("=" * 100)

        # Check if local parquet file exists
        local_parquet_path = settings.data_dir / "strava_training_features.parquet"

        if not local_parquet_path.exists():
            logger.error(f"Local parquet file not found: {local_parquet_path}")
            logger.error("Please run 'python scripts/run_strava_training_pipeline.py' first to create the initial dataset.")
            sys.exit(1)

        # Load local parquet
        logger.info(f"Loading local parquet file: {local_parquet_path}")
        df = pd.read_parquet(local_parquet_path)
        logger.info(f"Loaded {len(df)} activities from local file")

        # Initialize Blob Storage Manager
        logger.info("Initializing Blob Storage Manager...")
        blob_manager = BlobStorageManager()

        # Check if blob already exists
        if blob_manager.blob_exists():
            logger.warning("Blob already exists in storage - will overwrite!")

        # Upload to blob storage
        logger.info(f"Uploading to Azure Blob Storage...")
        blob_manager.upload_parquet(df, overwrite=True)

        logger.info("=" * 100)
        logger.info("UPLOAD COMPLETE")
        logger.info("=" * 100)
        logger.info(f"Container: {blob_manager.container_name}")
        logger.info(f"Blob name: {blob_manager.blob_name}")
        logger.info(f"Total activities: {len(df)}")
        logger.info("=" * 100)
        logger.info("\nNext steps:")
        logger.info("1. Deploy the Azure Function to Azure")
        logger.info("2. The function will automatically update the dataset on the 1st of each month")

    except Exception as e:
        logger.error("=" * 100)
        logger.error(f"UPLOAD FAILED: {e}")
        logger.error("=" * 100)
        logger.error("Error details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
