"""
Script to upload segment-based training dataset to Azure Blob Storage.

This uploads the segments parquet file created by run_segment_pipeline.py.
Segments are broken down by gradient changes and stored separately from
the raw per-activity data in blob storage.

Usage:
    python scripts/upload_segments_parquet_to_blob.py
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
    """Upload segments parquet file to Azure Blob Storage."""
    try:
        logger.info("=" * 100)
        logger.info("UPLOADING SEGMENTS PARQUET TO AZURE BLOB STORAGE")
        logger.info("=" * 100)

        # Check if local segments parquet file exists
        local_parquet_path = settings.data_dir / "segments_training_features.parquet"

        if not local_parquet_path.exists():
            logger.error(f"Local segments parquet file not found: {local_parquet_path}")
            logger.error("Please run 'python scripts/run_segment_pipeline.py' first to create the segments dataset.")
            sys.exit(1)

        # Load local parquet
        logger.info(f"Loading local segments parquet file: {local_parquet_path}")
        df = pd.read_parquet(local_parquet_path)
        logger.info(f"Loaded {len(df)} segments from local file")

        # Display segment statistics
        logger.info("\nSegment statistics:")
        logger.info(f"  - Total segments: {len(df)}")
        logger.info(f"  - Unique activities: {df['activity_id'].nunique()}")
        logger.info(f"  - Gradient distribution:")
        logger.info(f"      Uphill: {(df['max_uphill_gradient'] > 0).sum()} ({(df['max_uphill_gradient'] > 0).sum() / len(df) * 100:.1f}%)")
        logger.info(f"      Downhill: {(df['max_downhill_gradient'] > 0).sum()} ({(df['max_downhill_gradient'] > 0).sum() / len(df) * 100:.1f}%)")

        # Initialize Blob Storage Manager with custom blob name for segments
        logger.info("\nInitializing Blob Storage Manager...")
        blob_manager = BlobStorageManager(blob_name="segments_training_features.parquet")

        # Check if blob already exists
        if blob_manager.blob_exists():
            logger.warning("⚠️  Segments blob already exists in storage - will overwrite!")
        else:
            logger.info("✓ No existing segments blob found - creating new one")

        # Upload to blob storage
        logger.info(f"\nUploading to Azure Blob Storage...")
        blob_manager.upload_parquet(df, overwrite=True)

        logger.info("\n" + "=" * 100)
        logger.info("UPLOAD COMPLETE ✓")
        logger.info("=" * 100)
        logger.info(f"Container: {blob_manager.container_name}")
        logger.info(f"Blob name: {blob_manager.blob_name}")
        logger.info(f"Total segments uploaded: {len(df)}")
        logger.info(f"File size: {local_parquet_path.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info("=" * 100)
        logger.info("\nNext steps:")
        logger.info("1. Update ML training pipeline to use segments dataset")
        logger.info("2. Retrain models with segment-based features")
        logger.info("3. Compare performance with per-activity models")

    except Exception as e:
        logger.error("=" * 100)
        logger.error(f"UPLOAD FAILED: {e}")
        logger.error("=" * 100)
        logger.error("Error details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
