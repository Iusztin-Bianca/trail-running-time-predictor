import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from app.feature_engineering.features import FeatureExtractor
from app.config import settings

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """Orchestrates the feature engineering process from raw GPX to parquet dataset"""

    def __init__(
        self, input_dir: Optional[Path] = None, output_path: Optional[Path] = None
    ):
        self.input_dir = input_dir or settings.raw_gpx_dir
        self.output_path = output_path or settings.features_output_path
        self.extractor = FeatureExtractor()

    def process_single_gpx(self, gpx_file: Path) -> Optional[Dict[str, float]]:
        """Process a single GPX file and return features

        Args:
            gpx_file: Path to GPX file

        Returns:
            Dictionary of features with activity_id, or None if processing failed
        """
        try:
            with open(gpx_file, "rb") as f:
                features = self.extractor.extract_from_gpx(f.read())
                features["activity_id"] = gpx_file.stem
                return features
        except Exception as e:
            logger.error(f"Failed to process {gpx_file.name}: {e}")
            return None

    def run(self) -> pd.DataFrame:
        """Execute the full feature engineering pipeline

        Returns:
            DataFrame with extracted features
        """
        logger.info("=" * 60)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("=" * 60)

        # Find all GPX files
        gpx_files = list(self.input_dir.glob("*.gpx"))
        logger.info(f"Found {len(gpx_files)} GPX files in {self.input_dir}")

        if not gpx_files:
            logger.warning("No GPX files found to process!")
            return pd.DataFrame()

        # Process each GPX file
        rows = []
        successful = 0
        failed = 0

        for i, gpx_file in enumerate(gpx_files, 1):
            logger.info(f"Processing {i}/{len(gpx_files)}: {gpx_file.name}")
            features = self.process_single_gpx(gpx_file)

            if features:
                rows.append(features)
                successful += 1
            else:
                failed += 1

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Save to parquet
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, index=False)

        logger.info("=" * 60)
        logger.info(f"Feature Engineering Complete")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Output saved to: {self.output_path}")
        logger.info("=" * 60)

        return df
