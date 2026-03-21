"""
Run the feature engineering pipeline to process GPX files into features.

Usage:
    python -m scripts.run_feature_engineering

This script processes all GPX files in the raw_gpx directory and creates
a parquet file with extracted features for machine learning.
"""

import logging
from app.feature_engineering.pipeline import FeatureEngineeringPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    # Run the pipeline
    pipeline = FeatureEngineeringPipeline()
    df = pipeline.run()

    # Print summary
    if not df.empty:
        print(f"\n✓ Success! Processed {len(df)} activities")
        print(f"Features saved to: {pipeline.output_path}")
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {', '.join(df.columns.tolist())}")
    else:
        print("\n⚠ No GPX files were processed. Check the input directory.")


if __name__ == "__main__":
    main()
