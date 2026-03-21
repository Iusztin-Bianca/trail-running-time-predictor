"""
Test script for data initialization functionality.

This script tests the automatic data initialization logic that runs at application startup.
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app.data_ingestion import StravaClient, DataIngestionPipeline
from app.feature_engineering import FeatureExtractor
from app.ml.data import BlobStorageManager, DataInitializer
from app.config.settings import settings


def main():
    """
    Test the data initialization logic.

    This simulates what happens when the FastAPI application starts up.
    """
    print("=" * 100)
    print("TESTING DATA INITIALIZATION")
    print("=" * 100)
    print()
    print("This script will:")
    print("1. Check if training data exists in Azure Blob Storage")
    print("2. If no data exists, fetch ALL historical Strava activities")
    print("3. Upload the initial dataset to Blob Storage")
    print()
    print("NOTE: This is the same logic that runs automatically when the FastAPI app starts.")
    print()

    user_input = input("Do you want to proceed? (yes/no): ").strip().lower()
    if user_input not in ['yes', 'y']:
        print("Test cancelled.")
        return

    print()
    print("=" * 100)
    print("STARTING TEST...")
    print("=" * 100)
    print()

    # Run the initialization
    blob_manager = BlobStorageManager()
    strava_client = StravaClient(
        client_id=settings.strava_client_id,
        client_secret=settings.strava_client_secret,
        refresh_token=settings.strava_refresh_token
    )
    pipeline = DataIngestionPipeline(
        strava_client=strava_client,
        feature_extractor=FeatureExtractor(),
        min_elevation_gain_m=100.0
    )
    DataInitializer(blob_manager=blob_manager, pipeline=pipeline).initialize()

    print()
    print("=" * 100)
    print("TEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
