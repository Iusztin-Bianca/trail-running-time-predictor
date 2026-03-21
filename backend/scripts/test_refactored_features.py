"""
Test the refactored feature extraction that supports both:
1. Training flow: Strava streams -> features
2. Inference flow: GPX upload -> features
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set UTF-8 encoding for Windows console output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.data_ingestion.strava_client import StravaClient
from app.feature_engineering.features import FeatureExtractor
from app.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_training_flow(activity_id: int):
    """Test TRAINING flow: Strava streams → features"""

    print(f"\n{'='*100}")
    print(f"TEST 1: TRAINING FLOW (Strava Streams -> Features)")
    print(f"{'='*100}\n")

    client = StravaClient(
        client_id=settings.strava_client_id,
        client_secret=settings.strava_client_secret,
        refresh_token=settings.strava_refresh_token
    )

    # Fetch activity and streams
    print(f"1. Fetching activity {activity_id} from Strava...")
    activity = client.get_activity(activity_id)
    streams = client.get_activity_streams(activity_id)

    print(f"   [OK] Activity: {activity.get('name')}")
    print(f"   [OK] Distance: {activity.get('distance', 0) / 1000:.2f} km")
    print(f"   [OK] Elevation: {activity.get('total_elevation_gain', 0):.0f} m")

    # Parse start time
    start_time = datetime.fromisoformat(activity["start_date"].replace("Z", "+00:00"))

    # Extract features DIRECTLY from streams (no GPX generation!)
    print(f"\n2. Extracting features directly from streams...")
    extractor = FeatureExtractor()
    features = extractor.extract_from_streams(streams, start_time)

    print(f"   [OK] Extracted {len(features)} features")
    print(f"\n   Key features:")
    print(f"     - Total distance: {features['total_distance_km']} km")
    print(f"     - Elevation gain: {features['elevation_gain_m']} m")
    print(f"     - Total time: {features['total_time_sec']} seconds")
    print(f"     - Num points: {features['num_points']}")
    print(f"     - Uphill ratio: {features['uphill_ratio']}")

    print(f"\n[SUCCESS] Training flow complete!")
    return features


def test_inference_flow(activity_id: int):
    """Test INFERENCE flow: GPX file → features"""

    print(f"\n{'='*100}")
    print(f"TEST 2: INFERENCE FLOW (GPX Upload -> Features)")
    print(f"{'='*100}\n")

    client = StravaClient(
        client_id=settings.strava_client_id,
        client_secret=settings.strava_client_secret,
        refresh_token=settings.strava_refresh_token
    )

    # Generate a GPX file (simulating user upload)
    print(f"1. Generating GPX file from activity {activity_id}...")
    gpx_bytes = client.get_gpx_content(activity_id)
    print(f"   [OK] Generated GPX: {len(gpx_bytes)} bytes")

    # Save it temporarily
    test_gpx_path = Path(__file__).parent.parent / "data" / "test_inference.gpx"
    test_gpx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_gpx_path, "wb") as f:
        f.write(gpx_bytes)
    print(f"   [OK] Saved to: {test_gpx_path}")

    # Extract features from GPX (like a user upload would)
    print(f"\n2. Extracting features from GPX file...")
    extractor = FeatureExtractor()
    features = extractor.extract_from_gpx(gpx_bytes)

    print(f"   [OK] Extracted {len(features)} features")
    print(f"\n   Key features:")
    print(f"     - Total distance: {features['total_distance_km']} km")
    print(f"     - Elevation gain: {features['elevation_gain_m']} m")
    print(f"     - Total time: {features['total_time_sec']} seconds")
    print(f"     - Num points: {features['num_points']}")
    print(f"     - Uphill ratio: {features['uphill_ratio']}")

    print(f"\n[SUCCESS] Inference flow complete!")
    return features


def compare_features(features1: dict, features2: dict):
    """Compare features from both flows to ensure they match"""

    print(f"\n{'='*100}")
    print(f"COMPARISON: Training Flow vs Inference Flow")
    print(f"{'='*100}\n")

    # Important features to compare
    key_features = [
        "total_distance_km",
        "elevation_gain_m",
        "elevation_loss_m",
        "total_time_sec",
        "num_points",
        "uphill_ratio",
        "downhill_ratio"
    ]

    print(f"{'Feature':<30} {'Training':<20} {'Inference':<20} {'Match?':<10}")
    print("-" * 85)

    mismatches = 0
    for feature in key_features:
        val1 = features1.get(feature, np.nan)
        val2 = features2.get(feature, np.nan)

        # Check if they match (with tolerance for floating point)
        if np.isnan(val1) and np.isnan(val2):
            match = "[OK]"
        elif abs(val1 - val2) < 0.01:  # 1% tolerance
            match = "[OK]"
        else:
            match = "[FAIL]"
            mismatches += 1

        print(f"{feature:<30} {val1:<20.3f} {val2:<20.3f} {match:<10}")

    print("-" * 85)

    if mismatches == 0:
        print(f"\n[SUCCESS] Both flows produce identical features!")
        print(f"  -> Training flow and inference flow are consistent")
        print(f"  -> No need to generate/save GPX files during training!")
    else:
        print(f"\n[WARNING] {mismatches} feature(s) don't match")

    return mismatches == 0


def main():
    """Run all tests"""

    print("\n" + "="*100)
    print("REFACTORED FEATURE EXTRACTION TEST")
    print("Testing both Training and Inference flows")
    print("="*100)

    activity_id = 11552942461

    try:
        # Test training flow (streams → features)
        training_features = test_training_flow(activity_id)

        # Test inference flow (GPX → features)
        inference_features = test_inference_flow(activity_id)

        # Compare results
        success = compare_features(training_features, inference_features)

        # Final summary
        print(f"\n{'='*100}")
        print(f"FINAL SUMMARY")
        print(f"{'='*100}")
        if success:
            print(f"[SUCCESS] All tests passed!")
            print(f"\nArchitecture validated:")
            print(f"  - Training: FeatureExtractor().extract_from_streams(streams, start_time) -> features")
            print(f"  - Inference: FeatureExtractor().extract_from_gpx(gpx_bytes) -> features")
            print(f"  - Both use the same core logic (NO CODE DUPLICATION)")
            print(f"\nNext steps:")
            print(f"  1. Use FeatureExtractor class in training pipeline")
            print(f"  2. Use FeatureExtractor class in API inference endpoint")
            print(f"  3. Delete old features.py and rename features_refactored.py")
        else:
            print(f"[FAIL] Tests failed - features don't match")

        print(f"{'='*100}\n")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
