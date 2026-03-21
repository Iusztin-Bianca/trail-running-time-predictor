"""
Test to verify that our Haversine implementation matches gpxpy's distance_2d()
"""
import sys
from pathlib import Path
import gpxpy
from math import radians, sin, cos, sqrt, atan2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data_ingestion.strava_client import StravaClient
from app.config.settings import settings


def compute_distance_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Our implementation of Haversine formula"""
    R = 6371000  # Earth radius in meters

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)

    a = (sin(delta_lat / 2) ** 2 +
         cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def test_distance_calculations():
    """Compare our Haversine vs gpxpy's distance_2d()"""

    print(f"\n{'='*100}")
    print("DISTANCE CALCULATION COMPARISON TEST")
    print(f"{'='*100}\n")

    # Create Strava client
    client = StravaClient(
        client_id=settings.strava_client_id,
        client_secret=settings.strava_client_secret,
        refresh_token=settings.strava_refresh_token
    )

    activity_id = 11552942461

    # Get GPX file
    print(f"1. Fetching GPX for activity {activity_id}...")
    gpx_content = client.get_gpx_content(activity_id)

    # Parse GPX
    from io import BytesIO
    gpx = gpxpy.parse(BytesIO(gpx_content))

    # Get first track segment points
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            points = segment.points
            break
        break

    print(f"   ✓ Found {len(points)} GPS points in GPX")

    # Compare distance calculations for first 20 consecutive points
    print(f"\n2. Comparing distance calculations (first 20 point pairs)...\n")

    print(f"{'Pair':<8} {'GPXpy distance_2d':<25} {'Our Haversine':<25} {'Difference (m)':<20} {'Match?'}")
    print("-" * 100)

    total_diff = 0
    max_diff = 0
    mismatches = 0

    for i in range(min(20, len(points) - 1)):
        prev = points[i]
        curr = points[i + 1]

        # GPXpy's built-in method
        gpxpy_dist = prev.distance_2d(curr)

        # Our Haversine implementation
        our_dist = compute_distance_haversine(
            prev.latitude, prev.longitude,
            curr.latitude, curr.longitude
        )

        # Calculate difference
        diff = abs(gpxpy_dist - our_dist)
        total_diff += diff
        max_diff = max(max_diff, diff)

        # Check if they match (within 1mm tolerance)
        match = "✓" if diff < 0.001 else "✗"
        if diff >= 0.001:
            mismatches += 1

        print(f"{i:<8} {gpxpy_dist:<25.6f} {our_dist:<25.6f} {diff:<20.6f} {match}")

    print("-" * 100)

    avg_diff = total_diff / min(20, len(points) - 1)

    print(f"\nStatistics:")
    print(f"  Average difference: {avg_diff:.6f} meters ({avg_diff * 1000:.3f} mm)")
    print(f"  Maximum difference: {max_diff:.6f} meters ({max_diff * 1000:.3f} mm)")
    print(f"  Mismatches (>1mm): {mismatches}")

    if mismatches == 0 and avg_diff < 0.000001:
        print(f"\n✓ SUCCESS: Both methods produce identical results!")
        print(f"  → Our Haversine implementation matches gpxpy's distance_2d()")
        print(f"  → Safe to use for both training (streams) and inference (GPX)")
    elif max_diff < 0.01:  # Less than 1cm
        print(f"\n✓ ACCEPTABLE: Differences are negligible (<1cm)")
        print(f"  → Tiny numerical differences due to floating point precision")
        print(f"  → Safe to use in production")
    else:
        print(f"\n⚠ WARNING: Significant differences detected!")
        print(f"  → Need to investigate why results don't match")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    test_distance_calculations()
