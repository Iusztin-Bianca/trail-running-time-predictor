"""Test script to debug _generate_gpx and inspect trackpoints"""
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data_ingestion.strava_client import StravaClient
from app.config.settings import settings

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_generate_gpx_for_activity(activity_id: int):
    """Test GPX generation for a specific activity and print trackpoints info"""

    print(f"\n{'='*80}")
    print(f"Testing GPX Generation for Activity ID: {activity_id}")
    print(f"{'='*80}\n")

    # Create Strava client
    client = StravaClient(
        client_id=settings.strava_client_id,
        client_secret=settings.strava_client_secret,
        refresh_token=settings.strava_refresh_token
    )

    try:
        # Fetch activity details
        print("1. Fetching activity details...")
        activity = client.get_activity(activity_id)
        print(f"   ✓ Activity Name: {activity.get('name')}")
        print(f"   ✓ Activity Type: {activity.get('type')}")
        print(f"   ✓ Distance: {activity.get('distance', 0) / 1000:.2f} km")
        print(f"   ✓ Elevation Gain: {activity.get('total_elevation_gain', 0):.0f} m")
        print(f"   ✓ Start Date: {activity.get('start_date')}")

        # Fetch streams
        print("\n2. Fetching activity streams...")
        streams = client.get_activity_streams(activity_id)

        # Extract stream data
        latlng_data = streams.get("latlng", {}).get("data", [])
        altitude_data = streams.get("altitude", {}).get("data", [])
        time_data = streams.get("time", {}).get("data", [])

        print(f"   ✓ GPS Points: {len(latlng_data)}")
        print(f"   ✓ Altitude Points: {len(altitude_data)}")
        print(f"   ✓ Time Points: {len(time_data)}")

        if latlng_data:
            print(f"\n3. Sample Data (first 5 points):")
            print(f"   {'Index':<8} {'Latitude':<12} {'Longitude':<12} {'Altitude (m)':<15} {'Time (s)':<10}")
            print(f"   {'-'*70}")
            for i in range(min(5, len(latlng_data))):
                lat, lng = latlng_data[i]
                alt = altitude_data[i] if i < len(altitude_data) else "N/A"
                time_s = time_data[i] if i < len(time_data) else "N/A"
                print(f"   {i:<8} {lat:<12.6f} {lng:<12.6f} {alt:<15} {time_s:<10}")

        # Generate GPX
        print("\n4. Generating GPX file...")
        gpx_content = client.get_gpx_content(activity_id)

        # Save to test file
        test_file = Path(__file__).parent.parent / "data" / "test_output.gpx"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, "wb") as f:
            f.write(gpx_content)

        print(f"   ✓ GPX Generated: {len(gpx_content)} bytes")
        print(f"   ✓ Saved to: {test_file}")

        # Show GPX preview (first 1000 chars)
        print(f"\n5. GPX Preview (first 1000 characters):")
        print(f"   {'-'*70}")
        gpx_str = gpx_content.decode('utf-8')
        print(f"   {gpx_str[:1000]}")
        if len(gpx_str) > 1000:
            print(f"   ... ({len(gpx_str) - 1000} more characters)")

        print(f"\n{'='*80}")
        print(f"✓ Test Completed Successfully!")
        print(f"{'='*80}\n")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n✗ Test Failed: {e}\n")
        raise


def list_recent_activities():
    """List recent activities to help choose an activity_id"""

    print(f"\n{'='*80}")
    print("Fetching Recent Activities...")
    print(f"{'='*80}\n")

    client = StravaClient(
        client_id=settings.strava_client_id,
        client_secret=settings.strava_client_secret,
        refresh_token=settings.strava_refresh_token
    )

    try:
        activities = client.get_activities(page=1, per_page=10)

        print(f"Found {len(activities)} recent activities:\n")
        print(f"{'ID':<12} {'Name':<40} {'Type':<10} {'Date':<20} {'Elevation (m)':<15}")
        print(f"{'-'*110}")

        for act in activities:
            act_id = act.get('id')
            name = act.get('name', 'Unknown')[:40]
            act_type = act.get('type', 'N/A')
            date = act.get('start_date', 'N/A')[:10]
            elevation = act.get('total_elevation_gain', 0)
            print(f"{act_id:<12} {name:<40} {act_type:<10} {date:<20} {elevation:<15.0f}")

        print(f"\n{'='*80}\n")
        return activities

    except Exception as e:
        logger.error(f"Failed to list activities: {e}", exc_info=True)
        raise


def main():
    """Main test function"""

    print("\n" + "="*80)
    print("GPX GENERATION TEST TOOL")
    print("="*80)

    # First, list recent activities
    activities = list_recent_activities()

    if not activities:
        print("No activities found!")
        return

    # Use the first activity for testing
    #first_activity_id = activities[0]['id']
    first_activity_id = '11552942461'

    print(f"\nTesting with first activity: {first_activity_id}")
    print("(You can edit the script to test a different activity_id)\n")

    # Test GPX generation
    test_generate_gpx_for_activity(first_activity_id)


if __name__ == "__main__":
    main()
