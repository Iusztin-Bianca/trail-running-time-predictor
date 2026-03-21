"""
Quick script to check Strava workout_type values from API.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data_ingestion.strava_client import StravaClient
from app.config.settings import settings

def main():
    """Check workout_type values from Strava API."""
    print("=" * 100)
    print("CHECKING STRAVA WORKOUT_TYPE VALUES")
    print("=" * 100)

    # Initialize client
    client = StravaClient(
        client_id=settings.strava_client_id,
        client_secret=settings.strava_client_secret,
        refresh_token=settings.strava_refresh_token
    )

    # Fetch recent activities
    print("\nFetching recent Run/TrailRun activities...")
    activities = client.get_activities(page=1, per_page=50)

    # Filter Run/TrailRun
    run_activities = [
        activity for activity in activities
        if activity.get("type") in ["Run", "TrailRun"]
    ]

    print(f"Found {len(run_activities)} Run/TrailRun activities\n")

    # Display workout_type values
    print("=" * 100)
    print("ACTIVITY NAME | WORKOUT_TYPE VALUE")
    print("=" * 100)

    workout_types_found = {}

    for activity in run_activities[:30]:  # Show first 30
        name = activity.get("name", "Unknown")[:50]  # Truncate long names
        # Remove emojis and special characters for console display
        name_safe = name.encode('ascii', 'ignore').decode('ascii')
        workout_type = activity.get("workout_type")

        # Track unique workout_type values
        if workout_type not in workout_types_found:
            workout_types_found[workout_type] = []
        workout_types_found[workout_type].append(name)

        print(f"{name_safe:<50} | {workout_type}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY - UNIQUE WORKOUT_TYPE VALUES")
    print("=" * 100)

    for workout_type, names in sorted(workout_types_found.items(), key=lambda x: (x[0] is None, x[0])):
        # Remove emojis from example name
        example_name = names[0][:60].encode('ascii', 'ignore').decode('ascii')
        print(f"\nworkout_type = {workout_type}:")
        print(f"  Count: {len(names)}")
        print(f"  Example: {example_name}")

    print("\n" + "=" * 100)
    print("STRAVA WORKOUT_TYPE REFERENCE (from Strava docs):")
    print("=" * 100)
    print("  0 = Default / No specific type")
    print("  1 = Race")
    print("  2 = Long Run")
    print("  3 = Workout / Intervals")
    print("  None = No workout_type set")
    print("=" * 100)

if __name__ == "__main__":
    main()
