"""
Script to inspect ALL fields from Strava API activities.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from app.data_ingestion.strava_client import StravaClient
from app.config.settings import settings

def main():
    """Inspect all fields from Strava API."""
    print("=" * 100)
    print("INSPECTING ALL STRAVA API FIELDS")
    print("=" * 100)

    # Initialize client
    client = StravaClient(
        client_id=settings.strava_client_id,
        client_secret=settings.strava_client_secret,
        refresh_token=settings.strava_refresh_token
    )

    # Fetch one activity
    print("\nFetching recent activities...")
    activities = client.get_activities(page=1, per_page=5)

    if not activities:
        print("No activities found!")
        return

    # Display all fields from first activity
    print("\n" + "=" * 100)
    print("ALL FIELDS FROM FIRST ACTIVITY:")
    print("=" * 100)

    activity = activities[0]
    name = activity.get("name", "Unknown").encode('ascii', 'ignore').decode('ascii')

    print(f"\nActivity: {name}")
    print(f"Type: {activity.get('type')}")
    print("\n" + "-" * 100)
    print("ALL AVAILABLE FIELDS:")
    print("-" * 100)

    # Print all fields sorted alphabetically
    for key in sorted(activity.keys()):
        value = activity[key]
        # Truncate long values
        if isinstance(value, str) and len(value) > 60:
            value = value[:60] + "..."
        print(f"  {key:<30} = {value}")

    # Look specifically for any "recovery" related fields
    print("\n" + "=" * 100)
    print("SEARCHING FOR 'RECOVERY' RELATED FIELDS:")
    print("=" * 100)

    recovery_fields = [k for k in activity.keys() if 'recovery' in k.lower()]
    if recovery_fields:
        print("\nFound recovery-related fields:")
        for field in recovery_fields:
            print(f"  {field} = {activity[field]}")
    else:
        print("\n  No fields containing 'recovery' found!")

    # Check workout_type specifically
    print("\n" + "=" * 100)
    print("WORKOUT_TYPE ANALYSIS:")
    print("=" * 100)
    print(f"\nworkout_type = {activity.get('workout_type')}")
    print(f"Type: {type(activity.get('workout_type'))}")

    # Check all activities for unique workout_type values
    print("\n" + "=" * 100)
    print("UNIQUE WORKOUT_TYPE VALUES (from 50 activities):")
    print("=" * 100)

    activities = client.get_activities(page=1, per_page=50)
    workout_types = {}

    for act in activities:
        wt = act.get('workout_type')
        name = act.get('name', 'Unknown')[:40].encode('ascii', 'ignore').decode('ascii')
        if wt not in workout_types:
            workout_types[wt] = []
        workout_types[wt].append(name)

    for wt, names in sorted(workout_types.items(), key=lambda x: (x[0] is None, x[0])):
        print(f"\nworkout_type = {wt}:")
        print(f"  Count: {len(names)}")
        print(f"  Examples: {names[:3]}")

if __name__ == "__main__":
    main()
