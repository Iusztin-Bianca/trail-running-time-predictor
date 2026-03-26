"""
Point extraction utilities for both Strava streams and GPX files.

This module provides a unified interface for extracting GPS points from different sources:
- Strava API streams (for training data)
- GPX files (for inference/user uploads)
"""
import gpxpy
from io import BytesIO
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from app.constants import GPX_DOWNSAMPLE_THRESHOLD_M, GPX_MAX_POINTS


@dataclass(frozen=True)
class Point:
    """Represents a single GPS point with all required data."""
    latitude: float
    longitude: float
    elevation: float
    time: Optional[datetime]


class PointExtractor:
    """
    Handles extraction of GPS points from different data sources.

    This class provides unified point extraction that works for:
    - Training: Extract points from Strava API streams
    - Inference: Extract points from user-uploaded GPX files
    """

    @classmethod
    def extract_from_streams(
        cls,
        streams: Dict,
        start_time: datetime
    ) -> List[Point]:
        """
        Extract points from Strava API streams.

        This is used during TRAINING when fetching data from Strava API.

        Args:
            streams: Dictionary from Strava API with keys "latlng", "altitude", "time"
                     Example: {"latlng": {"data": [[lat1, lng1], ...]}, ...}
            start_time: Activity start datetime (from activity["start_date"])

        Returns:
            List of Point objects with latitude, longitude, elevation, and time

        Raises:
            ValueError: If required stream data is missing or invalid
        """
        # Extract stream data
        latlng_data = streams.get("latlng", {}).get("data", [])
        altitude_data = streams.get("altitude", {}).get("data", [])
        time_data = streams.get("time", {}).get("data", [])

        # Validate required data
        if not latlng_data:
            raise ValueError("No GPS data (latlng) in streams")
        if not altitude_data:
            raise ValueError("No altitude data in streams")
        if not time_data:
            raise ValueError("No time data in streams")

        if len(latlng_data) != len(altitude_data) or len(latlng_data) != len(time_data):
            raise ValueError(
                f"Stream data length mismatch: "
                f"latlng={len(latlng_data)}, altitude={len(altitude_data)}, time={len(time_data)}"
            )

        # Build points list
        points = []
        for i, (lat, lng) in enumerate(latlng_data):
            elevation = altitude_data[i]
            seconds_offset = time_data[i]

            # Calculate absolute time
            point_time = start_time + timedelta(seconds=seconds_offset)

            points.append(Point(
                latitude=lat,
                longitude=lng,
                elevation=elevation,
                time=point_time
            ))

        return points

    @classmethod
    def extract_from_gpx(cls, gpx_bytes: bytes) -> List[Point]:
        """
        Extract points from a GPX file.

        This is used during INFERENCE when users upload GPX files.

        Args:
            gpx_bytes: GPX file content as bytes

        Returns:
            List of Point objects with latitude, longitude, elevation, and time

        Raises:
            ValueError: If GPX has no valid points or missing required data
            gpxpy.gpx.GPXException: If GPX parsing fails

        """
        # Parse GPX
        try:
            gpx = gpxpy.parse(BytesIO(gpx_bytes))
        except Exception as e:
            raise ValueError(f"Failed to parse GPX file: {e}")

        points = []

        def collect_points_from_segment(segment_points):
            """Helper to collect points from track segments or routes."""
            for p in segment_points:
                # Require position and elevation; time is optional (route GPX files have no timestamps)
                if (p.latitude is not None and
                    p.longitude is not None and
                    p.elevation is not None):

                    points.append(Point(
                        latitude=p.latitude,
                        longitude=p.longitude,
                        elevation=p.elevation,
                        time=p.time  # may be None for planned routes
                    ))

        # Extract from tracks
        for track in gpx.tracks:
            for segment in track.segments:
                collect_points_from_segment(segment.points)

        # Extract from routes (some GPX files only have routes)
        for route in gpx.routes:
            collect_points_from_segment(route.points)

        # Validate we got data
        if len(points) < 2:
            raise ValueError(
                f"GPX file contains insufficient valid points (found {len(points)}, need at least 2)"
            )

        # Downsample only for long races (>30 km) to improve latency (GPX_DOWNSAMPLE_THRESHOLD_M=30000)
        # Gradient window is 30m — one point every ~10m is more than sufficient.
        # Keeps the last point to preserve total distance accuracy.
        # GPX_MAX_POINTS = 5000 
        total_distance_m = gpx.length_2d()
        if total_distance_m > GPX_DOWNSAMPLE_THRESHOLD_M and len(points) > GPX_MAX_POINTS:
            stride = len(points) // GPX_MAX_POINTS
            last = points[-1]
            points = points[::stride]
            if points[-1] is not last:
                points = list(points) + [last]

        return points
