"""
Feature extraction from GPS point data.

This module provides unified feature extraction that works for both:
- Training data (from Strava API streams)
- Inference data (from user-uploaded GPX files)
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

from app.feature_engineering.point_extractor import Point, PointExtractor

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Handles feature extraction from GPS data for both training and inference.

    This class provides unified feature extraction that works for:
    - Training: Extract features from Strava API streams
    - Inference: Extract features from user-uploaded GPX files

    Both flows use the same core feature calculation logic to ensure consistency.

    All functionality is encapsulated within this class including:
    - Distance calculations (Haversine formula)
    - Slope computations over distance segments
    - Feature extraction from GPS points
    """

    # Earth radius in meters (used by Haversine formula)
    EARTH_RADIUS_M = 6371000

    # Distance segments for slope analysis (in meters)
    DISTANCE_SEGMENTS = [20, 100]

    # Tolerance for continuous climb/descent detection (meters)
    CONTINUOUS_SEGMENT_TOLERANCE_M = 20

    def __init__(self):
        """Initialize the feature extractor with a point extractor."""
        self._point_extractor = PointExtractor()

    def extract_from_streams(
        self,
        streams: Dict,
        start_time: datetime
    ) -> Dict[str, float]:
        """
        Extract features from Strava API streams (for TRAINING).

        Args:
            streams: Strava streams dictionary with keys "latlng", "altitude", "time", "distance"
            start_time: Activity start datetime

        Returns:
            Dictionary of ML features
        """
        logger.debug("Extracting features from Strava streams")
        points = self._point_extractor.extract_from_streams(streams, start_time)

        # Extract distance stream if available (pre-calculated by Strava - much more accurate!)
        distance_stream = None
        if streams.get('distance'):
            distance_stream = streams['distance']['data'] if isinstance(streams['distance'], dict) else streams['distance']

        return self._extract_features_from_points(points, distance_stream=distance_stream)

    def extract_from_gpx(self, gpx_bytes: bytes) -> Dict[str, float]:
        """
        Extract features from a GPX file (for INFERENCE).

        Args:
            gpx_bytes: GPX file content as bytes

        Returns:
            Dictionary of ML features
        """
        logger.debug("Extracting features from GPX file")
        points = self._point_extractor.extract_from_gpx(gpx_bytes)
        return self._extract_features_from_points(points)

    @staticmethod
    def _compute_distance_haversine(lat1: float,lon1: float,
                                    lat2: float,lon2: float) -> float:
        """
        Compute distance between two GPS coordinates using Haversine formula.

        Args:
            lat1, lon1: First point coordinates (degrees)
            lat2, lon2: Second point coordinates (degrees)

        Returns:
            Distance in meters
        """
        R = FeatureExtractor.EARTH_RADIUS_M

        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)

        a = (sin(delta_lat / 2) ** 2 +
             cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    @staticmethod
    def _compute_distance_segment_slopes(
        df: pd.DataFrame,
        segment_m: float
    ) -> pd.Series:
        """
        Compute gradients over fixed distance segments.

        Args:
            df: DataFrame with columns: distance_m, elevation
            segment_m: Minimum segment length in meters

        Returns:
            Series of slope values (rise/run)
        """
        slopes = []
        acc_dist = 0.0
        start_elevation = df.iloc[0]["elevation"]

        for i in range(1, len(df)):
            acc_dist += df.iloc[i]["distance_m"]
            if acc_dist >= segment_m:
                elevation_diff = df.iloc[i]["elevation"] - start_elevation
                slopes.append(elevation_diff / acc_dist)

                # Reset segment
                acc_dist = 0.0
                start_elevation = df.iloc[i]["elevation"]

        return pd.Series(slopes)

    @staticmethod
    def _compute_max_continuous_segments(
        df: pd.DataFrame,
        tolerance_m: float = 20
    ) -> Dict[str, float]:
        """
        Maximum continuous climb and descent segments.

        A climb segment breaks when elevation drops more than tolerance_m
        below the highest point reached in the segment. Same logic inverted
        for descent.

        Args:
            df: DataFrame with columns: elevation_smooth, distance_m
            tolerance_m: Allowed distance before breaking segment (meters)

        Returns:
            Dict with max_continuous_climb_gradient and max_continuous_descent_gradient
        """
        elevations = df["elevation_smooth"].values
        distances = df["distance_m"].values

        # --- Climb segments ---
        best_climb_gain = 0.0
        best_climb_dist = 0.0

        seg_start_elev = elevations[0]
        seg_max_elev = elevations[0]
        seg_dist = 0.0
        seg_dist_at_max = 0.0

        for i in range(1, len(elevations)):
            seg_dist += distances[i]

            if elevations[i] > seg_max_elev:
                seg_max_elev = elevations[i]
                seg_dist_at_max = seg_dist

            if seg_max_elev - elevations[i] > tolerance_m:
                climb_gain = seg_max_elev - seg_start_elev
                if climb_gain > best_climb_gain and seg_dist_at_max > 0:
                    best_climb_gain = climb_gain
                    best_climb_dist = seg_dist_at_max

                seg_start_elev = elevations[i]
                seg_max_elev = elevations[i]
                seg_dist = 0.0
                seg_dist_at_max = 0.0

        # Check last segment
        climb_gain = seg_max_elev - seg_start_elev
        if climb_gain > best_climb_gain and seg_dist_at_max > 0:
            best_climb_gain = climb_gain
            best_climb_dist = seg_dist_at_max

        # --- Descent segments ---
        best_descent_loss = 0.0
        best_descent_dist = 0.0

        seg_start_elev = elevations[0]
        seg_min_elev = elevations[0]
        seg_dist = 0.0
        seg_dist_at_min = 0.0

        for i in range(1, len(elevations)):
            seg_dist += distances[i]

            if elevations[i] < seg_min_elev:
                seg_min_elev = elevations[i]
                seg_dist_at_min = seg_dist

            if elevations[i] - seg_min_elev > tolerance_m:
                descent_loss = seg_start_elev - seg_min_elev
                if descent_loss > best_descent_loss and seg_dist_at_min > 0:
                    best_descent_loss = descent_loss
                    best_descent_dist = seg_dist_at_min

                seg_start_elev = elevations[i]
                seg_min_elev = elevations[i]
                seg_dist = 0.0
                seg_dist_at_min = 0.0

        # Check last segment
        descent_loss = seg_start_elev - seg_min_elev
        if descent_loss > best_descent_loss and seg_dist_at_min > 0:
            best_descent_loss = descent_loss
            best_descent_dist = seg_dist_at_min

        return {
            "max_continuous_climb_gradient": round(
                best_climb_gain / best_climb_dist, 3
            ) if best_climb_dist > 0 else 0.0,
            "max_continuous_descent_gradient": round(
                best_descent_loss / best_descent_dist, 3
            ) if best_descent_dist > 0 else 0.0,
        }

    def _extract_features_from_points(self, points: List[Point], distance_stream: List[float] = None) -> Dict[str, float]:
        """
        CORE feature extraction logic from GPS points.

        This is the unified implementation used by both training and inference.

        Args:
            points: List of Point objects with latitude, longitude, elevation, time
            distance_stream: Optional list of cumulative distances from Strava (meters)

        Returns:
            Dictionary of features ready for ML model

        Raises:
            ValueError: If insufficient points or invalid data
        """
        if len(points) < 2:
            raise ValueError(f"Need at least 2 points, got {len(points)}")

        # Convert to DataFrame
        records = []
        for i, point in enumerate(points):
            records.append({
                "latitude": point.latitude,
                "longitude": point.longitude,
                "elevation": point.elevation,
                "time": point.time,
                "point_index": i
            })

        df = pd.DataFrame(records)

        # Use Strava's pre-calculated distance if available (MUCH more accurate!)
        if distance_stream is not None and len(distance_stream) == len(df):
            # Strava provides cumulative distances, we need segment distances
            cumulative_distances = distance_stream
            distances = [0.0]  # First point has distance 0
            for i in range(1, len(cumulative_distances)):
                segment_dist = cumulative_distances[i] - cumulative_distances[i - 1]
                distances.append(segment_dist)
            df["distance_m"] = distances
            df["distance_cum"] = cumulative_distances
        else:
            # Fallback: compute distances between consecutive points using Haversine (less accurate)
            distances = [np.nan]  # First point has no previous point
            for i in range(1, len(df)):
                dist = self._compute_distance_haversine(
                    df.iloc[i - 1]["latitude"],
                    df.iloc[i - 1]["longitude"],
                    df.iloc[i]["latitude"],
                    df.iloc[i]["longitude"]
                )
                distances.append(dist)

            df["distance_m"] = distances

            # Remove points with invalid distances
            df = df.dropna(subset=["distance_m"])

            if len(df) < 2:
                raise ValueError("After filtering, insufficient valid points remain")

            # Calculate cumulative distance
            df["distance_cum"] = df["distance_m"].cumsum()

        # Apply smoothing to elevation to reduce GPS noise
        df["elevation_smooth"] = df["elevation"].rolling(window=5, center=True, min_periods=1).mean()

        # Calculate gradient over 30m windows using interpolation
        # elev_prev is the elevation at 30m behind
        df["elev_prev"] = np.interp(
            df["distance_cum"] - 30,
            df["distance_cum"],
            df["elevation_smooth"]
        )
        df["grade_30m"] = (df["elevation_smooth"] - df["elev_prev"]) / 30

        # Separate positive (uphill) and negative (downhill) gradients
        df["positive_grade"] = np.clip(df["grade_30m"], 0, None)
        df["negative_grade"] = np.clip(df["grade_30m"], None, 0)

        # Elevation differences (using 30m gradient converted back to elevation diff)
        df["elevation_diff"] = df["grade_30m"] * df["distance_m"]

        # Classification
        uphill = df["elevation_diff"] > 0
        downhill = df["elevation_diff"] < 0
    
        steep_uphill = uphill & (df["elevation_diff"] / df["distance_m"] > 0.04)
        steep_downhill = downhill & (df["elevation_diff"] / df["distance_m"] < -0.04)

        # Distance metrics
        total_distance_m = df["distance_m"].sum()
        uphill_distance_m = df.loc[steep_uphill, "distance_m"].sum()
        downhill_distance_m = df.loc[steep_downhill, "distance_m"].sum()
        flat_distance_m = total_distance_m - uphill_distance_m - downhill_distance_m

        # Elevation metrics
        elevation_gain_m = df.loc[uphill, "elevation_diff"].sum()
        elevation_loss_m = -df.loc[downhill, "elevation_diff"].sum()
        average_elevation = df["elevation_smooth"].mean()
        min_elevation_m = df["elevation_smooth"].min()
        max_elevation_m = df["elevation_smooth"].max()
        average_uphill_slope = elevation_gain_m / total_distance_m
        average_downhill_slope = elevation_loss_m / total_distance_m
        point_slope = df["elevation_diff"] / df["distance_m"]
        steep_20_distance = df.loc[point_slope > 0.2, "distance_m"].sum()
        steep_ratio_20 = steep_20_distance / total_distance_m

        # Gradient metrics (from 30m window gradients)
        mean_positive_grade = df["positive_grade"].mean()
        max_positive_grade = df["positive_grade"].max()
        mean_negative_grade = df["negative_grade"].mean()
        max_negative_grade = df["negative_grade"].max()

        # Uphill cost: sum of (distance * gradient²) - penalizes steep sections more
        uphill_cost = (df["distance_m"] * df["positive_grade"] ** 3).sum()

        # Time metrics
        timestamps = df["time"].tolist()
        start_time = min(timestamps)
        end_time = max(timestamps)
        total_time_sec = (end_time - start_time).total_seconds()

        if total_time_sec <= 0:
            raise ValueError("Invalid time span (start >= end)")

        if total_distance_m == 0:
            raise ValueError("Total distance is zero")
        
        total_distance_km = total_distance_m / 1000

        # Build feature dictionary
        features = {
            #"total_distance_km": round(total_distance_m / 1000, 3),
            #"uphill_distance_km": round(uphill_distance_m / 1000, 3),
            #"downhill_distance_km": round(downhill_distance_m / 1000, 3),
            #"flat_distance_km": round(flat_distance_m / 1000, 3),
            "elevation_gain_m": round(elevation_gain_m, 1),
            "km_effort": round(total_distance_km + elevation_gain_m / 90, 3),
            #"vertical_rate": round(elevation_gain_m / total_distance_km, 3),
            "elevation_loss_m": round(elevation_loss_m, 1),
            "average_elevation": round(average_elevation, 1),
            #"min_elevation_m": round(min_elevation_m, 1),
            #"max_elevation_m": round(max_elevation_m, 1),
            #"average_uphill_slope": round(average_uphill_slope, 1),
            #"average_downhill_slope": round(average_downhill_slope, 1),
            #"steep_ratio_20": round(steep_ratio_20, 3),
            #"num_points": len(df),
            #"uphill_ratio": round(uphill_distance_m / total_distance_m, 3) if total_distance_m > 0 else 0,
            "downhill_ratio": round(downhill_distance_m / total_distance_m, 3) if total_distance_m > 0 else 0,
            "flat_ratio": round(flat_distance_m / total_distance_m, 3) if total_distance_m > 0 else 0,
            "mean_positive_grade": round(mean_positive_grade, 4),
            #"max_positive_grade": round(max_positive_grade, 4),
            #"mean_negative_grade": round(mean_negative_grade, 4),
            "uphill_cost": round(uphill_cost * elevation_gain_m, 2),
            "total_time_sec": round(total_time_sec, 1)
        }

        # Compute max continuous climb/descent gradients
        continuous = self._compute_max_continuous_segments(
            df, tolerance_m=self.CONTINUOUS_SEGMENT_TOLERANCE_M
        )
        features.update(continuous)

        # Compute slopes over fixed distance segments
        #for d in self.DISTANCE_SEGMENTS:
        slopes = self._compute_distance_segment_slopes(df, segment_m=100)
        uphill_slopes = slopes[slopes > 0]
        downhill_slopes = -slopes[slopes < 0]

        features.update({
               # f"average_uphill_slope_100": round(uphill_slopes.mean(), 3) if not uphill_slopes.empty else 0,
               # f"max_uphill_slope{d}m": round(uphill_slopes.max(), 3) if not uphill_slopes.empty else np.nan,
               # f"std_uphill_gradient{d}m": round(uphill_slopes.std(ddof=0), 3) if len(uphill_slopes) > 1 else np.nan,

               #f"average_downhill_slope_100": round(downhill_slopes.mean(), 3) if not downhill_slopes.empty else 0,
               # f"max_downhill_slope{d}m": round(downhill_slopes.max(), 3) if not downhill_slopes.empty else np.nan,
               # f"std_downhill_slope{d}m": round(downhill_slopes.std(ddof=0), 3) if len(downhill_slopes) > 1 else np.nan,

               # f"steep_ratio_10_uphill_{d}m": round((uphill_slopes > 0.1).mean(), 3) if not uphill_slopes.empty else np.nan,
               # f"steep_ratio_20_uphill_{d}m": round((uphill_slopes > 0.2).mean(), 3) if not uphill_slopes.empty else np.nan
            })

        return features
