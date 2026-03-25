"""
Segment-based feature extraction for trail running prediction.

Instead of treating each run as a single observation, this module breaks each run
into segments based on gradient changes. Each segment becomes a separate observation,
dramatically increasing the dataset size.

Segment definition:
- A segment has a consistent terrain type (uphill, downhill, or flat)
- Segments are created when the terrain type changes
- Gradient is computed over 30m windows

Terrain classification:
- Uphill: gradient > 3%
- Downhill: gradient < -3%
- Flat: gradient between -3% and 3%
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from app.feature_engineering.point_extractor import PointExtractor, Point

logger = logging.getLogger(__name__)


class SegmentFeatureExtractor:
    """Extract features for segments within a trail run.

    Each run is broken into segments based on gradient changes, and each segment
    becomes a separate training observation with its own features and time.
    """

    # Gradient thresholds for terrain classification (hysteresis)
    UPHILL_THRESHOLD = 0.03    # enter uphill: gradient > +3%
    DOWNHILL_THRESHOLD = -0.03  # enter downhill: gradient < -3%
    SEGMENT_EXIT = 0.0           # exit uphill/downhill to flat: gradient < 0%

    # Window size for gradient calculation
    GRADIENT_WINDOW_M = 30.0

    # Minimum segment length / time to avoid noise
    MIN_SEGMENT_DISTANCE_M = 50.0   # At least 50m per segment
    MIN_SEGMENT_TIME_SEC = 30.0     # At least 30s per segment

    # Maximum segment length — longer segments are split into equal sub-segments
    MAX_SEGMENT_DISTANCE_M = 1000.0

    def __init__(self):
        """Initialize the segment feature extractor."""
        pass

    @staticmethod
    def _compute_distance_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute distance between two GPS coordinates using Haversine formula."""
        from math import radians, sin, cos, sqrt, atan2

        R = 6371000  # Earth radius in meters

        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)

        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def _build_dataframe_from_points(
        self,
        points: List[Point],
        start_time: datetime = None,
        distance_stream: List = None
    ) -> pd.DataFrame:
        """Build DataFrame with cumulative distance from Point objects.

        Args:
            points: List of Point objects with latitude, longitude, elevation, time
            start_time: Activity start datetime (used to compute time_sec offsets)
            distance_stream: Optional list of cumulative distances from Strava (meters)

        Returns:
            DataFrame with columns: distance_m, altitude_m, time_sec
        """
        data = []

        has_time = start_time is not None

        # Use Strava's pre-calculated distance if available (MUCH more accurate!)
        if distance_stream is not None and len(distance_stream) == len(points):
            for i, point in enumerate(points):
                data.append({
                    'distance_m': distance_stream[i],
                    'altitude_m': point.elevation,
                    'time_sec': (point.time - start_time).total_seconds() if has_time else 0.0,
                })
        else:
            # Calculate distance with Haversine
            cumulative_distance = 0.0
            for i, point in enumerate(points):
                if i > 0:
                    prev = points[i - 1]
                    cumulative_distance += self._compute_distance_haversine(
                        prev.latitude, prev.longitude, point.latitude, point.longitude
                    )
                data.append({
                    'distance_m': cumulative_distance,
                    'altitude_m': point.elevation,
                    'time_sec': (point.time - start_time).total_seconds() if has_time else 0.0,
                })

        return pd.DataFrame(data)

    def _compute_gradients(self, df: pd.DataFrame) -> np.ndarray:
        """Compute gradient over 30m windows( each point will have a gradient 
        computed on a window of 30m before it)

        Args:
            df: DataFrame with all the points(distance_m, altitude_m)

        Returns:
            Array of gradients for each point 
        """
        gradients = []

        for i in range(len(df)):
            # Find points within 30m window ahead
            current_dist = df.iloc[i]['distance_m']
            window_mask = (df['distance_m'] >= current_dist) & (df['distance_m'] <= current_dist + self.GRADIENT_WINDOW_M)

            window_df = df[window_mask]

            if len(window_df) < 2:
                # Not enough points in window, use previous gradient or 0
                gradients.append(gradients[-1] if gradients else 0.0)
                continue

            # Calculate gradient over window
            elev_change = window_df.iloc[-1]['altitude_m'] - window_df.iloc[0]['altitude_m']
            dist_change = window_df.iloc[-1]['distance_m'] - window_df.iloc[0]['distance_m']

            if dist_change > 0:
                gradient = elev_change / dist_change
            else:
                gradient = 0.0

            gradients.append(gradient)

        return np.array(gradients)

    def _create_segments(self, df: pd.DataFrame, gradients: np.ndarray) -> List[Tuple[int, int, str]]:
        """Create segments based on terrain type changes.

        Uses different entry/exit thresholds:
        - Enter uphill:   gradient > UPHILL_THRESHOLD (+3% = 0.03)
        - Exit uphill:    gradient < SEGMENT_EXIT (0%)
        - Enter downhill: gradient < DOWNHILL_THRESHOLD (-3%)
        - Exit downhill:  gradient > SEGMENT_EXIT (0%)

        Args:
            df: DataFrame with activity data
            gradients: Array of gradients for each point

        Returns:
            List of (start_idx, end_idx, terrain_type) tuples
        """
        segments = []

        if len(df) == 0:
            return segments

        # Determine initial terrain type
        g0 = gradients[0]
        if g0 > self.UPHILL_THRESHOLD:
            current_type = 'uphill'
        elif g0 < self.DOWNHILL_THRESHOLD:
            current_type = 'downhill'
        else:
            current_type = 'flat'

        start_idx = 0

        for i in range(1, len(gradients)):
            g = gradients[i]

            if current_type == 'flat':
                if g > self.UPHILL_THRESHOLD:
                    new_type = 'uphill'
                elif g < self.DOWNHILL_THRESHOLD:
                    new_type = 'downhill'
                else:
                    new_type = 'flat'
            elif current_type == 'uphill':
                if g < self.DOWNHILL_THRESHOLD:
                    new_type = 'downhill'
                elif g < self.SEGMENT_EXIT:
                    new_type = 'flat'
                else:
                    new_type = 'uphill'
            else:  # downhill
                if g > self.UPHILL_THRESHOLD:
                    new_type = 'uphill'
                elif g > self.SEGMENT_EXIT:
                    new_type = 'flat'
                else:
                    new_type = 'downhill'

            if new_type != current_type:
                segments.append((start_idx, i - 1, current_type))
                start_idx = i
                current_type = new_type

        # Close final segment
        segments.append((start_idx, len(gradients) - 1, current_type))

        return segments

    def _merge_short_segments(
        self,
        segments: List[Tuple[int, int, str]],
        df: pd.DataFrame
    ) -> List[Tuple[int, int, str]]:
        """Merge segments shorter than minimum distance and time(<50m, <30s)
          with adjacent segments.
        """
        if not segments:
            return segments

        filtered = []
        i = 0
        has_time_data = df['time_sec'].max() > 0

        while i < len(segments):
            start_idx, end_idx, terrain_type = segments[i]
            segment_dist = df.iloc[end_idx]['distance_m'] - df.iloc[start_idx]['distance_m']
            segment_time = df.iloc[end_idx]['time_sec'] - df.iloc[start_idx]['time_sec']

            too_short = segment_dist < self.MIN_SEGMENT_DISTANCE_M
            too_fast = has_time_data and segment_time < self.MIN_SEGMENT_TIME_SEC
            if too_short or too_fast:
                if i < len(segments) - 1:
                    # Merge forward into next segment
                    next_start, next_end, next_type = segments[i + 1]
                    next_dist = df.iloc[next_end]['distance_m'] - df.iloc[next_start]['distance_m']
                    merged_type = next_type if next_dist > segment_dist else terrain_type
                    segments[i + 1] = (start_idx, next_end, merged_type)
                elif filtered:
                    # Last segment is short → merge backward into previous
                    prev_start, prev_end, prev_type = filtered[-1]
                    filtered[-1] = (prev_start, end_idx, prev_type)
                else:
                    # Only one segment and it's short — keep it
                    filtered.append((start_idx, end_idx, terrain_type))
            else:
                filtered.append((start_idx, end_idx, terrain_type))

            i += 1

        return filtered

    def _split_long_segments(
        self,
        segments: List[Tuple[int, int, str]],
        df: pd.DataFrame,
    ) -> List[Tuple[int, int, str]]:
        """Split segments longer than MAX_SEGMENT_DISTANCE_M(1000m) into equal sub-segments.
           A 2500m segment becomes 3 sub-segments of ~833m each, all with the same
           terrain_type. This reduces prediction error for long uniform segments.
        """
        result = []

        for start_idx, end_idx, terrain_type in segments:
            seg_dist = df.iloc[end_idx]['distance_m'] - df.iloc[start_idx]['distance_m']

            if seg_dist <= self.MAX_SEGMENT_DISTANCE_M:
                result.append((start_idx, end_idx, terrain_type))
                continue

            # How many equal sub-segments to create
            n = int(np.ceil(seg_dist / self.MAX_SEGMENT_DISTANCE_M))
            start_dist = df.iloc[start_idx]['distance_m']
            distances = df['distance_m'].values

            sub_start = start_idx
            for k in range(1, n):
                target_dist = start_dist + k * (seg_dist / n)
                # Find positional index closest to target_dist within this segment
                rel_idx = int(np.argmin(np.abs(distances[sub_start:end_idx + 1] - target_dist)))
                split_idx = sub_start + rel_idx
                result.append((sub_start, split_idx, terrain_type))
                sub_start = split_idx

            result.append((sub_start, end_idx, terrain_type))

        return result

    def _extract_segment_features(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        terrain_type: str,
        is_race: int,
        is_easy: int,
        gradients: np.ndarray = None,
    ) -> Dict[str, float]:
        """Extract features for a single segment.

        Args:
            df: DataFrame with activity data
            start_idx: Start index of segment
            end_idx: End index of segment
            terrain_type: 'uphill', 'downhill', or 'flat'
            is_race: 1 if parent activity is a race, 0 otherwise
            is_easy: 1 if parent activity is a recovery run, 0 otherwise

        Returns:
            Dictionary of features
        """
        segment_df = df.iloc[start_idx:end_idx + 1]

        # Basic segment metrics
        segment_distance_m = segment_df.iloc[-1]['distance_m'] - segment_df.iloc[0]['distance_m']
        segment_time_sec = segment_df.iloc[-1]['time_sec'] - segment_df.iloc[0]['time_sec']

        # Elevation change
        elevation_change = segment_df.iloc[-1]['altitude_m'] - segment_df.iloc[0]['altitude_m']
        elevation_gain_m = max(0, elevation_change)
        elevation_loss_m = max(0, -elevation_change)

        # Gradient
        if segment_distance_m > 0:
            avg_gradient = elevation_change / segment_distance_m
        else:
            avg_gradient = 0.0

        # Average elevation
        avg_elevation = segment_df['altitude_m'].mean()

        # Max gradient and std gradient consistent with terrain_type:
        # - uphill segment:   max_uphill_gradient = steepest point,  max_downhill_gradient = 0
        # - downhill segment: max_uphill_gradient = 0, max_downhill_gradient = steepest point
        # - flat segment:     both = 0
        # This avoids noise from point-level gradient oscillations.
        if gradients is not None and end_idx > start_idx:
            seg_gradients = gradients[start_idx:end_idx + 1]
            std_gradient = round(float(np.std(seg_gradients)), 4)
            if terrain_type == 'uphill':
                max_uphill_gradient = round(float(max(0.0, np.max(seg_gradients))), 4)
                max_downhill_gradient = 0.0
            elif terrain_type == 'downhill':
                max_uphill_gradient = 0.0
                max_downhill_gradient = round(float(abs(min(0.0, np.min(seg_gradients)))), 4)
            else:  # flat
                max_uphill_gradient = 0.0
                max_downhill_gradient = 0.0
        else:
            std_gradient = 0.0
            if terrain_type == 'uphill':
                max_uphill_gradient = round(max(0.0, avg_gradient), 4)
                max_downhill_gradient = 0.0
            elif terrain_type == 'downhill':
                max_uphill_gradient = 0.0
                max_downhill_gradient = round(abs(min(0.0, avg_gradient)), 4)
            else:
                max_uphill_gradient = 0.0
                max_downhill_gradient = 0.0

        uphill_cost = segment_distance_m * (1 + 6 * avg_gradient) if elevation_change > 0 else 0.0
        downhill_cost = segment_distance_m * (1 + 6 * abs(avg_gradient)) if elevation_change < 0 else 0.0

        # Minetti biomechanical energy cost (J/kg) for the segment
        # g is signed gradient: positive = uphill, negative = downhill
        # Formula: Minetti, valid for g in [-0.45, +0.45]
        g = avg_gradient  # signed (positive uphill, negative downhill)
        energy_cost_per_m = 155.4*g**5 - 30.4*g**4 - 43.3*g**3 + 46.3*g**2 + 19.5*g + 3.6
        segment_energy_cost = round(energy_cost_per_m * segment_distance_m, 3)

        # Cumulative elevation gain from activity start to end of this segment
        alt_diffs = df.iloc[0:end_idx + 1]['altitude_m'].diff().dropna()
        cumulative_elevation = round(float(alt_diffs[alt_diffs > 0].sum()), 1)

        segment_pace_mps = round(segment_distance_m / segment_time_sec, 4) if segment_time_sec > 0 else 1.0

        is_steep_uphill = 1 if terrain_type == 'uphill' and abs(avg_gradient) >= 0.3 else 0
        is_steep_downhill = 1 if terrain_type == 'downhill' and abs(avg_gradient) >= 0.3 else 0

        return {
            'segment_distance_m': round(segment_distance_m, 3),
            'segment_time_sec': round(segment_time_sec, 3),
            'segment_pace_mps': round(segment_pace_mps, 3),
            'elevation_gain_m': round(elevation_gain_m, 3),
            'elevation_loss_m': round(elevation_loss_m, 3),
            'avg_gradient': round(abs(avg_gradient), 3),
            'std_gradient': round(std_gradient, 4),
            'max_uphill_gradient': round(max_uphill_gradient, 3),
            'max_downhill_gradient': round(max_downhill_gradient, 3),
            'avg_elevation': round(avg_elevation, 1),
            'is_race': is_race,
            'is_easy': is_easy,
            'uphill_cost': round(uphill_cost, 3),
            'downhill_cost': round(downhill_cost, 3),
            'cumulative_elevation': round(cumulative_elevation, 3),
            'segment_energy_cost': segment_energy_cost,
            'is_steep_uphill': is_steep_uphill,
            'is_steep_downhill': is_steep_downhill,
        }


    def extract_features(
        self,
        points: List[Point],
        is_race: int,
        is_easy: int,
        distance_stream: List[float] = None,
    ) -> List[Dict[str, float]]:
        """Build segment features from a list of GPS points(comming from strava streams/ gpx file)
           Shared by both Strava training (pass distance_stream for accuracy)
           and GPX inference (distance_stream=None → Haversine).
        """
        if len(points) < 2:
            logger.warning("Not enough points for segmentation (need at least 2).")
            return []

        start_time = points[0].time  # may be None for route GPX files without timestamps

        df = self._build_dataframe_from_points(points, start_time, distance_stream)

        if df.empty or len(df) < 2:
            logger.warning("Not enough data points for segmentation.")
            return []

        gradients = self._compute_gradients(df)
        segments = self._create_segments(df, gradients)
        segments = self._merge_short_segments(segments, df)
        segments = self._split_long_segments(segments, df)

        logger.info(f"Created {len(segments)} segments from {len(points)} points.")

        segment_features = []
        for start_idx, end_idx, terrain_type in segments:
            features = self._extract_segment_features(
                df, start_idx, end_idx, terrain_type, is_race, is_easy,
                gradients=gradients,
            )

            # Drop GPS stop artifacts: short segments (<100m) with implausibly slow pace
            if features['segment_distance_m'] < 100.0 and features['segment_pace_mps'] <= 0.15:
                continue

            # Cap unrealistically long segment times >20min (GPS stoppages within a segment)
            # Try to eliminate the stationary times
            # *For steep or technic uphill/downhill (with gradient > 30%), we suppose that 
            # a person could run/power hike 1000m in maximum 25 de minutes
            # 1000m.......25min
            # seg_distance....seg_time => seg_time can be maximum seg_distance* 25/1000 =>
            # we limit the segment time to this value
            # *Also for a terrain that is not that steep/techinal => we suppose that a person 
            # can run that segment in maximum 20 minutes(this is the pesimistic case)
            if features['segment_time_sec'] > 1200:
                if features['avg_gradient'] >= 0.3:
                    features['segment_time_sec'] = round(features['segment_distance_m'] * 15 / 10, 3)
                else:
                    features['segment_time_sec'] = round(features['segment_distance_m'] * 12 / 10, 3)
                features['segment_pace_mps'] = round(
                    features['segment_distance_m'] / features['segment_time_sec'], 4
                )

            segment_features.append(features)

        return segment_features

