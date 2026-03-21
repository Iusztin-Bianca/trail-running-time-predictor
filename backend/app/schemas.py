"""
Pydantic models for API request/response validation and feature validation.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ExtractedFeatures(BaseModel):
    """
    Validated schema for features extracted from GPS data.

    This schema serves two purposes:
    1. TRAINING: All features + target variable (total_time_sec)
    2. INFERENCE: All features as INPUT to ML model (total_time_sec extracted from GPX for reference)

    Note: Slope metrics are Optional because they can be None/NaN when:
    - Trail is completely flat (no uphill/downhill segments)
    - Only downhill or only uphill (missing one direction)
    """
    # Distance metrics (km)
    total_distance_km: float = Field(gt=0, description="Total route distance in kilometers")
    uphill_distance_km: float = Field(ge=0, description="Distance of steep uphill segments")
    downhill_distance_km: float = Field(ge=0, description="Distance of steep downhill segments")
    flat_distance_km: float = Field(ge=0, description="Distance of flat segments")

    # Elevation metrics (meters)
    elevation_gain_m: float = Field(ge=0, description="Total elevation gain in meters")
    elevation_loss_m: float = Field(ge=0, description="Total elevation loss in meters")
    average_elevation: float = Field(description="Average elevation in meters")
    min_elevation_m: float = Field(description="Minimum elevation in meters")
    max_elevation_m: float = Field(description="Maximum elevation in meters")

    # Ratios
    uphill_ratio: float = Field(ge=0, le=1, description="Ratio of uphill distance to total")
    downhill_ratio: float = Field(ge=0, le=1, description="Ratio of downhill distance to total")

    # Time metrics
    total_time_sec: float = Field(gt=0, description="Total time in seconds")

    # Data quality metrics
    num_points: int = Field(gt=0, description="Number of GPS points")
    missing_elevation_data: float = Field(ge=0, le=1, description="Ratio of missing elevation data")
    num_segments_lt_20m: int = Field(ge=0, description="Number of segments shorter than 20m")

    # Slope metrics for 20m segments
    average_uphill_slope20m: Optional[float] = Field(default=None, description="Average uphill slope over 20m segments")
    max_uphill_slope20m: Optional[float] = Field(default=None, description="Max uphill slope over 20m segments")
    std_uphill_gradient20m: Optional[float] = Field(default=None, description="Std dev of uphill slopes over 20m")
    average_downhill_slope20m: Optional[float] = Field(default=None, description="Average downhill slope over 20m")
    max_downhill_slope20m: Optional[float] = Field(default=None, description="Max downhill slope over 20m")
    std_downhill_slope20m: Optional[float] = Field(default=None, description="Std dev of downhill slopes over 20m")
    steep_ratio_10_uphill_20m: Optional[float] = Field(default=None, ge=0, le=1, description="Ratio of 20m segments with >10% uphill grade")
    steep_ratio_20_uphill_20m: Optional[float] = Field(default=None, ge=0, le=1, description="Ratio of 20m segments with >20% uphill grade")

    # Slope metrics for 50m segments
    average_uphill_slope50m: Optional[float] = Field(default=None, description="Average uphill slope over 50m segments")
    max_uphill_slope50m: Optional[float] = Field(default=None, description="Max uphill slope over 50m segments")
    std_uphill_gradient50m: Optional[float] = Field(default=None, description="Std dev of uphill slopes over 50m")
    average_downhill_slope50m: Optional[float] = Field(default=None, description="Average downhill slope over 50m")
    max_downhill_slope50m: Optional[float] = Field(default=None, description="Max downhill slope over 50m")
    std_downhill_slope50m: Optional[float] = Field(default=None, description="Std dev of downhill slopes over 50m")
    steep_ratio_10_uphill_50m: Optional[float] = Field(default=None, ge=0, le=1, description="Ratio of 50m segments with >10% uphill grade")
    steep_ratio_20_uphill_50m: Optional[float] = Field(default=None, ge=0, le=1, description="Ratio of 50m segments with >20% uphill grade")

    # Slope metrics for 100m segments
    average_uphill_slope100m: Optional[float] = Field(default=None, description="Average uphill slope over 100m segments")
    max_uphill_slope100m: Optional[float] = Field(default=None, description="Max uphill slope over 100m segments")
    std_uphill_gradient100m: Optional[float] = Field(default=None, description="Std dev of uphill slopes over 100m")
    average_downhill_slope100m: Optional[float] = Field(default=None, description="Average downhill slope over 100m")
    max_downhill_slope100m: Optional[float] = Field(default=None, description="Max downhill slope over 100m")
    std_downhill_slope100m: Optional[float] = Field(default=None, description="Std dev of downhill slopes over 100m")
    steep_ratio_10_uphill_100m: Optional[float] = Field(default=None, ge=0, le=1, description="Ratio of 100m segments with >10% uphill grade")
    steep_ratio_20_uphill_100m: Optional[float] = Field(default=None, ge=0, le=1, description="Ratio of 100m segments with >20% uphill grade")

    @field_validator('total_distance_km')
    @classmethod
    def validate_distance(cls, v: float) -> float:
        """Ensure distance is reasonable (< 500km for trail running)."""
        if v > 500:
            raise ValueError("Distance exceeds maximum expected value of 500km")
        return v

    @field_validator('total_time_sec')
    @classmethod
    def validate_time(cls, v: float) -> float:
        """Ensure time is reasonable (< 7 days)."""
        if v > 7 * 24 * 3600:  # 7 days in seconds
            raise ValueError("Time exceeds maximum expected value of 7 days")
        return v

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "total_distance_km": 11.259,
                "uphill_distance_km": 4.98,
                "downhill_distance_km": 5.333,
                "flat_distance_km": 0.947,
                "elevation_gain_m": 783.0,
                "elevation_loss_m": 782.8,
                "average_elevation": 268.9,
                "min_elevation_m": 117.2,
                "max_elevation_m": 456.0,
                "num_points": 2198,
                "uphill_ratio": 0.442,
                "downhill_ratio": 0.474,
                "total_time_sec": 7225.0,
                "missing_elevation_data": 0.0,
                "num_segments_lt_20m": 2176
            }
        }


class PredictionResponse(BaseModel):
    """
    Response schema for ML model predictions.

    The ML model predicts total_time_sec (in seconds) internally,
    but we convert it to minutes for better UX.
    """
    predicted_time_minutes: float = Field(gt=0, description="Predicted completion time in minutes (converted from total_time_sec)")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "predicted_time_minutes": 125.5
            }
        }