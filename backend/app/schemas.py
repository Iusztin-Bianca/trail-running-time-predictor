"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response schema for ML model predictions."""
    predicted_time_minutes: float = Field(gt=0, description="Predicted completion time in minutes")
    predicted_time_formatted: str = Field(description="Predicted time in human-readable format (e.g. '2h 05m 30s')")
    num_segments: int = Field(gt=0, description="Number of terrain segments extracted from the GPX")

    @classmethod
    def from_seconds(cls, total_seconds: float, num_segments: int) -> "PredictionResponse":
        total_sec = int(total_seconds)
        hours = total_sec // 3600
        minutes = (total_sec % 3600) // 60
        seconds = total_sec % 60
        if hours > 0:
            formatted = f"{hours}h {minutes:02d}m {seconds:02d}s"
        else:
            formatted = f"{minutes}m {seconds:02d}s"
        return cls(
            predicted_time_minutes=round(total_seconds / 60, 1),
            predicted_time_formatted=formatted,
            num_segments=num_segments,
        )