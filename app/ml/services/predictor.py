"""Inference service: GPX bytes → predicted race time."""

import logging

import numpy as np
import pandas as pd

from app.constants import DEFAULT_EXCLUDE_COLUMNS
from app.feature_engineering.point_extractor import PointExtractor
from app.feature_engineering.segment_features import SegmentFeatureExtractor

logger = logging.getLogger(__name__)


class PredictorService:
    """Runs end-to-end inference from a GPX file.

    Flow:
        GPX bytes
        → PointExtractor      (parse GPS points)
        → SegmentFeatureExtractor  (segment-level features, same as training)
        → model.predict()     (segment times in seconds)
        → sum                 (total predicted time)
    """

    def __init__(self, model):
        self.model = model
        self._point_extractor = PointExtractor()
        self._segment_extractor = SegmentFeatureExtractor()

    def predict_from_gpx(self, gpx_bytes: bytes, is_race: int = 0, is_easy: int = 0) -> dict:
        """Predict total race time from a GPX file.

        Args:
            gpx_bytes: Raw bytes of the uploaded .gpx file.
            is_race:   1 if the user is predicting a race effort, 0 otherwise.
            is_easy:   1 if the user is predicting a recovery run, 0 otherwise.

        Returns:
            Dict with keys: total_seconds (float), num_segments (int).

        Raises:
            ValueError: If the GPX file has too few points or yields no segments.
        """
        points = self._point_extractor.extract_from_gpx(gpx_bytes)
        if len(points) < 2:
            raise ValueError("GPX file has too few GPS points for segmentation.")

        segments = self._segment_extractor.extract_features(points, is_race=is_race, is_easy=is_easy)
        if not segments:
            raise ValueError("No segments could be extracted from the GPX file.")

        df = pd.DataFrame(segments)
        feature_cols = [c for c in df.columns if c not in DEFAULT_EXCLUDE_COLUMNS]
        X = df[feature_cols].fillna(0).values

        y_pred: np.ndarray = self.model.predict(X)
        total_seconds = float(y_pred.sum())

        logger.info(
            "Prediction complete — %d segments, total predicted time: %.0fs (%.1f min)",
            len(segments),
            total_seconds,
            total_seconds / 60,
        )

        return {"total_seconds": total_seconds, "num_segments": len(segments)}
