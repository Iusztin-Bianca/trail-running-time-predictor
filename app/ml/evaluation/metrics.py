"""Evaluation metrics for regression models."""

import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates regression evaluation metrics."""

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate regression metrics.

        Returns dict with:
            mae  — Mean Absolute Error (in seconds)
            rmse — Root Mean Squared Error (in seconds)
            mape — Mean Absolute Percentage Error (%)
            r2   — R² score (1.0 = perfect, 0.0 = predicts the mean)
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # MAPE — avoid division by zero
        nonzero = y_true != 0
        if nonzero.any():
            mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        else:
            mape = 0

        return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}
