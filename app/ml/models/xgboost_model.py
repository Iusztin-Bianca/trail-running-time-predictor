"""
XGBoost model for trail running time prediction.

Gradient boosting model - more complex than Ridge baseline.
Configured with conservative hyperparameters for small datasets.
"""
from __future__ import annotations

import logging
import numpy as np
import xgboost as xgb

from app.ml.config.xgboost_params import XGBoostParams
from app.ml.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class XGBoostModel(BaseModel):
    """
    XGBoost Regression model.
    Configured with conservative hyperparameters to prevent overfitting
    on small datasets.
    """

    def __init__(self, hyperparameters: XGBoostParams):
        super().__init__(name="xgboost", hyperparameters=hyperparameters)
        params = vars(hyperparameters).copy()
        self.early_stopping_rounds = params.pop("early_stopping_rounds", 0)
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            early_stopping_rounds=self.early_stopping_rounds or None,
            **params,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostModel:
        """
        Train the XGBoost model.

        If early_stopping_rounds > 0, splits training data 80/20 internally
        and stops when validation MAE stops improving.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        logger.info(f"Training XGBoost model with {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Key hyperparameters: max_depth={self.hyperparameters.max_depth}, "
                   f"n_estimators={self.hyperparameters.n_estimators}, "
                   f"learning_rate={self.hyperparameters.learning_rate}")

        if self.early_stopping_rounds > 0 and X.shape[0] >= 10:
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            best_iter = getattr(self.model, "best_iteration", self.hyperparameters.n_estimators)
            logger.info(f"Early stopping: used {best_iter} / {self.hyperparameters.n_estimators} trees")
        else:
            self.model.fit(X, y)

        self.is_fitted = True
        logger.info("XGBoost model trained successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the XGBoost model.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted values (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict(X)

