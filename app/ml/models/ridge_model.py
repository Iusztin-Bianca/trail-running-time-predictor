"""
Ridge Regression model (baseline).

Simple linear model with L2 regularization.
Good baseline for comparison with more complex models.
"""
from __future__ import annotations

import logging
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from app.ml.config.ridge_params import RidgeRegressionParams
from app.ml.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class RidgeModel(BaseModel):
    """
    Ridge Regression model with feature scaling.
    This is the baseline model : simple, interpretable, and works
    well with small datasets due to L2 regularization.
    """

    def __init__(self, hyperparameters: RidgeRegressionParams):
        
        super().__init__(name="ridge", hyperparameters=hyperparameters)

        self.scaler = StandardScaler()
        self.model = Ridge(**vars(hyperparameters))

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> RidgeModel:
        """
        Train the Ridge model.
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        logger.info(f"Training Ridge model with {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Hyperparameters: alpha={self.hyperparameters.alpha}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        self.is_fitted = True

        logger.info(f"Ridge model trained. Intercept: {self.model.intercept_:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the Ridge model.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted values (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

