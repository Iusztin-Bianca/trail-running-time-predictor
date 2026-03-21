"""SHAP-based feature importance analysis."""

import logging

import numpy as np
import pandas as pd
import shap

from app.ml.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Computes SHAP values for trained models.

    Automatically selects the right SHAP explainer based on model type:
      - XGBoost → TreeExplainer (exact, fast)
      - Ridge   → LinearExplainer (exact for linear models)

    Usage:
        analyzer = SHAPAnalyzer()
        result = analyzer.analyze(model, X_train_df)
    """

    def analyze(self, model: BaseModel, X: pd.DataFrame) -> dict[str, float]:
        """Compute mean |SHAP value| per feature.

        Args:
            model: A fitted BaseModel instance (ridge or xgboost).
            X: Training data as DataFrame (with feature names as columns).

        Returns:
            Dict mapping feature name → mean |SHAP value| (in target units, i.e. seconds),
            sorted descending by importance.
        """
        feature_names = list(X.columns)

        if model.name == "xgboost":
            shap_values = self._explain_xgboost(model, X)
        elif model.name == "ridge":
            shap_values = self._explain_ridge(model, X)
        else:
            logger.warning("No SHAP explainer for model type: %s", model.name)
            return {}

        mean_abs = np.abs(shap_values).mean(axis=0)

        importance = {
            name: float(val)
            for name, val in zip(feature_names, mean_abs)
        }
        # Sort descending
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        logger.info(
            "SHAP analysis for %s — top 10: %s",
            model.name,
            ", ".join(f"{k}: {v:.1f}s" for k, v in list(importance.items())[:3]),
        )

        return importance

    def _explain_xgboost(self, model: BaseModel, X: pd.DataFrame) -> np.ndarray:
        """Use TreeExplainer for XGBoost models."""
        explainer = shap.TreeExplainer(model.model)
        shap_result = explainer(X)
        return shap_result.values

    def _explain_ridge(self, model: BaseModel, X: pd.DataFrame) -> np.ndarray:
        """Use LinearExplainer for Ridge models (needs scaled data)."""
        X_scaled = model.scaler.transform(X.values)
        explainer = shap.LinearExplainer(model.model, X_scaled)
        shap_result = explainer(X_scaled)
        return shap_result.values