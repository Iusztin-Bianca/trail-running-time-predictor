"""Hyperparameter tuning using sklearn GridSearchCV."""

import logging
from dataclasses import fields

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from app.constants import DEFAULT_CV_SPLITS
from app.ml.data.data_splitter import TemporalSplitter
from app.ml.config.ridge_params import RidgeRegressionParams
from app.ml.config.xgboost_params import XGBoostParams


logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Finds the best hyperparameters using GridSearchCV with activity-level TimeSeriesSplit.

    Fold indices are generated at activity level (no activity appears in both train
    and validation within a fold), then mapped back to segment row indices for
    GridSearchCV. This prevents leakage from segment-level splitting.

    Usage:
        tuner = HyperparameterTuner("ridge", RIDGE_PARAM_GRID, splitter)
        result = tuner.tune(X_train, y_train, train_df)
    """

    def __init__(
        self,
        model_name: str,
        param_grid: dict[str, list],
        splitter: TemporalSplitter,
    ):
        self.model_name = model_name
        self.param_grid = param_grid
        self.splitter = splitter

    def _build_estimator(self):
        """Build a sklearn estimator for grid search."""
        if self.model_name == "ridge":
            # Pipeline: scale first, then Ridge
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ])
        return xgb.XGBRegressor(
            objective="reg:squarederror", random_state=42, verbosity=0
        )

    def _prefixed_grid(self) -> dict:
        """Add 'model__' prefix for Pipeline params (Ridge only)."""
        if self.model_name == "ridge":
            return {f"model__{k}": v for k, v in self.param_grid.items()}
        return self.param_grid

    def _activity_level_cv_folds(
        self, train_df: pd.DataFrame, n_splits: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate (train_row_indices, val_row_indices) for each CV fold.

        TimeSeriesSplit is applied to the sorted list of activity_ids so that
        no activity ever appears in both the train and validation portions of a fold.
        """
        activity_ids = np.array(self.splitter._get_sorted_activity_ids(train_df))
        tscv = TimeSeriesSplit(n_splits)
        
        act_to_rows = {}
        for row_pos, act_id in enumerate(train_df["activity_id"]):
            act_to_rows.setdefault(act_id, []).append(row_pos)

        folds = []
        for train_idx, val_idx in tscv.split(activity_ids):
            train_rows = np.concatenate([act_to_rows[a] for a in activity_ids[train_idx]])
            val_rows = np.concatenate([act_to_rows[a] for a in activity_ids[val_idx]])
            folds.append((train_rows, val_rows))
        return folds

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_df: pd.DataFrame,
        n_splits: int = DEFAULT_CV_SPLITS,
    ) -> dict:
        """Run GridSearchCV with activity-level CV folds and return the best params and score.

        Args:
            X: Feature matrix aligned with train_df rows.
            y: Target array aligned with train_df rows.
            train_df: DataFrame used to determine activity-level folds

        Returns dict with:
            best_params — the winning hyperparameter combination for model
            best_score  — the best average MAE across CV folds
        """
        estimator = self._build_estimator()
        grid = self._prefixed_grid()
        cv_folds = self._activity_level_cv_folds(train_df, n_splits=n_splits)

        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            cv=cv_folds,
            scoring="neg_mean_absolute_error",
            refit=False,
            n_jobs=1,
        )
        search.fit(X, y)

        # Clean param names (remove pipeline prefix)
        best_params = {
            k.replace("model__", ""): v for k, v in search.best_params_.items()
        }
        best_score = -search.best_score_

        logger.info("Best params: %s (MAE: %.1fs)", best_params, best_score)

        return {
            "best_params": best_params,
            "best_score": best_score,
        }

    def update_hyperparams(self, param_overrides: dict):
        """Override the updated hyperparameters"""
        
        defaults = {"ridge": RidgeRegressionParams, "xgboost": XGBoostParams}
        base = defaults[self.model_name]()

        for key, value in param_overrides.items():
            if key in {f.name for f in fields(base)}:
                setattr(base, key, value)

        return base
