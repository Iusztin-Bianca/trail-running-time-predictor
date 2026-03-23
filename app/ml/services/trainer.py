"""Model trainer: segment-level training, race-level evaluation."""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from app.constants import DEFAULT_CV_SPLITS, DEFAULT_EXCLUDE_COLUMNS, SEGMENT_TARGET_COLUMN
from app.ml.data.data_splitter import TemporalSplitter
from app.ml.evaluation.metrics import MetricsCalculator
from app.ml.models.base_model import BaseModel
from app.ml.models.model_factory import ModelFactory
from app.ml.services.hyperparameter_tuner import HyperparameterTuner

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Orchestrates segment-level training with race-level cross-validation and evaluation.

    Training:   one row = one segment (segment_time_sec as target)
    Evaluation: predictions are summed per activity_id → compared to actual race time
    CV splits:  TimeSeriesSplit on sorted activity_ids (no activity is splited between train/val)
    TimeSeriesSplit: training is done at segment-level
    """

    def __init__(
        self,
        model: BaseModel,
        splitter: TemporalSplitter,
        metrics: MetricsCalculator,
        target_column: str = SEGMENT_TARGET_COLUMN,
        exclude_columns: set[str] | None = None,
        log_transform_target: bool = False,
    ):
        self.model = model
        self.splitter = splitter
        self.metrics = metrics
        self.target_column = target_column
        self.exclude_columns = exclude_columns or DEFAULT_EXCLUDE_COLUMNS
        self.log_transform_target = log_transform_target

    def _transform_y(self, y: np.ndarray) -> np.ndarray:
        if self.log_transform_target:
            return np.log1p(y)
        return y

    def _inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        if self.log_transform_target:
            return np.expm1(y)
        return y

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        return [
            col for col in df.columns
            if col not in self.exclude_columns and col != self.target_column
        ]

    def _prepare_xy(self, df: pd.DataFrame, feature_cols: list[str]):
        """Extract X (features) and y (target) arrays. Fills NaN with 0."""
        X = df[feature_cols].fillna(0).values
        y = df[self.target_column].values
        return X, y

    @staticmethod
    def _compute_sample_weights(df: pd.DataFrame, race_weight: float = 3.0) -> np.ndarray:
        """Return per-sample weights: race_weight for race segments (intensity_level==2), 1 otherwise."""
        return np.where(df["intensity_level"].values == 2, race_weight, 1.0)


    def _race_level_metrics(self, df: pd.DataFrame, y_pred_segments: np.ndarray) -> dict:
        """Aggregate segment predictions to race level and compute metrics.

        Time target:  predicted_time = sum of predicted segment times
        Actual time is always taken from segment_time_sec.
        """
        df = df.copy()
        df["_pred"] = y_pred_segments

        race_pred = df.groupby("activity_id")["_pred"].sum().values

        race_actual = df.groupby("activity_id")["segment_time_sec"].sum().values
        return self.metrics.calculate(race_actual, race_pred)

    def _segment_level_metrics(self, df: pd.DataFrame, y_pred_segments: np.ndarray) -> dict:
        """Compute metrics directly at segment level."""
        y_actual = df[self.target_column].values
        return self.metrics.calculate(y_actual, y_pred_segments)

    def cross_validate(
        self, df: pd.DataFrame, n_splits: int = DEFAULT_CV_SPLITS, model=None
    ) -> dict[str, float]:
        """Temporal CV at activity level, metrics reported at race level.

        TimeSeriesSplit is applied to the sorted list of activity_ids.
        Each fold trains on segments of earlier activities and validates
        on segments of later activities. Predictions are then aggregated
        per activity to compute race-level metrics.
        """
        model = model or self.model
        feature_cols = self._get_feature_columns(df)
        activity_ids = np.array(self.splitter._get_sorted_activity_ids(df))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_metrics = []

        for fold, (train_act_idx, val_act_idx) in enumerate(tscv.split(activity_ids), start=1):
            train_acts = activity_ids[train_act_idx]
            val_acts = activity_ids[val_act_idx]

            train_df = df[df["activity_id"].isin(train_acts)]
            val_df = df[df["activity_id"].isin(val_acts)].copy()

            X_train, y_train = self._prepare_xy(train_df, feature_cols)
            X_val, _ = self._prepare_xy(val_df, feature_cols)
            weights = self._compute_sample_weights(train_df)

            model.fit(X_train, self._transform_y(y_train), sample_weight=weights)
            y_pred_seg = self._inverse_transform_y(model.predict(X_val))

            metrics = self._race_level_metrics(val_df, y_pred_seg)
            fold_metrics.append(metrics)

            logger.info(
                "Fold %d/%d — %d val activities — MAE: %.1fs, RMSE: %.1fs, R²: %.3f",
                fold, n_splits, len(val_acts),
                metrics["mae"], metrics["rmse"], metrics["r2"],
            )

        avg_metrics = {
            key: float(np.mean([m[key] for m in fold_metrics]))
            for key in fold_metrics[0]
        }
        logger.info(
            "CV complete (%d folds) — Avg Race MAE: %.1fs, Avg RMSE: %.1fs, Avg R²: %.3f",
            n_splits, avg_metrics["mae"], avg_metrics["rmse"], avg_metrics["r2"],
        )
        return avg_metrics

    def tune_and_train(self, df: pd.DataFrame, param_grid: dict[str, list]) -> dict:
        """Training pipeline with hyperparameter tuning:
        1. Split at activity level (temporal holdout for test set)
        2. Tune hyperparameters on train segments (GridSearchCV, activity-level folds)
        3. Train best model on all train segments
        4. Cross-validate best model (activity-level, race-level metrics)
        5. Evaluate on test activities (race-level metrics)
        """
        logger.info("Starting tune_and_train for %s.", self.model.name)

        # 1. Activity-level temporal holdout
        train_df, test_df = self.splitter.split_train_test(df)
        feature_cols = self._get_feature_columns(train_df)
        X_train, y_train = self._prepare_xy(train_df, feature_cols)
        y_train_t = self._transform_y(y_train)

        # 2. Tune hyperparameters on train segments
        tuner = HyperparameterTuner(
            model_name=self.model.name,
            param_grid=param_grid,
            splitter=self.splitter,
        )
        tune_weights = self._compute_sample_weights(train_df)
        tune_result = tuner.tune(X_train, y_train_t, train_df, sample_weight=tune_weights)

        # 3. Build best model with tuned hyperparameters
        best_hyperparams = tuner.update_hyperparams(tune_result["best_params"])
        best_model = ModelFactory.create(self.model.name, best_hyperparams)

        # 4. Race-level CV with best model on train set
        # cross_validate() refits the model on each fold, so refit on full train after
        cv_metrics = self.cross_validate(train_df, model=best_model)
        train_weights = self._compute_sample_weights(train_df)
        best_model.fit(X_train, y_train_t, sample_weight=train_weights)
        logger.info("Best model fitted on full train set (%d segments) after CV.", len(X_train))

        # 5. Train set metrics (overfitting check: compare with test metrics)
        y_pred_train = self._inverse_transform_y(best_model.predict(X_train))
        train_metrics_race = self._race_level_metrics(train_df, y_pred_train)
        train_metrics_segment = self._segment_level_metrics(train_df, y_pred_train)

        logger.info(
            "Train set — %d activities — Race MAE: %.1fs, RMSE: %.1fs, R²: %.3f | "
            "Segment MAE: %.1fs, RMSE: %.1fs, R²: %.3f",
            train_df["activity_id"].nunique(),
            train_metrics_race["mae"], train_metrics_race["rmse"], train_metrics_race["r2"],
            train_metrics_segment["mae"], train_metrics_segment["rmse"], train_metrics_segment["r2"],
        )

        # 6. Race-level and segment-level evaluation on test set
        X_test, _ = self._prepare_xy(test_df, feature_cols)
        y_pred_seg = self._inverse_transform_y(best_model.predict(X_test))
        test_metrics_race = self._race_level_metrics(test_df, y_pred_seg)
        test_metrics_segment = self._segment_level_metrics(test_df, y_pred_seg)

        logger.info(
            "Test set — %d activities — Race MAE: %.1fs, RMSE: %.1fs, R²: %.3f | "
            "Segment MAE: %.1fs, RMSE: %.1fs, R²: %.3f",
            test_df["activity_id"].nunique(),
            test_metrics_race["mae"], test_metrics_race["rmse"], test_metrics_race["r2"],
            test_metrics_segment["mae"], test_metrics_segment["rmse"], test_metrics_segment["r2"],
        )

        return {
            "best_params": tune_result["best_params"],
            "cv_metrics": cv_metrics,
            "train_metrics_race": train_metrics_race,
            "train_metrics_segment": train_metrics_segment,
            "test_metrics_race": test_metrics_race,
            "test_metrics_segment": test_metrics_segment,
            "model": best_model,
        }

    @classmethod
    def train_all(
        cls,
        df: pd.DataFrame,
        splitter: TemporalSplitter,
        metrics: MetricsCalculator,
        log_transform_target: bool = False,
    ) -> dict:
        """
        Train every available model with hyperparameter tuning.

        Loops over all models registered in ModelFactory, creates a dedicated
        ModelTrainer for each, and runs tune_and_train(). 
        """
        model_results: dict = {}

        for model_name in ModelFactory.available_models():
            logger.info("=" * 50)
            logger.info("Training model: %s", model_name.upper())
            logger.info("=" * 50)

            model = ModelFactory.create(model_name)
            trainer = cls(
                model=model,
                splitter=splitter,
                metrics=metrics,
                target_column=SEGMENT_TARGET_COLUMN,
                log_transform_target=log_transform_target,
            )
            model_results[model_name] = trainer.tune_and_train(
                df, ModelFactory.get_param_grid(model_name)
            )

        return model_results
