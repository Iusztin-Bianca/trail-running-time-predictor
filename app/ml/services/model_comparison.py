"""Compare trained models, compute SHAP importance, and save results."""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

LOCAL_COMPARISON_PATH = Path(__file__).parent.parent / "evaluation" / "model_draft_comparison.json"

from app.constants import DEFAULT_EXCLUDE_COLUMNS
from app.ml.evaluation.shap_analyzer import SHAPAnalyzer

logger = logging.getLogger(__name__)

# Descriptions for each feature used in training.
_SHAP_FEATURE_DESCRIPTIONS = {
    "segment_distance_m":     "Segment length (m)",
    "segment_time_sec":       "Segment duration (s) — model target, not an input feature",
    "segment_pace_mps":       "Average segment speed (m/s)",
    "elevation_gain_m":       "Net elevation gain in segment (m)",
    "elevation_loss_m":       "Net elevation loss in segment (m)",
    "avg_gradient":           "Mean absolute gradient of segment (0.1 = 10%) — unsigned",
    "std_gradient":           "Standard deviation of gradient in segment — measures terrain irregularity",
    "max_uphill_gradient":    "Max uphill gradient in segment (0 for non-uphill segments)",
    "max_downhill_gradient":  "Max downhill gradient in segment (0 for non-downhill segments)",
    "avg_elevation":          "Mean elevation of segment (m above sea level)",
    "uphill_cost":            "Uphill cost: distance * (1 + 6 * gradient) — 0 for non-uphill segments",
    "downhill_cost":          "Downhill cost: distance * (1 + 6 * |gradient|) — 0 for non-downhill segments",
    "cumulative_elevation":   "Cumulative elevation gain from activity start to end of this segment (m) — proxy for accumulated fatigue",
    "cumulative_distance":    "Cumulative distance from activity start to end of this segment (m) — proxy for accumulated fatigue",
    "segment_energy_cost":    "Minetti energy cost (J/kg): 155.4*g^5 - 30.4*g^4 - 43.3*g^3 + 46.3*g^2 + 19.5*g + 3.6, multiplied by distance — validated biomechanical formula for running on slopes",
    "is_race":                "1 if parent activity is a race, 0 otherwise — captures intentional maximum effort",
    "is_easy":                "1 if parent activity is a recovery run, 0 otherwise — captures intentional low effort"
}


class ModelComparisonService:
    """Compares already-trained models, computes SHAP importance, and saves results.

    This service receives the training data and model results, then:
      - Computes SHAP feature importance for each model
      - Builds a structured comparison dict (metrics + best model selection)
      - Saves results to results/model_draft_comparison.json (local) and on Blob Storage
      - Appends a new entry to results/models_compare_history.json (local) or Blob Storage

    Args:
        shap_analyzer: Optional SHAPAnalyzer instance.
        blob_manager: Optional BlobStorageManager. When provided, results are saved
                      to Azure Blob Storage instead of the local filesystem.
    """

    def __init__(self, shap_analyzer: SHAPAnalyzer | None = None, blob_manager=None):
        self.shap_analyzer = shap_analyzer or SHAPAnalyzer()
        self.blob_manager = blob_manager

    def run(self, df: pd.DataFrame, model_results: dict) -> dict:
        """
        Compute SHAP, build comparison, and save results.

        Args:
            df: Segment-level DataFrame used for training (needed for SHAP and statistics).
            model_results: the already trained models
        """
        logger.info("Starting model comparison")

        # 1. SHAP feature importance for each trained model
        feature_cols = [c for c in df.columns if c not in DEFAULT_EXCLUDE_COLUMNS]
        X_df = df[feature_cols].fillna(0)

        for model_name, result in model_results.items():
            result["shap_importance"] = self.shap_analyzer.analyze(result["model"], X_df)
            logger.info("SHAP computed for %s.", model_name)

        # 2. Build comparison dict and pick the best model
        comparison = self._build_comparison(df, model_results)

        # 3. Save results locally and to Blob Storage
        self._save_results_locally(comparison)
        self._save_results_to_blob(comparison)
        self._append_to_history_blob(comparison)

        return comparison

    def _build_comparison(self, df: pd.DataFrame, model_results: dict) -> dict:
        """Build a structured comparison from all model results."""
        total_activities = df["activity_id"].nunique()
        avg_segment_duration = round(df["segment_time_sec"].mean(), 1) if "segment_time_sec" in df.columns else None
        avg_race_duration = round(
            df.groupby("activity_id")["segment_time_sec"].sum().mean(), 1
        ) if "segment_time_sec" in df.columns else None

        comparison = {
            "timestamp": datetime.now().isoformat(),
            "dataset": {
                "total_segments": len(df),
                "total_activities": total_activities,
                "avg_segments_per_race": round(len(df) / total_activities, 1),
                "avg_segment_duration_sec": avg_segment_duration,
                "avg_race_duration_sec": avg_race_duration,
            },
            "models": {},
            "best_model": None,
        }

        best_name = None
        best_mae = float("inf")

        for name, result in model_results.items():
            race_test  = result["test_metrics_race"]
            race_train = result["train_metrics_race"]
            shap_values = result.get("shap_importance", {})

            entry = {
                "best_params": result["best_params"],
                "cv_metrics_race": {
                    "note": (
                        "Cross validation metrics(race level) "
                        "The model is trained on older activities and validated on the recent activities(TimeSeriesSplit)"
                        "MAE/RMSE measured in seconds, MAPE in percentage, R2 is between 0 and 1"
                    ),
                    **result["cv_metrics"],
                },
                "train_metrics_race": {
                    "note": (
                        "Metrics on the final training set(race-level): segment prediction are added up"
                        "to obtain the race prediction. Low error on training set/ High error on test set"
                        "can be a sign of overfitting!"
                    ),
                    **race_train,
                },
                "test_metrics_race": {
                    "note": (
                        "Metrics on the final training set (race-level, unseen running activities): "
                        "segment predictions are added up for each activity and compared with the actual race time of the activity "
                        "This is the main evaluation metric"
                    ),
                    **race_test,
                },
                "shap_importance": {
                    "note": (
                        "The importance of each feature calculated with SHAP (SHapley Additive exPlanations) "
                        "Each value represents the average of the absolute SHAP values ​​over the entire dataset(in seconds)"
                        "Interpretation: a feature that has a shap value of 60 means that "
                        "on average, that feature change the segment time prediction by "
                        "±60 secunde. The features are sorted descending based on the impact."
                    ),
                    "feature_descriptions": {
                        feat: _SHAP_FEATURE_DESCRIPTIONS.get(feat, "—")
                        for feat in shap_values
                    },
                    "values": shap_values,
                },
            }
            comparison["models"][name] = entry

            # Select best model by lowest race-level test MAE
            if race_test["mae"] < best_mae:
                best_mae = race_test["mae"]
                best_name = name

        comparison["best_model"] = best_name

        logger.info("=" * 50)
        logger.info("COMPARISON RESULTS")
        logger.info("=" * 50)
        logger.info("Best model: %s (Race Test MAE: %.1fs)", best_name, best_mae)

        return comparison

    def _save_results_locally(self, comparison: dict) -> None:
        """Save latest comparison results to local file."""
        LOCAL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_COMPARISON_PATH.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
        logger.info("Saved model_draft_comparison.json locally at %s", LOCAL_COMPARISON_PATH)

    def _save_results_to_blob(self, comparison: dict) -> None:
        """Upload latest comparison results to Blob Storage."""
        blob_client = self.blob_manager.blob_service_client.get_blob_client(
            container=self.blob_manager.container_name,
            blob="results/model_draft_comparison.json",
        )
        blob_client.upload_blob(
            json.dumps(comparison, indent=2, default=str).encode("utf-8"),
            overwrite=True,
        )
        logger.info("Uploaded results/model_draft_comparison.json to Blob Storage")

    def _append_to_history_blob(self, comparison: dict) -> None:
        """Append a training run summary to models_compare_history.json in Blob Storage."""
        history_client = self.blob_manager.blob_service_client.get_blob_client(
            container=self.blob_manager.container_name,
            blob="results/models_compare_history.json",
        )

        if history_client.exists():
            history = json.loads(history_client.download_blob().readall().decode("utf-8"))
        else:
            history = {
                "shap_feature_descriptions": _SHAP_FEATURE_DESCRIPTIONS,
                "training_runs": [],
            }

        entry = {
            "trained_at": comparison["timestamp"],
            "dataset": comparison["dataset"],
            "best_model": comparison["best_model"],
            "models": {
                model_name: {
                    "best_params": model_data["best_params"],
                    "test_metrics_race": {
                        k: v for k, v in model_data["test_metrics_race"].items()
                        if k != "note"
                    },
                    "train_metrics_race": {
                        k: v for k, v in model_data["train_metrics_race"].items()
                        if k != "note"
                    },
                    "shap_values": model_data["shap_importance"]["values"],
                }
                for model_name, model_data in comparison["models"].items()
            },
        }
        history["training_runs"].append(entry)

        history_client.upload_blob(
            json.dumps(history, indent=2, default=str).encode("utf-8"),
            overwrite=True,
        )
        logger.info(
            "History updated: %d total runs — results/models_compare_history.json",
            len(history["training_runs"]),
        )
