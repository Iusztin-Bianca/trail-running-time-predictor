"""
Train and compare all models on segment-level data.

- Loads segments_training_features.parquet from blob storage
- Splits at activity level (no activity spans train + val/test)
- Trains at segment level, evaluates at race level
- Saves results to results/model_draft_comparison.json
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.data.blob_storage import BlobStorageManager
from app.ml.data.data_splitter import TemporalSplitter
from app.ml.evaluation.metrics import MetricsCalculator
from app.ml.services.model_comparisor import ModelComparisonService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("MODEL TRAINING - STARTING")
    logger.info("=" * 70)

    blob_manager = BlobStorageManager()
    splitter = TemporalSplitter(test_ratio=0.15)
    metrics = MetricsCalculator()

    service = ModelComparisonService(
        blob_manager=blob_manager,
        splitter=splitter,
        metrics=metrics,
        log_transform_target=False,
    )

    results = service.run()

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("Dataset: %d segments, %d activities",
                results["dataset"]["total_segments"],
                results["dataset"]["total_activities"])
    logger.info("Best model: %s", results["best_model"].upper())

    for name, entry in results["models"].items():
        r = entry["test_metrics_race"]
        s = entry["test_metrics_segment"]
        logger.info(
            "%s — CV Race MAE: %.1fs | Test race: MAE %.1fs, MAPE %.1f%%, R² %.3f"
            " | Test seg: MAE %.1fs, MAPE %.1f%%, R² %.3f",
            name.upper(), entry["cv_metrics_race"]["mae"],
            r["mae"], r["mape"], r["r2"],
            s["mae"], s["mape"], s["r2"],
        )


if __name__ == "__main__":
    main()
