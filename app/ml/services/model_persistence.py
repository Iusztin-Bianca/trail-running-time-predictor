"""Save a pre-trained model to Blob Storage (versioned).

   ModelPersistenceService → receives model_results and saves the chosen model to Blob
   Usage (in main.py / Azure Function)
"""

import logging
from typing import Optional
from app.ml.data.blob_storage import BlobStorageManager

logger = logging.getLogger(__name__)

class ModelPersistenceService:

    def __init__(
        self,
        blob_manager: BlobStorageManager,
        production_model_name: str = "ridge",
    ):
        self.blob_manager = blob_manager
        self.production_model_name = production_model_name

    def save(self, model_results: dict) -> Optional[int]:
        """
        Save the production model to Blob Storage as a new versioned blob.
        Returns:
            The new version number, or None if the model was not found in model_results.
        """
        result = model_results.get(self.production_model_name)
        if result is None:
            logger.warning(
                "Model '%s' not found in available trained models — skipping save.",
                self.production_model_name,
            )
            return None

        model = result["model"]
        race_metrics = result["test_metrics_race"]

        version = self.blob_manager.upload_model(
            model, self.production_model_name, metrics=race_metrics
        )

        logger.info(
            "Model: '%s' saved as v%d — "
            "Test race MAE: %.1fs, MAPE: %.1f%%, R²: %.3f",
            self.production_model_name, version,
            race_metrics.get("mae", 0),
            race_metrics.get("mape", 0),
            race_metrics.get("r2", 0),
        )
        return version
