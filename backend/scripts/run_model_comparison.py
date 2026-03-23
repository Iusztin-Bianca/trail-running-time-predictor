"""
Script to run model comparison (Ridge vs XGBoost) with hyperparameter tuning.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from app.ml.data.blob_storage import BlobStorageManager
from app.ml.data.data_splitter import TemporalSplitter
from app.ml.evaluation.metrics import MetricsCalculator
from app.ml.services.model_comparison import ModelComparisonService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Run model comparison."""
    try:
        logger.info("=" * 100)
        logger.info("MODEL COMPARISON - STARTING")
        logger.info("=" * 100)

        # Initialize components
        blob_manager = BlobStorageManager()
        splitter = TemporalSplitter()  # uses DEFAULT_TEST_SIZE from constants.py
        metrics = MetricsCalculator()

        # Run comparison
        service = ModelComparisonService(
            blob_manager=blob_manager,
            splitter=splitter,
            metrics=metrics,
            log_transform_target=False
        )

        results = service.run()

        logger.info("=" * 100)
        logger.info("MODEL COMPARISON - COMPLETE")
        logger.info("=" * 100)

    except Exception as e:
        logger.error("=" * 100)
        logger.error("MODEL COMPARISON - FAILED")
        logger.error("=" * 100)
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
