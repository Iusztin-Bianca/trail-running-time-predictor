import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.ml.data import BlobStorageManager
from .routes import health, predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

LOCAL_MODEL_PATH = Path(__file__).parent.parent / "models" / "model_latest.joblib"

def _load_local_model():
    """Load model from local fallback file."""
    if LOCAL_MODEL_PATH.exists():
        model = joblib.load(LOCAL_MODEL_PATH)
        logger.info("Model loaded from local fallback: %s", LOCAL_MODEL_PATH)
        return model
    logger.warning("No local model found at %s.", LOCAL_MODEL_PATH)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_name = None
    app.state.model_version = None

    if settings.azure_storage_connection_string:
        try:
            blob_manager = BlobStorageManager()
            if blob_manager.model_exists():
                app.state.model = blob_manager.download_model()
                metadata = blob_manager.download_model_metadata()
                app.state.model_name = metadata.get("model_name", "unknown") if metadata else "unknown"
                app.state.model_version = metadata.get("version") if metadata else None
                logger.info("Model v%s (%s) loaded from Blob Storage.", app.state.model_version, app.state.model_name)
            else:
                logger.warning("No model in Blob Storage — falling back to local model.")
                app.state.model = _load_local_model()
        except Exception as e:
            logger.warning("Blob Storage unavailable (%s) — falling back to local model.", e)
            app.state.model = _load_local_model()
    else:
        logger.warning("Azure Storage not configured — loading local model.")
        app.state.model = _load_local_model()

    yield

app = FastAPI(title="Trail Running Time Predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
