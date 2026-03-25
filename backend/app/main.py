import logging
from contextlib import asynccontextmanager

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_name = None
    app.state.model_version = None

    if not settings.azure_storage_connection_string:
        logger.warning("Azure Storage not configured — model will not be loaded.")
        yield
        return

    blob_manager = BlobStorageManager()

    if blob_manager.model_exists():
        app.state.model = blob_manager.download_model()
        metadata = blob_manager.download_model_metadata()
        app.state.model_name = metadata.get("model_name", "unknown") if metadata else "unknown"
        app.state.model_version = metadata.get("version") if metadata else None
        logger.info("Model v%s (%s) loaded from Blob Storage.", app.state.model_version, app.state.model_name)
    else:
        logger.warning(
            "No trained model found in Blob Storage. "
        )

    yield

app = FastAPI(title="Trail Running Time Predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
