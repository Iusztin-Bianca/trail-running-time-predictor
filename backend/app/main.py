from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
import logging

from app.feature_engineering import FeatureExtractor
from app.ml.data import BlobStorageManager
from app.config.settings import settings
from .schemas import PredictionResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup logic.
    Loads the latest model from Blob Storage.
    """
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
            "Trigger the Azure Function 'strava_monthly_update' to fetch data and train the first model."
        )

    yield

app = FastAPI(title="Trail Running Time Predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict():
    predicted_time = 360  # placeholder
    return PredictionResponse(predicted_time_minutes=predicted_time)

@app.post("/predict-from-gpx")
def predict_from_gpx(file: UploadFile = File(...)):
    print("File: ", file);
    if not file.filename.endswith(".gpx"):
        raise HTTPException(status_code=400, detail="Invalid file type. GPX required.")

    file_bytes = file.read()

    try:
        extractor = FeatureExtractor()
        features = extractor.extract_from_gpx(file_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to parse GPX file.")

    # Temporary placeholder prediction logic
    predicted_time_minutes = 1500

    return {
        "predicted_time_minutes": round(predicted_time_minutes, 1)
    }
