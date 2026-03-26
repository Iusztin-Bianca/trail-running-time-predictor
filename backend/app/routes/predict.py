import logging

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from app.ml.services.predictor import PredictorService
from ..schemas import PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/predict-from-gpx", response_model=PredictionResponse)
async def predict_from_gpx(
    request: Request,
    file: UploadFile = File(...),
    is_race: int = Query(default=1, ge=0, le=1, description="1 if predicting a race effort, 0 otherwise"),
    is_easy: int = Query(default=0, ge=0, le=1, description="1 if predicting a recovery run, 0 otherwise"),
):
    if not file.filename.endswith(".gpx"):
        raise HTTPException(status_code=400, detail="Invalid file type. GPX required.")

    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again later.")

    file_bytes = await file.read()

    try:
        result = PredictorService(request.app.state.model).predict_from_gpx(
            file_bytes, is_race=is_race, is_easy=is_easy
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during prediction.")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    return PredictionResponse.from_seconds(result["total_seconds"], result["num_segments"])
