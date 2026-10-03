from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from app.schemas import MFCCFeatures, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from app.predictor import AcousticClassifierService

predictor = AcousticClassifierService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up: load tree ensemble weights once into memory
    predictor.load()
    yield
    # Teardown logic if required

app = FastAPI(
    title="Dysarthric Acoustic Classification API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "model_ready": predictor.model is not None}

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(features: MFCCFeatures):
    try:
        return predictor.predict_sample(features)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post("/api/v1/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(payload: BatchPredictionRequest):
    predictions = [predictor.predict_sample(sample) for sample in payload.samples]
    return BatchPredictionResponse(predictions=predictions)