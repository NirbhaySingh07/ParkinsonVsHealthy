from pydantic import BaseModel, Field
from typing import List, Dict

class MFCCFeatures(BaseModel):
    MFCC1_mean: float
    MFCC1_std: float
    MFCC2_mean: float
    MFCC2_std: float
    MFCC3_mean: float
    MFCC3_std: float
    MFCC4_mean: float
    MFCC4_std: float
    MFCC5_mean: float
    MFCC5_std: float
    MFCC6_mean: float
    MFCC6_std: float
    MFCC7_mean: float
    MFCC7_std: float
    MFCC8_mean: float
    MFCC8_std: float
    MFCC9_mean: float
    MFCC9_std: float
    MFCC10_mean: float
    MFCC10_std: float
    MFCC11_mean: float
    MFCC11_std: float
    MFCC12_mean: float
    MFCC12_std: float
    MFCC13_mean: float
    MFCC13_std: float

    # Pydantic v1 configuration style
    class Config:
        schema_extra = {
            "example": {
                "MFCC1_mean": -73.06, "MFCC1_std": 7.90,
                "MFCC2_mean": 88.97,  "MFCC2_std": 3.49,
                "MFCC3_mean": -50.95, "MFCC3_std": 6.58,
                "MFCC4_mean": -1.44,  "MFCC4_std": 5.64,
                "MFCC5_mean": -28.24, "MFCC5_std": 6.21,
                "MFCC6_mean": 12.30,  "MFCC6_std": 4.10,
                "MFCC7_mean": -8.10,  "MFCC7_std": 3.80,
                "MFCC8_mean": 15.20,  "MFCC8_std": 5.01,
                "MFCC9_mean": -5.40,  "MFCC9_std": 4.11,
                "MFCC10_mean": 19.50, "MFCC10_std": 3.72,
                "MFCC11_mean": -39.16,"MFCC11_std": 4.42,
                "MFCC12_mean": 7.41,  "MFCC12_std": 4.09,
                "MFCC13_mean": -32.81,"MFCC13_std": 4.00
            }
        }

class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Diagnosis label ('HC' or 'PD')")
    class_id: int
    probabilities: Dict[str, float]
    inference_latency_ms: float

class BatchPredictionRequest(BaseModel):
    samples: List[MFCCFeatures]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]