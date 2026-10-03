import joblib
import numpy as np
import time
from typing import List
from app.schemas import MFCCFeatures, PredictionResponse

class AcousticClassifierService:
    def __init__(self, artifact_path: str = "artifacts/parkinson_acoustic_model.joblib"):
        self.artifact_path = artifact_path
        self.model = None
        self.features = []
        self.classes = []

    def load(self):
        data = joblib.load(self.artifact_path)
        self.model = data["model"]
        self.features = data["features"]
        self.classes = data["classes"]

    def predict_sample(self, sample: MFCCFeatures) -> PredictionResponse:
        start_time = time.perf_counter()
        
        # Enforce exact feature order expected by scikit-learn
        feature_vector = np.array([[getattr(sample, feat) for feat in self.features]], dtype=np.float64)
        
        class_id = int(self.model.predict(feature_vector)[0])
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        latency = (time.perf_counter() - start_time) * 1000

        return PredictionResponse(
            predicted_class=self.classes[class_id],
            class_id=class_id,
            probabilities={
                self.classes[i]: float(prob) for i, prob in enumerate(probabilities)
            },
            inference_latency_ms=round(latency, 3)
        )