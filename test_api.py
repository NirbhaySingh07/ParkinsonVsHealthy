from fastapi.testclient import TestClient
from app.main import app

# Create a dummy client to talk to your API without actually starting a server
client = TestClient(app)

def test_health_check():
    """Test if the API wakes up successfully."""
    response = client.get("/health")
    assert response.status_code == 200

def test_pydantic_validation_error():
    """Test if your data contracts block bad data."""
    # Sending missing/wrong data should trigger a 422 Unprocessable Entity error
    bad_payload = {"MFCC1_mean": 42.0} # Missing the other 25 features
    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 422