import sys
import os
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Sinhala Agentic Fake News Detection API"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint():
    payload = {
        "text": "මෙම පුවතේ දැන්වීම අනතුරක් බවට පත් වෙලා තිබේ. රජය මේ ගැන නිවේදනය කරලා නැහැ.",
        "top_k": 3
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "claim" in data
    assert "verdict" in data
    assert "explanation_si" in data["verdict"]
    assert data["claim"]["claim_text"] is not None
