import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image

from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "MedVision-CNN"
    assert "disclaimer" in data

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "architecture" in data
    assert "metrics" in data

def test_predict_endpoint():
    # Create a dummy 28x28 grayscale image in memory
    img = Image.new("L", (28, 28), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test_xray.png", buf, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "gradcam_heatmap_b64" in data
    assert "medical_disclaimer" in data
