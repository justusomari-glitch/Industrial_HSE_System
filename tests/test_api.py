from fastapi.testclient import TestClient
from src.predict import app


def test_home_route():
    with TestClient(app) as client:
        response=client.get('/')
        assert response.status_code==200
        assert "Message" in response.json()

def test_predict_valid_input():
    with TestClient(app) as client:
        payload={
            "temperature": 100.3,
            "humidity": 2.5,
            "noise_level": 12.5,
            "gas_level": 44.1,
            "vibration": 87.7,
            "voltage": 6.8,
            "pressure": 3.6,
            "co_ppm": 9.2,
            "smoke_level": 5.8,
            "hours_worked": 4,
            "days_consecutive": 44.6,
            "ppe_compliance": 0.6,
            "break_compliance": 1,
            "shift": "Morning",
            "zone": "Zone_D_HighRisk"
        }
        response=client.post("/predict",json=payload)
        assert response.status_code==200
        data=response.json

def test_predict_invaid_input():
    with TestClient(app) as client:
        payload={
            "temperature": 100.3,
            "humidity": 2.5,
            "noise_level": 12.5,
            "gas_level": 44.1,
        }
        response=client.post("/predict",json=payload)
        assert response.status_code==422
