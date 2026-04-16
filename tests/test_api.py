"""
Unit Tests for API Endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestGeneralEndpoints:
    """Tests for general API endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_status(self, client):
        """Test status endpoint."""
        response = client.get("/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "models_loaded" in data
        assert "uptime_seconds" in data


class TestPredictionEndpoints:
    """Tests for prediction API endpoints."""
    
    def test_predict_anomaly(self, client):
        """Test anomaly prediction endpoint."""
        reading = {
            "voltage": 38.5,
            "current": 9.2,
            "power_output": 354.2,
            "panel_temperature": 42.5,
            "inverter_temperature": 48.3,
            "ambient_temperature": 32.1,
            "irradiance": 850.0,
            "humidity": 45.0,
            "wind_speed": 3.5,
            "dust_index": 15.0
        }
        
        response = client.post(
            "/predict/anomaly?asset_id=test_panel",
            json=reading
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "anomaly_score" in data
        assert "is_anomaly" in data
        assert 0 <= data["anomaly_score"] <= 1
    
    def test_predict_failure(self, client):
        """Test failure prediction endpoint."""
        reading = {
            "voltage": 38.5,
            "current": 9.2,
            "power_output": 354.2,
            "panel_temperature": 55.0,
            "inverter_temperature": 65.0,
            "ambient_temperature": 35.0,
            "irradiance": 900.0
        }
        
        response = client.post(
            "/predict/failure?asset_id=test_panel",
            json=reading
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "failure_probability" in data
        assert "risk_level" in data
        assert 0 <= data["failure_probability"] <= 1
    
    def test_predict_efficiency(self, client):
        """Test efficiency prediction endpoint."""
        readings = {
            "asset_id": "test_panel",
            "readings": [
                {
                    "voltage": 38.5,
                    "current": 9.2,
                    "power_output": 354.2,
                    "panel_temperature": 42.5,
                    "inverter_temperature": 48.3,
                    "ambient_temperature": 32.1,
                    "irradiance": 850.0
                }
            ]
        }
        
        response = client.post(
            "/predict/efficiency?asset_id=test_panel",
            json=readings
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "current_efficiency" in data
        assert "forecasted_efficiency" in data
        assert "trend" in data


class TestHealthEndpoints:
    """Tests for health-related endpoints."""
    
    def test_get_asset_health(self, client):
        """Test health score endpoint."""
        response = client.get(
            "/health/test_panel",
            params={
                "anomaly_score": 0.2,
                "failure_probability": 0.1,
                "performance_ratio": 0.85
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "health_score" in data
        assert "status" in data
        assert "recommendation" in data
        assert 0 <= data["health_score"] <= 100
    
    def test_health_score_calculation(self, client):
        """Test health score varies with inputs."""
        # Good health
        response_good = client.get(
            "/health/test",
            params={"anomaly_score": 0.1, "failure_probability": 0.05, "performance_ratio": 0.95}
        )
        
        # Poor health
        response_poor = client.get(
            "/health/test",
            params={"anomaly_score": 0.8, "failure_probability": 0.7, "performance_ratio": 0.4}
        )
        
        assert response_good.json()["health_score"] > response_poor.json()["health_score"]


class TestAlertEndpoints:
    """Tests for alert-related endpoints."""
    
    def test_get_alerts(self, client):
        """Test get alerts endpoint."""
        response = client.get("/alerts")
        
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_get_alert_summary(self, client):
        """Test alert summary endpoint."""
        response = client.get("/alerts/summary")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total" in data
        assert "active" in data


class TestInputValidation:
    """Tests for input validation."""
    
    def test_invalid_sensor_reading(self, client):
        """Test handling of invalid sensor readings."""
        # Missing required fields
        reading = {
            "voltage": 38.5
            # Missing other required fields
        }
        
        response = client.post(
            "/predict/anomaly?asset_id=test",
            json=reading
        )
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_negative_values(self, client):
        """Test handling of negative values where not allowed."""
        reading = {
            "voltage": -10,  # Invalid
            "current": 9.2,
            "power_output": 354.2,
            "panel_temperature": 42.5,
            "inverter_temperature": 48.3,
            "ambient_temperature": 32.1,
            "irradiance": 850.0
        }
        
        response = client.post(
            "/predict/anomaly?asset_id=test",
            json=reading
        )
        
        # Should return validation error for negative voltage
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
