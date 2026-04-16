"""
FastAPI Backend for Predictive Maintenance System

Endpoints:
- POST /predict/anomaly - Real-time anomaly scoring
- POST /predict/failure - Failure probability prediction
- POST /predict/efficiency - Efficiency forecast
- GET /alerts - Current active alerts
- GET /health/{asset_id} - Asset health score
- GET /status - System status
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import (
    calculate_health_score, get_health_status,
    check_alert_conditions, AlertManager
)

# Initialize FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="ML-powered predictive maintenance for renewable energy infrastructure",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
alert_manager = AlertManager()
models_loaded = False
anomaly_detector = None
failure_predictor = None
efficiency_forecaster = None


# ==================== Pydantic Models ====================

class SensorReading(BaseModel):
    """Single sensor reading."""
    timestamp: Optional[str] = None
    voltage: float = Field(..., ge=0, description="Voltage in V")
    current: float = Field(..., ge=0, description="Current in A")
    power_output: float = Field(..., ge=0, description="Power output in W")
    panel_temperature: float = Field(..., description="Panel temperature in °C")
    inverter_temperature: float = Field(..., description="Inverter temperature in °C")
    ambient_temperature: float = Field(..., description="Ambient temperature in °C")
    irradiance: float = Field(..., ge=0, description="Solar irradiance in W/m²")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Humidity %")
    wind_speed: Optional[float] = Field(None, ge=0, description="Wind speed m/s")
    dust_index: Optional[float] = Field(None, ge=0, le=100, description="Dust index")


class SensorBatch(BaseModel):
    """Batch of sensor readings for time-series analysis."""
    asset_id: str = Field(..., description="Asset identifier")
    readings: List[SensorReading] = Field(..., description="List of sensor readings")


class AnomalyResponse(BaseModel):
    """Anomaly detection response."""
    asset_id: str
    anomaly_score: float = Field(..., ge=0, le=1)
    is_anomaly: bool
    confidence: float
    timestamp: str
    breakdown: Optional[Dict[str, float]] = None


class FailureResponse(BaseModel):
    """Failure prediction response."""
    asset_id: str
    failure_probability: float = Field(..., ge=0, le=1)
    risk_level: str
    days_to_failure: Optional[int] = None
    confidence: float
    timestamp: str
    top_risk_factors: Optional[List[Dict[str, Any]]] = None


class EfficiencyResponse(BaseModel):
    """Efficiency forecast response."""
    asset_id: str
    current_efficiency: float
    forecasted_efficiency: List[float]
    forecast_timestamps: List[str]
    trend: str
    efficiency_change: float
    timestamp: str


class HealthResponse(BaseModel):
    """Asset health response."""
    asset_id: str
    health_score: float = Field(..., ge=0, le=100)
    status: str
    color: str
    recommendation: str
    components: Dict[str, float]
    timestamp: str


class Alert(BaseModel):
    """Alert model."""
    id: str
    asset_id: str
    type: str
    severity: str
    message: str
    value: float
    threshold: float
    timestamp: str
    acknowledged: bool


class SystemStatus(BaseModel):
    """System status response."""
    status: str
    models_loaded: bool
    uptime_seconds: float
    active_alerts: int
    last_prediction: Optional[str] = None


# ==================== Helper Functions ====================

def prepare_features(reading: SensorReading) -> np.ndarray:
    """Convert sensor reading to feature array."""
    # Basic features (order must match trained model)
    features = [
        reading.voltage,
        reading.current,
        reading.power_output,
        reading.panel_temperature,
        reading.inverter_temperature,
        reading.ambient_temperature,
        reading.irradiance,
        reading.humidity or 50.0,
        reading.wind_speed or 3.0,
        reading.dust_index or 10.0,
    ]
    
    # Calculate derived features
    expected_power = reading.irradiance * 2.0 * 0.2 * 10  # area * efficiency * num_panels
    performance_ratio = reading.power_output / max(expected_power, 1)
    
    features.extend([
        expected_power,
        performance_ratio,
        max(0, reading.panel_temperature - 35) / 10,  # temp_stress
    ])
    
    return np.array(features).reshape(1, -1)


def get_risk_level(probability: float) -> str:
    """Convert probability to risk level."""
    if probability >= 0.7:
        return "Critical"
    elif probability >= 0.4:
        return "High"
    elif probability >= 0.2:
        return "Medium"
    else:
        return "Low"


# ==================== Startup Event ====================

start_time = datetime.now()


@app.on_event("startup")
async def load_models():
    """Load ML models on startup."""
    global models_loaded, anomaly_detector, failure_predictor, efficiency_forecaster
    
    model_dir = PROJECT_ROOT / "models"
    
    try:
        # Check if models exist
        if (model_dir / "anomaly_detector").exists():
            from src.models.anomaly_detector import EnsembleAnomalyDetector
            anomaly_detector = EnsembleAnomalyDetector()
            anomaly_detector.load(str(model_dir / "anomaly_detector"))
            print("✓ Anomaly detector loaded")
        
        if (model_dir / "failure_predictor.joblib").exists():
            from src.models.failure_predictor import FailurePredictor
            failure_predictor = FailurePredictor()
            failure_predictor.load(str(model_dir / "failure_predictor.joblib"))
            print("✓ Failure predictor loaded")
        
        if (model_dir / "efficiency_forecaster.pt").exists():
            from src.models.efficiency_forecaster import EfficiencyForecaster
            efficiency_forecaster = EfficiencyForecaster()
            efficiency_forecaster.load(str(model_dir / "efficiency_forecaster.pt"))
            print("✓ Efficiency forecaster loaded")
        
        models_loaded = True
        print("\n✓ All available models loaded successfully")
        
    except Exception as e:
        print(f"Warning: Could not load models: {e}")
        print("API will work in demo mode with simulated predictions")
        models_loaded = False


# ==================== API Endpoints ====================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Predictive Maintenance API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/status", response_model=SystemStatus, tags=["General"])
async def get_status():
    """Get system status."""
    uptime = (datetime.now() - start_time).total_seconds()
    
    return SystemStatus(
        status="operational" if models_loaded else "demo_mode",
        models_loaded=models_loaded,
        uptime_seconds=uptime,
        active_alerts=len(alert_manager.get_active_alerts()),
        last_prediction=datetime.now().isoformat()
    )


@app.post("/predict/anomaly", response_model=AnomalyResponse, tags=["Predictions"])
async def predict_anomaly(
    asset_id: str = Query(..., description="Asset identifier"),
    reading: SensorReading = None
):
    """
    Detect anomalies in sensor readings.
    
    Returns anomaly score (0-1) where higher means more anomalous.
    """
    timestamp = datetime.now().isoformat()
    
    if models_loaded and anomaly_detector is not None:
        features = prepare_features(reading)
        scores = anomaly_detector.get_scores_breakdown(features)
        anomaly_score = float(scores['ensemble'][0])
        breakdown = {k: float(v[0]) for k, v in scores.items()}
    else:
        # Demo mode: simulate based on input
        temp_factor = max(0, (reading.panel_temperature - 50) / 30)
        power_factor = 1 - min(1, reading.power_output / 4000)
        anomaly_score = 0.3 * temp_factor + 0.3 * power_factor + 0.1 * np.random.random()
        anomaly_score = float(np.clip(anomaly_score, 0, 1))
        breakdown = None
    
    is_anomaly = anomaly_score > 0.5
    
    # Generate alerts if needed
    alerts = check_alert_conditions(
        anomaly_score=anomaly_score,
        failure_probability=0,
        efficiency=1,
        asset_id=asset_id
    )
    alert_manager.add_alerts([a for a in alerts if a['type'] == 'anomaly'])
    
    return AnomalyResponse(
        asset_id=asset_id,
        anomaly_score=round(anomaly_score, 4),
        is_anomaly=is_anomaly,
        confidence=0.85 if models_loaded else 0.6,
        timestamp=timestamp,
        breakdown=breakdown
    )


@app.post("/predict/failure", response_model=FailureResponse, tags=["Predictions"])
async def predict_failure(
    asset_id: str = Query(..., description="Asset identifier"),
    reading: SensorReading = None
):
    """
    Predict probability of failure within the next 7 days.
    """
    timestamp = datetime.now().isoformat()
    
    if models_loaded and failure_predictor is not None:
        features = prepare_features(reading)
        failure_prob = float(failure_predictor.predict_proba(features)[0])
        
        # Get top risk factors
        importance = failure_predictor.get_feature_importance(top_n=5)
        top_factors = importance.to_dict('records')
    else:
        # Demo mode
        temp_risk = max(0, (reading.inverter_temperature - 60) / 20)
        power_risk = max(0, (3000 - reading.power_output) / 3000)
        failure_prob = 0.2 * temp_risk + 0.2 * power_risk + 0.05 * np.random.random()
        failure_prob = float(np.clip(failure_prob, 0, 1))
        top_factors = None
    
    risk_level = get_risk_level(failure_prob)
    days_to_failure = int(7 * (1 - failure_prob)) if failure_prob > 0.3 else None
    
    # Generate alerts
    alerts = check_alert_conditions(
        anomaly_score=0,
        failure_probability=failure_prob,
        efficiency=1,
        asset_id=asset_id
    )
    alert_manager.add_alerts([a for a in alerts if a['type'] == 'failure'])
    
    return FailureResponse(
        asset_id=asset_id,
        failure_probability=round(failure_prob, 4),
        risk_level=risk_level,
        days_to_failure=days_to_failure,
        confidence=0.88 if models_loaded else 0.65,
        timestamp=timestamp,
        top_risk_factors=top_factors
    )


@app.post("/predict/efficiency", response_model=EfficiencyResponse, tags=["Predictions"])
async def predict_efficiency(
    asset_id: str = Query(..., description="Asset identifier"),
    readings: SensorBatch = None
):
    """
    Forecast efficiency trend for the next 24 hours.
    
    Requires at least 24 hours of historical readings.
    """
    timestamp = datetime.now().isoformat()
    
    # Calculate current efficiency
    if readings and len(readings.readings) > 0:
        last_reading = readings.readings[-1]
        expected_power = last_reading.irradiance * 2.0 * 0.2 * 10
        current_efficiency = last_reading.power_output / max(expected_power, 1)
    else:
        current_efficiency = 0.85
    
    # Generate forecast (demo mode or with model)
    forecast_hours = 24
    forecast_points = forecast_hours * 4  # 15-min intervals
    
    if models_loaded and efficiency_forecaster is not None and readings:
        # Use actual model
        try:
            # Prepare data from readings
            features_list = [prepare_features(r).flatten() for r in readings.readings]
            X = np.array(features_list)
            y = np.array([r.power_output / max(r.irradiance * 4, 1) for r in readings.readings])
            
            forecasted = efficiency_forecaster.predict(X, y)
            forecasted = forecasted.tolist()[:forecast_points]
        except:
            forecasted = None
    else:
        forecasted = None
    
    if forecasted is None:
        # Demo mode: generate synthetic forecast
        base = current_efficiency
        trend = np.random.choice([-0.001, 0, 0.0005])
        noise = np.random.normal(0, 0.02, forecast_points)
        forecasted = (base + trend * np.arange(forecast_points) + noise).tolist()
        forecasted = [max(0, min(1, f)) for f in forecasted]
    
    # Generate timestamps
    from datetime import timedelta
    base_time = datetime.now()
    forecast_timestamps = [
        (base_time + timedelta(minutes=15 * i)).isoformat()
        for i in range(1, len(forecasted) + 1)
    ]
    
    # Determine trend
    avg_forecast = np.mean(forecasted)
    if avg_forecast > current_efficiency + 0.02:
        trend = "improving"
    elif avg_forecast < current_efficiency - 0.02:
        trend = "declining"
    else:
        trend = "stable"
    
    efficiency_change = avg_forecast - current_efficiency
    
    # Generate alerts
    if current_efficiency < 0.7:
        alerts = check_alert_conditions(
            anomaly_score=0,
            failure_probability=0,
            efficiency=current_efficiency,
            asset_id=asset_id
        )
        alert_manager.add_alerts([a for a in alerts if a['type'] == 'efficiency'])
    
    return EfficiencyResponse(
        asset_id=asset_id,
        current_efficiency=round(current_efficiency, 4),
        forecasted_efficiency=[round(f, 4) for f in forecasted[:24]],  # Return hourly
        forecast_timestamps=forecast_timestamps[:24],
        trend=trend,
        efficiency_change=round(efficiency_change, 4),
        timestamp=timestamp
    )


@app.get("/health/{asset_id}", response_model=HealthResponse, tags=["Health"])
async def get_asset_health(
    asset_id: str,
    anomaly_score: float = Query(0.2, ge=0, le=1),
    failure_probability: float = Query(0.1, ge=0, le=1),
    performance_ratio: float = Query(0.85, ge=0, le=1)
):
    """
    Get overall health score for an asset.
    
    Health score is calculated from anomaly, failure, and efficiency metrics.
    """
    health_score = calculate_health_score(
        anomaly_score=anomaly_score,
        failure_probability=failure_probability,
        performance_ratio=performance_ratio
    )
    
    status_info = get_health_status(health_score)
    
    return HealthResponse(
        asset_id=asset_id,
        health_score=round(health_score, 1),
        status=status_info['status'],
        color=status_info['color'],
        recommendation=status_info['recommendation'],
        components={
            'anomaly_contribution': round((1 - anomaly_score) * 25, 1),
            'failure_contribution': round((1 - failure_probability) * 35, 1),
            'efficiency_contribution': round(performance_ratio * 40, 1)
        },
        timestamp=datetime.now().isoformat()
    )


@app.get("/alerts", response_model=List[Alert], tags=["Alerts"])
async def get_alerts(
    active_only: bool = Query(True, description="Return only unacknowledged alerts"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(50, ge=1, le=100)
):
    """Get current alerts."""
    if active_only:
        alerts = alert_manager.get_active_alerts()
    else:
        alerts = alert_manager.alerts
    
    if severity:
        alerts = [a for a in alerts if a['severity'] == severity]
    
    return [Alert(**a) for a in alerts[:limit]]


@app.post("/alerts/{alert_id}/acknowledge", tags=["Alerts"])
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    success = alert_manager.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert acknowledged", "alert_id": alert_id}


@app.get("/alerts/summary", tags=["Alerts"])
async def get_alert_summary():
    """Get summary of alert counts."""
    return alert_manager.get_alert_summary()


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
