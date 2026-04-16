# Predictive Maintenance for Renewable Infrastructure

An ML-based predictive maintenance system for solar panels using sensor data and weather patterns.

## Features

- **Anomaly Detection**: Isolation Forest + Autoencoder for early fault detection
- **Failure Prediction**: Random Forest + XGBoost for 7-day failure forecasting
- **Efficiency Forecasting**: LSTM/GRU for efficiency trend prediction
- **Real-time API**: FastAPI backend for predictions
- **Interactive Dashboard**: Streamlit-based monitoring interface

## Project Structure

```
predictive-maintenance/
├── data/                       # Generated datasets
├── src/
│   ├── data_generation/        # Synthetic data generators
│   ├── feature_engineering/    # Feature extraction
│   ├── models/                 # ML models
│   ├── api/                    # FastAPI backend
│   └── utils/                  # Helper functions
├── dashboard/                  # Streamlit app
├── tests/                      # Unit tests
└── models/                     # Saved model files
```

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies

pip install -r requirements.txt
```

## Usage

### 1. Generate Synthetic Data
```bash
python src/data_generation/generate_all.py
```

### 2. Train Models
```bash
python src/models/train_all.py
```

### 3. Run API Server
```bash
uvicorn src.api.main:app --reload
```

### 4. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

## API Endpoints

- `POST /predict/anomaly` - Real-time anomaly scoring
- `POST /predict/failure` - Failure probability prediction
- `POST /predict/efficiency` - Efficiency forecast
- `GET /alerts` - Current active alerts
- `GET /health/{asset_id}` - Asset health score

## Tech Stack

- **ML**: scikit-learn, PyTorch, XGBoost
- **API**: FastAPI
- **Dashboard**: Streamlit, Plotly
- **Data**: Pandas, NumPy

## License

MIT License
