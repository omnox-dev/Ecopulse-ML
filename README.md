# EcoPulse AI: Predictive Maintenance for Renewable Infrastructure ⚡

EcoPulse AI is a state-of-the-art predictive maintenance and monitoring ecosystem designed for modern renewable energy fleets (Solar and Wind). Built for the Dubai Hackathon, it combines real-time physics-based asset simulation, advanced machine learning, and an operator-focused interactive dashboard.

---

## 🌟 Key Features

1. **Real-time Asset Simulation Engine**
   - **Wind Turbine Simulator**: Models physics-based power curves, rotor speeds, vibration anomalies, and thermal limits base on fluctuating wind speeds.
   - **Solar Array Simulator**: Generates realistic irradiance, panel-level thermal output, efficiency degradation, and voltage profiles.

2. **Machine Learning Intelligence**
   - **Failure Prediction Model**: An ensemble (Random Forest + XGBoost) predicting 7-day critical failures based on vibration profiles and thermal readings.
   - **Efficiency Forecaster**: A PyTorch LSTM/GRU sequence model forecasting performance ratio trends and efficiency drops.
   - **Zero-Day Anomaly Detection**: Isolation Forest techniques for immediate detection of sensor drift.

3. **Operator Dashboard**
   - Built with Streamlit and Plotly.
   - Streams live CSV data populated dynamically by the simulators.
   - Offers real-time alerts, physical control toggles, multi-dimensional 2D/3D visual mappings, and spatial volumetric thermal maps.

---

## 📂 Project Structure

```text
predictive-maintenance/
├── dashboard/                  # Streamlit operator portal 
│   └── app.py                  # Main entry point for GUI
├── simulators/                 # Real-time stateful physics mockers
│   ├── run_simulation.py       # Starts the background data generator
│   ├── wind.py & solar.py      # Asset logic
├── src/                        
│   ├── feature_engineering/    # Automated pipeline transformations
│   ├── models/                 # Model training and prediction architectures
├── models/                     # Compiled binaries (.joblib, .pt)
├── data/simulated/             # Auto-generated live telemetry CSVs
└── README.md                   # This file!
```

---

## 🚀 Quickstart Guide

### 1. Installation 
Ensure you have Python 3.9+ installed.

```powershell
# Clone the repository
git clone https://github.com/omnox-dev/Ecopulse-ML.git
cd Ecopulse-ML

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Asset Simulators
You need to generate real-time fleet data. Open a terminal and run the background simulation system. Keep this terminal running!

```powershell
python simulators/run_simulation.py
```
*This instantly generates 7 days of historical context and begins streaming live data points every 60 seconds.*

### 3. Launch the Operator Dashboard
In a **new terminal tab**, boot up the Streamlit interface:

```powershell
streamlit run dashboard/app.py
```
Navigate to `http://localhost:8501` in your browser. 

> **Pro-tip**: Click on the **"🔐 Special Analysis"** tab in the dashboard and enter the access code `omnox123` to unlock 3D volumetric thermal mappings and trajectory analysis.

---

## 🛠️ Technology Stack

- **Machine Learning**: `scikit-learn`, `PyTorch` (LSTM), `XGBoost`, `joblib`
- **Data Engineering**: `Pandas`, `NumPy`
- **Frontend / Dashboard**: `Streamlit`, `Plotly Express / Graph Objects`
- **Simulation**: Custom pure-Python dynamic state machines

---

## 📜 License

MIT License. Developed for the Dubai Hackathon.
