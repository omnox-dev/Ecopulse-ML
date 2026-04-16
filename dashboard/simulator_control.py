"""
Simulator Remote Control Panel
Used for launching anomalies and seasonal changes dynamically into the running simulation system without CLI commands.
"""

import streamlit as st
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Simulation Remote", page_icon="🎮", layout="wide")

st.markdown("""
<style>
    .stButton>button { width: 100%; height: 60px; font-weight: bold; font-size: 16px; margin-bottom: 15px;}
</style>
""", unsafe_allow_html=True)

st.title("🎮 EcoPulse Simulation Director")
st.caption("WARNING: Changing modes here instantly injects parameters into the ongoing background simulation processes.")

def update_asset_state(asset_file: str, key: str, value: str):
    path = PROJECT_ROOT / "simulators" / asset_file
    if path.exists():
        with open(path, 'r') as f:
            state = json.load(f)
        state[key] = value
        with open(path, 'w') as f:
            json.dump(state, f, indent=4)
        st.success(f"State applied: {asset_file} -> {key} = {value}")
    else:
        st.error(f"Cannot find state file {asset_file}. Is the simulator running?")

def get_asset_state(asset_file: str, key: str, default: str):
    path = PROJECT_ROOT / "simulators" / asset_file
    if path.exists():
        with open(path, 'r') as f:
            state = json.load(f)
        return state.get(key, default)
    return default


c1, c2, c3 = st.columns(3)

# ==================== WIND ====================
with c1:
    st.header("🌪️ Wind Turbine")
    current_wind = get_asset_state("state_WIND_001.json", "anomaly_mode", "normal")
    st.info(f"Current Status: **{current_wind.upper()}**")
    
    st.subheader("Event Dispatcher")
    if st.button("🟢 Restore Normal Operations", key="w_norm"):
        update_asset_state("state_WIND_001.json", "anomaly_mode", "normal")
    if st.button("🚨 Inject Mechanical Failure", help="Triggers intense vibration and thermal runaway. Designed to trip Machine Learning alert thresholds.", key="w_fail"):
        update_asset_state("state_WIND_001.json", "anomaly_mode", "gearbox_fault")
    if st.button("🛑 Force Grid Curtailment", help="Simulates grid rejection, limiting rotational power by 80%.", key="w_curt"):
        update_asset_state("state_WIND_001.json", "anomaly_mode", "curtailment")
    if st.button("⚠️ Detach Telemetry Sensors", help="Simulates hardware sensor ice-over or failure.", key="w_sens"):
        update_asset_state("state_WIND_001.json", "anomaly_mode", "sensor_drift")

# ==================== SOLAR ====================
with c2:
    st.header("☀️ Solar Array")
    current_solar = get_asset_state("state_SOLAR_001.json", "anomaly_mode", "normal")
    st.info(f"Current Status: **{current_solar.upper()}**")
    
    st.subheader("Event Dispatcher")
    if st.button("🟢 Restore Normal Operations", key="s_norm"):
        update_asset_state("state_SOLAR_001.json", "anomaly_mode", "normal")
    if st.button("🔥 Inverter Overheat Trip", help="Forces thermal inversion and massive efficiency loss.", key="s_over"):
        update_asset_state("state_SOLAR_001.json", "anomaly_mode", "inverter_overheat")
    if st.button("🌪️ Generate Severe Dust Storm", help="Physical obstruction of panels resulting in 50% loss.", key="s_soil"):
        update_asset_state("state_SOLAR_001.json", "anomaly_mode", "soiling")
    if st.button("🔌 Physical Substation Disconnect", help="Cuts line completely, plunging output to 0.0MW.", key="s_off"):
        update_asset_state("state_SOLAR_001.json", "anomaly_mode", "offline")

# ==================== SEASON ====================
with c3:
    st.header("🌍 Environmental Season")
    current_season = get_asset_state("state_WIND_001.json", "season_mode", "spring")
    st.info(f"Current Climate: **{current_season.upper()}**")
    
    st.subheader("Global Climate Override")
    if st.button("🌸 Spring (Mild ~25°C)", key="c_spr"):
        update_asset_state("state_WIND_001.json", "season_mode", "spring")
        update_asset_state("state_SOLAR_001.json", "season_mode", "spring")
    if st.button("🌞 Summer Heatwave (+15°C)", key="c_sum"):
        update_asset_state("state_WIND_001.json", "season_mode", "summer")
        update_asset_state("state_SOLAR_001.json", "season_mode", "summer")
    if st.button("⛄ Winter Freeze (-10°C)", key="c_win"):
        update_asset_state("state_WIND_001.json", "season_mode", "winter")
        update_asset_state("state_SOLAR_001.json", "season_mode", "winter")
