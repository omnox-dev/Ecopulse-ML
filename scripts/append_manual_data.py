import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Define paths
DATA_DIR = r"c:\Users\Om\Documents\Dubai Hkthon\predictive-maintenance\data\raw"
COMBINED_FILE = os.path.join(DATA_DIR, "combined_data.csv")

def append_data():
    print(f"Reading existing data from {COMBINED_FILE}...")
    try:
        df = pd.read_csv(COMBINED_FILE)
        last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
    except (FileNotFoundError, IndexError):
        # Fallback if file doesn't exist or is empty
        last_timestamp = datetime(2026, 2, 8, 10, 0, 0)

    # User wants to remove Feb 9-10 and keep only Feb 8th data till 10 PM.
    # First, let's define the new range
    start_time = datetime(2026, 2, 8, 0, 0, 0)
    end_time = datetime(2026, 2, 8, 22, 0, 0)
    
    timestamps = pd.date_range(start=start_time, end=end_time, freq='15min')
    n = len(timestamps)
    
    print(f"Generating {n} new data points for Feb 8th ({timestamps[0]} to {timestamps[-1]})...")

    # Generate synthetic data
    # Basic sinusoidal irradiance pattern for mid-day
    # Peak at noon (12:00)
    hour_float = timestamps.hour + timestamps.minute / 60.0
    
    # Vectorized generation is fine, same logic as before
    irradiance = np.maximum(0, 1000 * np.sin(np.pi * (hour_float - 6) / 12)) 
    
    # Add noise
    irradiance += np.random.normal(0, 50, n)
    irradiance = np.maximum(0, irradiance)

    # Temperature
    ambient_temp = 30 + 5 * np.sin(np.pi * (hour_float - 9) / 12) + np.random.normal(0, 1, n)
    panel_temp = ambient_temp + (irradiance / 800) * 25 + np.random.normal(0, 2, n)
    inverter_temp = ambient_temp + (irradiance / 800) * 15 + np.random.normal(0, 2, n)

    # Electrical
    # P = I * V
    voltage = 40 * (1 - 0.004 * (panel_temp - 25)) + np.random.normal(0, 0.5, n)
    # Power proportional to irradiance
    efficiency = 0.20 # 20%
    area = 1.6 # m2
    num_panels = 10
    
    # Expected power calculation
    expected_power = irradiance * area * efficiency * num_panels
    
    # Actual power with some random efficiency loss
    power_output = expected_power * (0.95 + 0.05 * np.random.random(n))
    
    # Current
    current = power_output / voltage
    
    # Performance Ratio
    pr = power_output / (expected_power + 1e-6)
    
    # Create DataFrame
    new_data = pd.DataFrame({
        'timestamp': timestamps,
        'voltage': voltage.round(2),
        'current': current.round(2),
        'power_output': power_output.round(2),
        'expected_power': expected_power.round(2),
        'performance_ratio': pr.round(4),
        'panel_temperature': panel_temp.round(1),
        'inverter_temperature': inverter_temp.round(1),
        'ambient_temperature': ambient_temp.round(1),
        'irradiance': irradiance.round(0),
        'degradation_factor': 1.0,
        'is_fault': 0,
        'fault_id': 0,
        'humidity': np.random.randint(30, 60, n),
        'wind_speed': np.random.uniform(2, 8, n).round(1),
        'dust_index': np.random.randint(10, 50, n),
        'rain_index': 0.0,
        'cloud_cover': np.random.randint(0, 20, n)
    })
    
    # Combine and save
    if os.path.exists(COMBINED_FILE):
        # Read existing
        existing_df = pd.read_csv(COMBINED_FILE)
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
        
        # Clean up: Remove any data after Feb 8th 00:00 (this removes the 9th-10th we added)
        # We will then append our new Feb 8th 00:00 - 22:00 chunk.
        
        cutoff_clean = datetime(2026, 2, 8, 0, 0, 0)
        print(f"Removing any data after {cutoff_clean} to clean up...")
        existing_df = existing_df[existing_df['timestamp'] < cutoff_clean]
        
        # Align columns
        for col in existing_df.columns:
            if col not in new_data.columns:
                new_data[col] = 0 
        new_data = new_data[existing_df.columns]
        
        # Append
        updated_df = pd.concat([existing_df, new_data], ignore_index=True)
        updated_df['timestamp'] = pd.to_datetime(updated_df['timestamp'])
        updated_df.sort_values('timestamp', inplace=True)
        
        updated_df.to_csv(COMBINED_FILE, index=False)
        print(f"Data updated. Kept valid history, removed future data, added Feb 8th (00:00-22:00). Total rows: {len(updated_df)}")

    else:
        new_data.to_csv(COMBINED_FILE, index=False)
        print(f"Created new file with {len(new_data)} rows.")

if __name__ == "__main__":
    append_data()
