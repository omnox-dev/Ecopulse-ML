import sys
import time
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulators.solar import SolarPanelSimulator
from simulators.wind import WindTurbineSimulator

def main():
    print("🚀 Starting EcoPulse Asset Simulation System...")
    
    # Initialize Simulators
    solar_sim = SolarPanelSimulator("SOLAR_001")
    wind_sim = WindTurbineSimulator("WIND_001")
    
    # Ensure data directory exists (absolute path — works from any CWD)
    data_dir = PROJECT_ROOT / "data" / "simulated"
    data_dir.mkdir(parents=True, exist_ok=True)

    solar_csv = data_dir / "solar_live.csv"
    wind_csv  = data_dir / "wind_live.csv"

    # Check for initial data need
    if not solar_csv.exists():
        print("📊 No existing solar data. Generating 7 days history...")
        historical_end = datetime.now() - timedelta(minutes=15)
        solar_sim.state['last_timestamp'] = (historical_end - timedelta(days=7)).isoformat()
        df = solar_sim.simulate_until(historical_end)
        df.to_csv(solar_csv, index=False)
        print(f"  ✅ Generated {len(df)} historical points.")

    if not wind_csv.exists():
        print("📊 No existing wind data. Generating 7 days history...")
        historical_end = datetime.now() - timedelta(minutes=15)
        wind_sim.state['last_timestamp'] = (historical_end - timedelta(days=7)).isoformat()
        df = wind_sim.simulate_until(historical_end)
        df.to_csv(wind_csv, index=False)
        print(f"  ✅ Generated {len(df)} historical points.")

    while True:
        now = datetime.now()
        print(f"[{now.strftime('%H:%M:%S')}] Checking for simulation updates...")
        
        # Simulate until current time (15 min intervals by default)
        solar_df = solar_sim.simulate_until(now)
        wind_df = wind_sim.simulate_until(now)
        
        # Save updates
        if not solar_df.empty:
            solar_df.to_csv(solar_csv, mode='a', index=False, header=not solar_csv.exists())
            print(f"  ✅ Added {len(solar_df)} points to solar_live.csv")

        if not wind_df.empty:
            wind_df.to_csv(wind_csv, mode='a', index=False, header=not wind_csv.exists())
            print(f"  ✅ Added {len(wind_df)} points to wind_live.csv")
            
        if solar_df.empty and wind_df.empty:
            print("  💤 No new steps needed.")

        # Wait for next simulation cycle (e.g., 60 seconds)
        print("Waiting 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    main()
