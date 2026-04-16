import sys
import time
import os
import argparse
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
    parser = argparse.ArgumentParser(description="Run EcoPulse Asset Simulation")
    parser.add_argument("--solar-mode", type=str, default="normal", help="Solar mode: normal, inverter_overheat, soiling, offline")
    parser.add_argument("--wind-mode", type=str, default="normal", help="Wind mode: normal, gearbox_fault, sensor_drift, curtailment")
    parser.add_argument("--interval", type=int, default=15, help="Simulation time step in minutes (default 15)")
    args = parser.parse_args()

    print(f"🚀 Starting EcoPulse Asset Simulation System...")
    print(f"   - Solar Mode: {args.solar_mode}")
    print(f"   - Wind Mode:  {args.wind_mode}")
    print(f"   - Temp Step:  {args.interval} mins")
    
    # Initialize Simulators
    solar_sim = SolarPanelSimulator("SOLAR_001")
    wind_sim = WindTurbineSimulator("WIND_001")
    
    # Apply modes
    solar_sim.set_anomaly_mode(args.solar_mode)
    wind_sim.set_anomaly_mode(args.wind_mode)
    
    interval = args.interval
    
    # Ensure data directory exists (absolute path — works from any CWD)
    data_dir = PROJECT_ROOT / "data" / "simulated"
    data_dir.mkdir(parents=True, exist_ok=True)

    solar_csv = data_dir / "solar_live.csv"
    wind_csv  = data_dir / "wind_live.csv"

    # Check for initial data need
    if not solar_csv.exists():
        print("📊 No existing solar data. Generating 7 days history...")
        historical_end = datetime.now() - timedelta(minutes=interval)
        solar_sim.state['last_timestamp'] = (historical_end - timedelta(days=7)).isoformat()
        df = solar_sim.simulate_until(historical_end, interval_minutes=interval)
        df.to_csv(solar_csv, index=False)
        print(f"  ✅ Generated {len(df)} historical points.")

    if not wind_csv.exists():
        print("📊 No existing wind data. Generating 7 days history...")
        historical_end = datetime.now() - timedelta(minutes=interval)
        wind_sim.state['last_timestamp'] = (historical_end - timedelta(days=7)).isoformat()
        df = wind_sim.simulate_until(historical_end, interval_minutes=interval)
        df.to_csv(wind_csv, index=False)
        print(f"  ✅ Generated {len(df)} historical points.")

    while True:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Advancing simulation by {interval} minutes...")
            
            # Fast-forward mode: Generate exactly 1 new step continuously
            next_time_solar = datetime.fromisoformat(solar_sim.state['last_timestamp']) + timedelta(minutes=interval)
            next_time_wind = datetime.fromisoformat(wind_sim.state['last_timestamp']) + timedelta(minutes=interval)
            
            solar_df = solar_sim.simulate_until(next_time_solar, interval_minutes=interval)
            wind_df = wind_sim.simulate_until(next_time_wind, interval_minutes=interval)
        
            # Save updates
            if not solar_df.empty:
                solar_df.to_csv(solar_csv, mode='a', index=False, header=not solar_csv.exists())
                print(f"  ✅ Added {len(solar_df)} points to solar_live.csv")

            if not wind_df.empty:
                wind_df.to_csv(wind_csv, mode='a', index=False, header=not wind_csv.exists())
                print(f"  ✅ Added {len(wind_df)} points to wind_live.csv")
            
            # Sleep briefly so the dashboard updates look 'live' but fast-paced
            time.sleep(3)
            
        except KeyboardInterrupt:
            print("\n🛑 Simulation stopped by user.")
            sys.exit(0)
        except Exception as e:
            print(f"⚠️ Error during simulation tick: {e}")
            time.sleep(3)

if __name__ == "__main__":
    main()
