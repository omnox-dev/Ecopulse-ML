"""
Main script to generate all synthetic data for the predictive maintenance system.

This script generates:
1. Sensor data from solar panels
2. Weather data
3. Combined dataset with features
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.data_generation.sensor_generator import SensorDataGenerator
from src.data_generation.weather_generator import WeatherDataGenerator


def generate_all_data(
    days: int = 365,
    sampling_minutes: int = 15,
    output_dir: str = "data/raw",
    seed: int = 42
) -> dict:
    """
    Generate all synthetic data for the project.
    
    Args:
        days: Number of days of data
        sampling_minutes: Sampling interval
        output_dir: Output directory for CSV files
        seed: Random seed
        
    Returns:
        Dictionary with generated dataframes and metadata
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE DATA GENERATION")
    print("=" * 60)
    
    # Generate data ending tomorrow (to have some 'future' data for forecasting demo)
    end_date = datetime.now() + timedelta(days=1)
    start_date = end_date - timedelta(days=days)
    
    print(f"\nDate Range: {start_date.date()} to {end_date.date()}")
    print(f"Sampling Interval: {sampling_minutes} minutes")
    print(f"Expected Samples: ~{int(days * 24 * 60 / sampling_minutes):,}")
    
    # Generate weather data first
    print("\n" + "-" * 40)
    print("Generating Weather Data...")
    print("-" * 40)
    
    weather_gen = WeatherDataGenerator(
        start_date=start_date,
        end_date=end_date,
        sampling_interval_minutes=sampling_minutes,
        random_seed=seed
    )
    weather_df = weather_gen.generate()
    weather_events = weather_gen.weather_events
    
    weather_path = os.path.join(output_dir, "weather_data.csv")
    weather_df.to_csv(weather_path, index=False)
    print(f"✓ Weather data saved: {weather_path}")
    print(f"  - Samples: {len(weather_df):,}")
    print(f"  - Weather events: {len(weather_events)}")
    
    # Generate sensor data
    print("\n" + "-" * 40)
    print("Generating Sensor Data...")
    print("-" * 40)
    
    sensor_gen = SensorDataGenerator(
        start_date=start_date,
        end_date=end_date,
        sampling_interval_minutes=sampling_minutes,
        num_panels=10,
        random_seed=seed
    )
    sensor_df, fault_events = sensor_gen.generate()
    
    sensor_path = os.path.join(output_dir, "sensor_data.csv")
    sensor_df.to_csv(sensor_path, index=False)
    print(f"✓ Sensor data saved: {sensor_path}")
    print(f"  - Samples: {len(sensor_df):,}")
    print(f"  - Fault events: {len(fault_events)}")
    
    # Merge datasets
    print("\n" + "-" * 40)
    print("Creating Combined Dataset...")
    print("-" * 40)
    
    # Use weather irradiance for sensor calculations
    combined_df = sensor_df.merge(
        weather_df[['timestamp', 'humidity', 'wind_speed', 'dust_index', 'rain_index', 'cloud_cover']],
        on='timestamp',
        how='left'
    )
    
    combined_path = os.path.join(output_dir, "combined_data.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"✓ Combined data saved: {combined_path}")
    print(f"  - Total features: {len(combined_df.columns)}")
    
    # Create fault events file
    print("\n" + "-" * 40)
    print("Creating Fault Events Log...")
    print("-" * 40)
    
    faults_df = pd.DataFrame(fault_events)
    faults_path = os.path.join(output_dir, "fault_events.csv")
    faults_df.to_csv(faults_path, index=False)
    print(f"✓ Fault events saved: {faults_path}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    
    print("\nSensor Data Summary:")
    print(sensor_df[['power_output', 'voltage', 'current', 'panel_temperature']].describe().round(2))
    
    print("\nWeather Data Summary:")
    print(weather_df[['irradiance', 'ambient_temperature', 'humidity', 'wind_speed']].describe().round(2))
    
    print("\nFault Events Summary:")
    for fault in fault_events:
        print(f"  [{fault['fault_id']}] {fault['type']:20s} | "
              f"Duration: {fault['duration_hours']:3d}h | "
              f"Severity: {fault['severity']:.1%}")
    
    return {
        'sensor_df': sensor_df,
        'weather_df': weather_df,
        'combined_df': combined_df,
        'fault_events': fault_events,
        'weather_events': weather_events
    }


def create_training_labels(df: pd.DataFrame, lookahead_hours: int = 168) -> pd.DataFrame:
    """
    Create training labels for failure prediction.
    
    Args:
        df: Combined dataset with is_fault column
        lookahead_hours: Hours to look ahead for failure prediction
        
    Returns:
        DataFrame with additional label columns
    """
    df = df.copy()
    
    # Calculate samples in lookahead window
    # Assuming 15-minute intervals
    samples_per_hour = 4
    lookahead_samples = lookahead_hours * samples_per_hour
    
    # Create binary failure label (will there be a fault in next X hours?)
    df['failure_within_7d'] = 0
    
    # Get fault start indices
    fault_starts = df[df['is_fault'] == 1].index.tolist()
    
    if fault_starts:
        # For each fault start, mark preceding samples
        for idx in fault_starts:
            start_idx = max(0, idx - lookahead_samples)
            df.loc[start_idx:idx-1, 'failure_within_7d'] = 1
    
    # Create efficiency drop label (based on performance ratio trend)
    window = 24 * samples_per_hour  # 24 hour window
    df['pr_rolling_mean'] = df['performance_ratio'].rolling(window=window, min_periods=1).mean()
    df['pr_change'] = df['pr_rolling_mean'].pct_change(periods=window)
    df['efficiency_dropping'] = (df['pr_change'] < -0.05).astype(int)  # >5% drop
    
    return df


if __name__ == "__main__":
    # Generate all data
    data = generate_all_data(
        days=365,
        sampling_minutes=15,
        output_dir="data/raw"
    )
    
    # Create training labels
    print("\n" + "-" * 40)
    print("Creating Training Labels...")
    print("-" * 40)
    
    labeled_df = create_training_labels(data['combined_df'])
    labeled_path = "data/processed/labeled_data.csv"
    os.makedirs("data/processed", exist_ok=True)
    labeled_df.to_csv(labeled_path, index=False)
    print(f"✓ Labeled data saved: {labeled_path}")
    
    # Print label distribution
    print("\nLabel Distribution:")
    print(f"  - Failure within 7 days: {labeled_df['failure_within_7d'].sum():,} "
          f"({labeled_df['failure_within_7d'].mean():.2%})")
    print(f"  - Efficiency dropping: {labeled_df['efficiency_dropping'].sum():,} "
          f"({labeled_df['efficiency_dropping'].mean():.2%})")
