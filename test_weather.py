"""
Quick test to isolate the issue
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime, timedelta
from src.data_generation.weather_generator import WeatherDataGenerator

start_date = datetime(2024, 1, 1)
end_date = start_date + timedelta(days=365)

print("Testing weather generator...")
weather_gen = WeatherDataGenerator(
    start_date=start_date,
    end_date=end_date,
    sampling_interval_minutes=15,
    random_seed=42
)

print("Calling generate()...")
weather_df = weather_gen.generate()
print(f"Success! Generated {len(weather_df)} rows")
print(weather_df.head())
