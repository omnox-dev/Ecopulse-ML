"""Test import"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("About to import...")
from src.data_generation.weather_generator import WeatherDataGenerator
print("Import successful!")
