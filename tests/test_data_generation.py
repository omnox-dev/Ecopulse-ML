"""
Unit Tests for Data Generation Module
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.sensor_generator import SensorDataGenerator, generate_sensor_data
from src.data_generation.weather_generator import WeatherDataGenerator, generate_weather_data


class TestSensorDataGenerator:
    """Tests for sensor data generation."""
    
    def test_initialization(self):
        """Test generator initialization."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        gen = SensorDataGenerator(start, end, sampling_interval_minutes=15)
        
        assert len(gen.timestamps) > 0
        assert gen.timestamps[0] == pd.Timestamp(start)
    
    def test_generate_data_shape(self):
        """Test generated data has correct shape."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        gen = SensorDataGenerator(start, end, sampling_interval_minutes=15)
        df, faults = gen.generate()
        
        # 24 hours * 4 samples/hour = 96 samples + 1
        expected_samples = 24 * 4 + 1
        assert len(df) == expected_samples
        
        # Check required columns
        required_cols = [
            'timestamp', 'voltage', 'current', 'power_output',
            'panel_temperature', 'inverter_temperature', 'irradiance'
        ]
        for col in required_cols:
            assert col in df.columns
    
    def test_data_ranges(self):
        """Test generated values are in valid ranges."""
        start = datetime(2024, 6, 15)  # Summer day
        end = datetime(2024, 6, 16)
        
        gen = SensorDataGenerator(start, end, sampling_interval_minutes=15)
        df, _ = gen.generate(include_faults=False)
        
        # Voltage should be positive
        assert (df['voltage'] >= 0).all()
        
        # Current should be non-negative
        assert (df['current'] >= 0).all()
        
        # Power should be non-negative
        assert (df['power_output'] >= 0).all()
        
        # Irradiance should be non-negative
        assert (df['irradiance'] >= 0).all()
    
    def test_fault_injection(self):
        """Test fault injection works."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 1)  # 2 months
        
        gen = SensorDataGenerator(start, end, sampling_interval_minutes=15)
        df, faults = gen.generate(include_faults=True)
        
        # Should have some faults
        assert len(faults) > 0
        
        # Fault mask should indicate faults
        assert df['is_fault'].sum() > 0
    
    def test_degradation_injection(self):
        """Test degradation injection works."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)  # 1 year
        
        gen = SensorDataGenerator(start, end, sampling_interval_minutes=60)
        df, _ = gen.generate(include_degradation=True, include_faults=False)
        
        # Degradation factor should decrease over time
        first_quarter = df['degradation_factor'].iloc[:len(df)//4].mean()
        last_quarter = df['degradation_factor'].iloc[-len(df)//4:].mean()
        
        assert last_quarter < first_quarter


class TestWeatherDataGenerator:
    """Tests for weather data generation."""
    
    def test_initialization(self):
        """Test generator initialization."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        gen = WeatherDataGenerator(start, end, sampling_interval_minutes=15)
        
        assert len(gen.timestamps) > 0
    
    def test_generate_data_shape(self):
        """Test generated data has correct shape."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        gen = WeatherDataGenerator(start, end, sampling_interval_minutes=15)
        df = gen.generate()
        
        # Check required columns
        required_cols = [
            'timestamp', 'irradiance', 'ambient_temperature',
            'humidity', 'wind_speed', 'dust_index'
        ]
        for col in required_cols:
            assert col in df.columns
    
    def test_irradiance_day_night(self):
        """Test irradiance is zero at night."""
        start = datetime(2024, 6, 15)
        end = datetime(2024, 6, 16)
        
        gen = WeatherDataGenerator(start, end, sampling_interval_minutes=15)
        df = gen.generate()
        
        # Filter night hours (before 6 AM and after 7 PM)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        night_data = df[(df['hour'] < 5) | (df['hour'] > 20)]
        
        # Most night readings should be near zero
        assert night_data['irradiance'].mean() < 50
    
    def test_humidity_range(self):
        """Test humidity is within valid range."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        
        gen = WeatherDataGenerator(start, end, sampling_interval_minutes=60)
        df = gen.generate()
        
        assert (df['humidity'] >= 0).all()
        assert (df['humidity'] <= 100).all()
    
    def test_wind_speed_positive(self):
        """Test wind speed is non-negative."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 7)
        
        gen = WeatherDataGenerator(start, end, sampling_interval_minutes=30)
        df = gen.generate()
        
        assert (df['wind_speed'] >= 0).all()


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_generate_sensor_data_function(self):
        """Test generate_sensor_data convenience function."""
        df, faults = generate_sensor_data(days=7, sampling_minutes=60)
        
        assert len(df) > 0
        assert isinstance(faults, list)
    
    def test_generate_weather_data_function(self):
        """Test generate_weather_data convenience function."""
        df, events = generate_weather_data(days=7, sampling_minutes=60)
        
        assert len(df) > 0
        assert isinstance(events, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
