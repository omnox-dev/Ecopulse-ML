"""
Weather Data Generator for Predictive Maintenance

Generates realistic synthetic weather data including:
- Solar Irradiance (W/m²)
- Ambient Temperature (°C)
- Humidity (%)
- Wind Speed (m/s)
- Dust/Rain Index

Features:
- Seasonal patterns
- Daily cycles
- Weather events (cloudy days, rain, dust storms)
- Correlation with sensor data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List
import random


class WeatherDataGenerator:
    """Generate synthetic weather data with realistic patterns."""
    
    # Climate parameters (Dubai-like desert climate)
    CLIMATE = {
        'base_temp': 28,              # Average annual temperature °C
        'temp_amplitude': 12,          # Seasonal temperature variation °C
        'daily_temp_range': 10,        # Daily temperature variation °C
        'base_humidity': 55,           # Average humidity %
        'max_irradiance': 1050,        # Peak solar irradiance W/m²
        'avg_wind_speed': 4.5,         # Average wind speed m/s
        'dust_probability': 0.05,      # Daily probability of dust event
        'rain_probability': 0.02,      # Daily probability of rain
    }
    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        sampling_interval_minutes: int = 15,
        location_lat: float = 25.2,    # Dubai latitude
        random_seed: Optional[int] = 42
    ):
        """
        Initialize the weather data generator.
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
            sampling_interval_minutes: Time between readings
            location_lat: Latitude for solar calculations
            random_seed: Random seed for reproducibility
        """
        self.start_date = start_date
        self.end_date = end_date
        self.sampling_interval = sampling_interval_minutes
        self.location_lat = location_lat
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Generate timestamps
        self.timestamps = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=f'{self.sampling_interval}min'
        )
        self.n_samples = len(self.timestamps)
        
        # Weather event tracking
        self.weather_events = []
        
    def _get_day_of_year_factor(self) -> np.ndarray:
        """Get seasonal factor based on day of year (0-1 sinusoidal)."""
        day_of_year = np.array(self.timestamps.dayofyear)
        # Peak in summer (around day 172), minimum in winter
        return np.sin(np.radians((day_of_year - 80) * 360 / 365))
    
    def _get_hour_factor(self) -> np.ndarray:
        """Get daily factor based on hour of day (0-1)."""
        hour = np.array(self.timestamps.hour) + np.array(self.timestamps.minute) / 60
        # Peak at 14:00, minimum at 6:00
        return np.sin(np.radians((hour - 6) * 180 / 12))
    
    def generate_irradiance(self) -> np.ndarray:
        """
        Generate solar irradiance with realistic patterns.
        
        Returns:
            Array of irradiance values in W/m²
        """
        hour = np.array(self.timestamps.hour) + np.array(self.timestamps.minute) / 60
        day_of_year = np.array(self.timestamps.dayofyear)
        
        # Solar elevation calculation
        declination = 23.45 * np.sin(np.radians((360/365) * (day_of_year - 81)))
        hour_angle = 15 * (hour - 12)
        
        sin_elevation = (
            np.sin(np.radians(self.location_lat)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(self.location_lat)) * np.cos(np.radians(declination)) *
            np.cos(np.radians(hour_angle))
        )
        
        elevation = np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))
        
        # Clear sky irradiance
        air_mass = np.where(
            elevation > 0,
            1 / (np.sin(np.radians(elevation)) + 0.00176759 * 
                 np.power(elevation + 1.00128, -1.48413)),
            0
        )
        
        # Direct normal irradiance (simplified model)
        clear_sky = np.where(
            elevation > 0,
            self.CLIMATE['max_irradiance'] * np.exp(-0.14 * air_mass) * 
            np.sin(np.radians(elevation)),
            0
        )
        
        # Add atmospheric variability
        cloud_noise = 0.85 + 0.15 * np.random.random(self.n_samples)
        irradiance = clear_sky * cloud_noise
        
        # Seasonal adjustment
        seasonal = 0.9 + 0.1 * self._get_day_of_year_factor()
        irradiance *= seasonal
        
        return np.maximum(0, irradiance)
    
    def generate_temperature(self, irradiance: np.ndarray) -> np.ndarray:
        """
        Generate ambient temperature correlated with irradiance.
        
        Args:
            irradiance: Solar irradiance array
            
        Returns:
            Array of temperature values in °C
        """
        climate = self.CLIMATE
        
        # Seasonal base temperature
        seasonal = climate['base_temp'] + climate['temp_amplitude'] * self._get_day_of_year_factor()
        
        # Daily variation (peak around 14:00)
        hour = np.array(self.timestamps.hour) + np.array(self.timestamps.minute) / 60
        daily = climate['daily_temp_range'] * np.sin(np.radians((hour - 8) * 180 / 14))
        daily = np.clip(daily, -climate['daily_temp_range']/2, climate['daily_temp_range'])
        
        # Irradiance correlation (higher irradiance = higher temp)
        irradiance_effect = (irradiance / self.CLIMATE['max_irradiance']) * 5
        
        # Random variation
        noise = np.random.normal(0, 1.5, self.n_samples)
        
        temperature = seasonal + daily + irradiance_effect + noise
        
        return temperature
    
    def generate_humidity(self, temperature: np.ndarray) -> np.ndarray:
        """
        Generate humidity inversely correlated with temperature.
        
        Args:
            temperature: Temperature array
            
        Returns:
            Array of humidity values in %
        """
        climate = self.CLIMATE
        
        # Base humidity (higher in morning/evening, lower midday)
        hour = np.array(self.timestamps.hour) + np.array(self.timestamps.minute) / 60
        daily_pattern = climate['base_humidity'] + 15 * np.cos(np.radians((hour - 6) * 15))
        
        # Inverse correlation with temperature
        temp_effect = -0.8 * (temperature - climate['base_temp'])
        
        # Seasonal (more humid in summer due to Arabian Gulf)
        seasonal = 10 * self._get_day_of_year_factor()
        
        # Random variation
        noise = np.random.normal(0, 5, self.n_samples)
        
        humidity = daily_pattern + temp_effect + seasonal + noise
        humidity = np.clip(humidity, 15, 100)
        
        return humidity
    
    def generate_wind_speed(self) -> np.ndarray:
        """
        Generate wind speed with daily and seasonal patterns.
        
        Returns:
            Array of wind speed values in m/s
        """
        climate = self.CLIMATE
        
        # Base wind speed increases in afternoon
        hour = np.array(self.timestamps.hour) + np.array(self.timestamps.minute) / 60
        daily_pattern = climate['avg_wind_speed'] * (0.7 + 0.5 * 
            np.sin(np.radians((hour - 10) * 180 / 8)))
        daily_pattern = np.maximum(1, daily_pattern)
        
        # Seasonal variation (windier in summer)
        seasonal = 1 + 0.3 * self._get_day_of_year_factor()
        
        # Add gusts (occasional spikes)
        gust_mask = np.random.random(self.n_samples) < 0.05
        gusts = np.where(gust_mask, np.random.uniform(1.5, 3, self.n_samples), 1)
        
        # Random variation (Weibull-like distribution)
        noise = np.random.weibull(2, self.n_samples) * 0.5
        
        wind_speed = daily_pattern * seasonal * gusts + noise
        wind_speed = np.clip(wind_speed, 0, 25)  # Cap at 25 m/s
        
        return wind_speed
    
    def generate_dust_index(self, wind_speed: np.ndarray) -> np.ndarray:
        """
        Generate dust index (0-100) based on wind and events.
        
        Args:
            wind_speed: Wind speed array
            
        Returns:
            Array of dust index values (0-100)
        """
        # Ensure wind_speed is a numpy array (not pandas Index/Series)
        wind_speed = np.asarray(wind_speed, dtype=float)
        
        # Base dust level (correlated with wind)
        base_dust = np.minimum(30.0, wind_speed * 3.0).astype(float)
        
        # Add random dust events
        n_dust_events = int(self.CLIMATE['dust_probability'] * (self.n_samples / 96))  # 96 = daily samples
        
        # Start with base dust as a list for easier modification
        dust_list = list(base_dust)
        
        for i in range(n_dust_events):
            # Random event start
            start_idx = np.random.randint(0, self.n_samples)
            
            # Duration: 4-24 hours
            duration = np.random.randint(4, 24) * (60 // self.sampling_interval)
            end_idx = min(start_idx + duration, self.n_samples)
            
            # Intensity
            intensity = np.random.uniform(50, 100)
            
            # Apply dust event (gradual onset and decay)
            event_length = end_idx - start_idx
            if event_length > 0:
                envelope = np.sin(np.linspace(0, np.pi, event_length))
                # Apply modification element by element
                for j in range(event_length):
                    dust_list[start_idx + j] += intensity * envelope[j]
                
                self.weather_events.append({
                    'type': 'dust_storm',
                    'start_time': self.timestamps[start_idx],
                    'end_time': self.timestamps[end_idx - 1],
                    'intensity': intensity
                })
        
        # Convert back to numpy array
        dust_index = np.array(dust_list, dtype=float)
        
        # Add noise
        noise = np.random.normal(0, 5, self.n_samples)
        dust_index = dust_index + noise
        dust_index = np.clip(dust_index, 0, 100)
        
        return dust_index
    
    def generate_rain_index(self) -> np.ndarray:
        """
        Generate rain index (0-100) with rare rain events.
        
        Returns:
            Array of rain index values (0-100)
        """
        rain_index = np.zeros(self.n_samples)
        
        # Rain is rare in desert climate
        n_rain_events = int(self.CLIMATE['rain_probability'] * (self.n_samples / 96))
        samples_per_hour = 60 // self.sampling_interval
        
        for _ in range(n_rain_events):
            # Random event start
            start_idx = np.random.randint(0, self.n_samples)
            
            # Duration: 1-6 hours
            duration = np.random.randint(1, 6) * samples_per_hour
            end_idx = min(start_idx + duration, self.n_samples)
            
            # Intensity
            intensity = np.random.uniform(20, 80)
            
            # Apply rain event
            event_length = end_idx - start_idx
            if event_length > 0:
                rain_index[start_idx:end_idx] = intensity
                
                self.weather_events.append({
                    'type': 'rain',
                    'start_time': self.timestamps[start_idx],
                    'end_time': self.timestamps[end_idx - 1],
                    'intensity': intensity
                })
        
        return rain_index
    
    def generate_cloud_cover(self, irradiance: np.ndarray) -> np.ndarray:
        """
        Generate cloud cover percentage based on irradiance reduction.
        
        Args:
            irradiance: Solar irradiance array
            
        Returns:
            Array of cloud cover values (0-100%)
        """
        # Get clear sky irradiance for comparison
        hour = np.array(self.timestamps.hour) + np.array(self.timestamps.minute) / 60
        day_of_year = np.array(self.timestamps.dayofyear)
        
        # Simplified clear sky reference
        clear_sky = self.CLIMATE['max_irradiance'] * np.sin(
            np.radians(np.clip((hour - 6) * 15, 0, 180))
        )
        clear_sky = np.maximum(1, clear_sky)  # Avoid division by zero
        
        # Cloud cover estimated from irradiance reduction
        cloud_cover = (1 - irradiance / clear_sky) * 100
        cloud_cover = np.clip(cloud_cover, 0, 100)
        
        # At night, set to 0 (not measurable)
        cloud_cover = np.where(irradiance < 10, 0, cloud_cover)
        
        return cloud_cover
    
    def apply_weather_effects(
        self,
        irradiance: np.ndarray,
        dust_index: np.ndarray,
        rain_index: np.ndarray
    ) -> np.ndarray:
        """
        Apply weather effects to reduce irradiance.
        
        Args:
            irradiance: Base irradiance
            dust_index: Dust index
            rain_index: Rain index
            
        Returns:
            Modified irradiance
        """
        # Dust reduces irradiance
        dust_effect = 1 - (dust_index / 100) * 0.5
        
        # Rain/clouds reduce irradiance
        rain_effect = 1 - (rain_index / 100) * 0.7
        
        modified = irradiance * dust_effect * rain_effect
        return np.maximum(0, modified)
    
    def generate(self) -> pd.DataFrame:
        """
        Generate complete weather dataset.
        
        Returns:
            DataFrame with all weather measurements
        """
        # Generate base measurements
        irradiance = self.generate_irradiance()
        wind_speed = np.array(self.generate_wind_speed())
        dust_index = self.generate_dust_index(wind_speed)
        rain_index = self.generate_rain_index()
        
        # Apply weather effects to irradiance
        irradiance = self.apply_weather_effects(irradiance, dust_index, rain_index)
        
        # Generate temperature and humidity
        temperature = self.generate_temperature(irradiance)
        humidity = self.generate_humidity(temperature)
        
        # Generate cloud cover
        cloud_cover = self.generate_cloud_cover(irradiance)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'irradiance': irradiance.round(1),
            'ambient_temperature': temperature.round(1),
            'humidity': humidity.round(1),
            'wind_speed': wind_speed.round(2),
            'dust_index': dust_index.round(1),
            'rain_index': rain_index.round(1),
            'cloud_cover': cloud_cover.round(1)
        })
        
        return df


def generate_weather_data(
    days: int = 365,
    sampling_minutes: int = 15,
    output_path: Optional[str] = None
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Convenience function to generate weather data.
    
    Args:
        days: Number of days of data to generate
        sampling_minutes: Sampling interval in minutes
        output_path: Optional path to save CSV
        
    Returns:
        Tuple of (weather_dataframe, weather_events)
    """
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=days)
    
    generator = WeatherDataGenerator(
        start_date=start_date,
        end_date=end_date,
        sampling_interval_minutes=sampling_minutes,
        random_seed=42
    )
    
    df = generator.generate()
    events = generator.weather_events
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved weather data to {output_path}")
        print(f"Generated {len(df)} samples over {days} days")
        print(f"Weather events: {len(events)}")
    
    return df, events


if __name__ == "__main__":
    # Generate sample data
    df, events = generate_weather_data(
        days=365,
        sampling_minutes=15,
        output_path="data/raw/weather_data.csv"
    )
    
    print("\nDataset Summary:")
    print(df.describe())
    
    print("\nWeather Events:")
    for event in events[:10]:
        print(f"  - {event['type']}: {event['start_time']} (intensity: {event['intensity']:.1f})")
