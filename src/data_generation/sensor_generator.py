"""
Sensor Data Generator for Solar Panel Monitoring

Generates realistic synthetic sensor data including:
- Voltage (V)
- Current (A)
- Power Output (W)
- Panel Temperature (°C)
- Inverter Temperature (°C)

Includes injection of:
- Gradual efficiency degradation
- Sudden faults
- Weather-induced variations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List
import random


class SensorDataGenerator:
    """Generate synthetic solar panel sensor data with realistic patterns."""
    
    # Solar panel specifications (typical residential panel)
    PANEL_SPECS = {
        'rated_power': 400,          # Watts
        'rated_voltage': 40,          # Volts (Vmp)
        'rated_current': 10,          # Amps (Imp)
        'area': 2.0,                  # m²
        'efficiency': 0.20,           # 20% efficiency
        'temp_coefficient': -0.004,   # Power reduction per °C above 25°C
    }
    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        sampling_interval_minutes: int = 15,
        num_panels: int = 1,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize the sensor data generator.
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
            sampling_interval_minutes: Time between readings (5-15 mins typical)
            num_panels: Number of solar panels to simulate
            random_seed: Random seed for reproducibility
        """
        self.start_date = start_date
        self.end_date = end_date
        self.sampling_interval = sampling_interval_minutes
        self.num_panels = num_panels
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Generate timestamps
        self.timestamps = self._generate_timestamps()
        self.n_samples = len(self.timestamps)
        
        # Track degradation and fault states
        self.degradation_factor = np.ones(self.n_samples)
        self.fault_mask = np.zeros(self.n_samples, dtype=bool)
        self.fault_labels = np.zeros(self.n_samples, dtype=int)
        
    def _generate_timestamps(self) -> pd.DatetimeIndex:
        """Generate timestamps for the simulation period."""
        return pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=f'{self.sampling_interval}min'
        )
    
    def _get_solar_position(self, timestamps: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate approximate solar elevation and azimuth.
        Simplified model assuming location at ~25°N latitude (Dubai region).
        """
        # Day of year
        day_of_year = np.array(timestamps.dayofyear)
        
        # Hour of day (decimal)
        hour = np.array(timestamps.hour) + np.array(timestamps.minute) / 60
        
        # Solar declination (simplified)
        declination = 23.45 * np.sin(np.radians((360/365) * (day_of_year - 81)))
        
        # Hour angle
        hour_angle = 15 * (hour - 12)
        
        # Latitude (Dubai area)
        latitude = 25.0
        
        # Solar elevation
        sin_elevation = (
            np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * 
            np.cos(np.radians(hour_angle))
        )
        elevation = np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))
        
        # Solar azimuth (simplified)
        azimuth = hour_angle  # Simplified
        
        return elevation, azimuth
    
    def _generate_base_irradiance(self, weather_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate base solar irradiance based on time of day and season.
        
        If weather_data is provided, use it; otherwise generate synthetic.
        """
        elevation, _ = self._get_solar_position(self.timestamps)
        
        # Base irradiance (W/m²) based on solar elevation
        # Maximum ~1000 W/m² at solar noon on clear day
        irradiance = np.maximum(0, np.sin(np.radians(elevation)) * 1000)
        
        # Seasonal variation (summer has higher peak)
        day_of_year = np.array(self.timestamps.dayofyear)
        seasonal_factor = 0.85 + 0.15 * np.sin(np.radians((day_of_year - 80) * 360 / 365))
        
        irradiance *= seasonal_factor
        
        # Add some random cloud variations
        cloud_factor = 0.7 + 0.3 * np.random.random(self.n_samples)
        irradiance *= cloud_factor
        
        return irradiance
    
    def _generate_temperatures(self, irradiance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate panel and inverter temperatures based on irradiance and ambient.
        
        Returns:
            Tuple of (panel_temp, inverter_temp)
        """
        # Ambient temperature (°C) - varies with time of day and season
        hour = np.array(self.timestamps.hour) + np.array(self.timestamps.minute) / 60
        day_of_year = np.array(self.timestamps.dayofyear)
        
        # Base ambient: 15-45°C range for Dubai-like climate
        base_temp = 25 + 10 * np.sin(np.radians((day_of_year - 80) * 360 / 365))
        daily_variation = 8 * np.sin(np.radians((hour - 6) * 15))
        ambient_temp = base_temp + daily_variation + np.random.normal(0, 2, self.n_samples)
        
        # Panel temperature: ambient + heating from irradiance
        # Approximately 30°C above ambient at full irradiance
        panel_temp = ambient_temp + (irradiance / 1000) * 30 + np.random.normal(0, 2, self.n_samples)
        
        # Inverter temperature: ambient + load-based heating
        # Increases with power output
        power_factor = irradiance / 1000
        inverter_temp = ambient_temp + 15 + power_factor * 20 + np.random.normal(0, 3, self.n_samples)
        
        return ambient_temp, panel_temp, inverter_temp
    
    def _calculate_power_output(
        self,
        irradiance: np.ndarray,
        panel_temp: np.ndarray,
        ambient_temp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate electrical output based on irradiance and temperature.
        
        Returns:
            Tuple of (voltage, current, power)
        """
        specs = self.PANEL_SPECS
        
        # Temperature derating
        temp_diff = panel_temp - 25  # STC is 25°C
        temp_factor = 1 + specs['temp_coefficient'] * temp_diff
        temp_factor = np.clip(temp_factor, 0.5, 1.1)
        
        # Expected power based on irradiance
        irradiance_factor = irradiance / 1000  # Normalized to STC (1000 W/m²)
        
        # Base power output
        power = (
            specs['rated_power'] * 
            irradiance_factor * 
            temp_factor * 
            self.degradation_factor *
            self.num_panels
        )
        
        # Add realistic noise
        power = power * (1 + np.random.normal(0, 0.02, self.n_samples))
        power = np.maximum(0, power)
        
        # Calculate voltage and current
        # Voltage varies with temperature (decreases with heat)
        voltage = specs['rated_voltage'] * (1 - 0.003 * temp_diff)
        voltage = voltage * (0.8 + 0.2 * irradiance_factor)  # Lower at low light
        voltage = voltage * (1 + np.random.normal(0, 0.01, self.n_samples))
        voltage = np.maximum(0, voltage)
        
        # Current from P = V * I
        current = np.divide(power, voltage, where=voltage > 0, out=np.zeros_like(power))
        current = np.maximum(0, current)
        
        return voltage, current, power
    
    def inject_gradual_degradation(
        self,
        degradation_rate: float = 0.0005,
        start_after_days: int = 30
    ) -> None:
        """
        Inject gradual efficiency degradation over time.
        
        Args:
            degradation_rate: Daily degradation rate (0.0005 = 0.05% per day)
            start_after_days: Days before degradation starts
        """
        # Calculate days from start
        time_diffs = (self.timestamps - self.timestamps[0])
        days_elapsed = np.array(time_diffs.days)
        
        # Apply degradation after initial period
        degradation_days = np.maximum(0, days_elapsed - start_after_days)
        self.degradation_factor = 1 - (degradation_rate * degradation_days)
        self.degradation_factor = np.maximum(0.5, self.degradation_factor)
        
    def inject_sudden_faults(
        self,
        num_faults: int = 5,
        fault_duration_hours: Tuple[int, int] = (24, 168),
        severity_range: Tuple[float, float] = (0.3, 0.8)
    ) -> List[Dict]:
        """
        Inject sudden fault events.
        
        Args:
            num_faults: Number of fault events to inject
            fault_duration_hours: (min, max) duration of faults in hours
            severity_range: (min, max) power reduction during fault
            
        Returns:
            List of fault event dictionaries
        """
        fault_events = []
        samples_per_hour = 60 // self.sampling_interval
        
        for fault_id in range(1, num_faults + 1):
            # Random fault start (avoid first and last 10% of data)
            start_idx = int(np.random.uniform(0.1, 0.8) * self.n_samples)
            
            # Random duration
            duration_hours = np.random.randint(fault_duration_hours[0], fault_duration_hours[1])
            duration_samples = duration_hours * samples_per_hour
            end_idx = min(start_idx + duration_samples, self.n_samples)
            
            # Random severity
            severity = np.random.uniform(severity_range[0], severity_range[1])
            
            # Apply fault
            self.degradation_factor[start_idx:end_idx] = self.degradation_factor[start_idx:end_idx] * severity
            self.fault_mask[start_idx:end_idx] = True
            self.fault_labels[start_idx:end_idx] = fault_id
            
            fault_events.append({
                'fault_id': fault_id,
                'start_time': self.timestamps[start_idx],
                'end_time': self.timestamps[end_idx - 1],
                'duration_hours': duration_hours,
                'severity': 1 - severity,  # Higher number = more severe
                'type': np.random.choice(['inverter_failure', 'panel_damage', 'connection_issue', 'overheating'])
            })
            
        return fault_events
    
    def inject_weather_anomalies(
        self,
        irradiance: np.ndarray,
        num_events: int = 10
    ) -> np.ndarray:
        """
        Inject weather-related anomalies (dust storms, heavy clouds, etc.).
        
        Args:
            irradiance: Base irradiance array
            num_events: Number of weather events to inject
            
        Returns:
            Modified irradiance array
        """
        modified_irradiance = irradiance.copy()
        samples_per_hour = 60 // self.sampling_interval
        
        for _ in range(num_events):
            # Random event start
            start_idx = int(np.random.uniform(0.05, 0.95) * self.n_samples)
            
            # Duration: 2-48 hours
            duration_hours = np.random.randint(2, 48)
            duration_samples = duration_hours * samples_per_hour
            end_idx = min(start_idx + duration_samples, self.n_samples)
            
            # Reduction factor (dust storm or heavy clouds)
            reduction = np.random.uniform(0.2, 0.6)
            
            modified_irradiance[start_idx:end_idx] = modified_irradiance[start_idx:end_idx] * reduction
            
        return modified_irradiance
    
    def generate(
        self,
        weather_data: Optional[pd.DataFrame] = None,
        include_degradation: bool = True,
        include_faults: bool = True,
        include_weather_anomalies: bool = True
    ) -> pd.DataFrame:
        """
        Generate complete sensor dataset.
        
        Args:
            weather_data: Optional external weather data
            include_degradation: Whether to include gradual degradation
            include_faults: Whether to include sudden faults
            include_weather_anomalies: Whether to include weather anomalies
            
        Returns:
            DataFrame with all sensor readings
        """
        # Generate base irradiance
        irradiance = self._generate_base_irradiance(weather_data)
        
        # Inject weather anomalies
        if include_weather_anomalies:
            irradiance = self.inject_weather_anomalies(irradiance)
        
        # Inject degradation
        if include_degradation:
            self.inject_gradual_degradation()
        
        # Inject faults
        fault_events = []
        if include_faults:
            fault_events = self.inject_sudden_faults()
        
        # Generate temperatures
        ambient_temp, panel_temp, inverter_temp = self._generate_temperatures(irradiance)
        
        # Calculate electrical output
        voltage, current, power = self._calculate_power_output(
            irradiance, panel_temp, ambient_temp
        )
        
        # Calculate expected power (without degradation/faults)
        expected_power = (
            self.PANEL_SPECS['rated_power'] * 
            (irradiance / 1000) * 
            self.num_panels
        )
        
        # Performance ratio
        performance_ratio = np.divide(
            power, expected_power, 
            where=expected_power > 0, 
            out=np.zeros_like(power)
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'voltage': voltage,
            'current': current,
            'power_output': power,
            'expected_power': expected_power,
            'performance_ratio': performance_ratio,
            'panel_temperature': panel_temp,
            'inverter_temperature': inverter_temp,
            'ambient_temperature': ambient_temp,
            'irradiance': irradiance,
            'degradation_factor': self.degradation_factor,
            'is_fault': self.fault_mask.astype(int),
            'fault_id': self.fault_labels
        })
        
        # Round values for realism
        df['voltage'] = df['voltage'].round(2)
        df['current'] = df['current'].round(3)
        df['power_output'] = df['power_output'].round(1)
        df['expected_power'] = df['expected_power'].round(1)
        df['performance_ratio'] = df['performance_ratio'].round(4)
        df['panel_temperature'] = df['panel_temperature'].round(1)
        df['inverter_temperature'] = df['inverter_temperature'].round(1)
        df['ambient_temperature'] = df['ambient_temperature'].round(1)
        df['irradiance'] = df['irradiance'].round(1)
        
        return df, fault_events


def generate_sensor_data(
    days: int = 365,
    sampling_minutes: int = 15,
    output_path: Optional[str] = None
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Convenience function to generate sensor data.
    
    Args:
        days: Number of days of data to generate
        sampling_minutes: Sampling interval in minutes
        output_path: Optional path to save CSV
        
    Returns:
        Tuple of (sensor_dataframe, fault_events)
    """
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=days)
    
    generator = SensorDataGenerator(
        start_date=start_date,
        end_date=end_date,
        sampling_interval_minutes=sampling_minutes,
        num_panels=10,
        random_seed=42
    )
    
    df, fault_events = generator.generate()
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved sensor data to {output_path}")
        print(f"Generated {len(df)} samples over {days} days")
        print(f"Injected {len(fault_events)} fault events")
    
    return df, fault_events


if __name__ == "__main__":
    # Generate sample data
    df, faults = generate_sensor_data(
        days=365,
        sampling_minutes=15,
        output_path="data/raw/sensor_data.csv"
    )
    
    print("\nDataset Summary:")
    print(df.describe())
    
    print("\nFault Events:")
    for fault in faults:
        print(f"  - Fault {fault['fault_id']}: {fault['type']} "
              f"({fault['duration_hours']}h, severity: {fault['severity']:.2%})")
