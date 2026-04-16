import numpy as np
from datetime import datetime
from .base import BaseSimulator

class WindTurbineSimulator(BaseSimulator):
    """Simulates a Wind Turbine Asset."""
    
    def __init__(self, asset_id, config=None):
        default_config = {
            'rated_power': 2.5, # MW
            'rotor_diameter': 100, # meters
            'cut_in_speed': 3.5,
            'rated_speed': 12.0,
            'cut_out_speed': 25.0
        }
        if config:
            default_config.update(config)
        super().__init__(asset_id, default_config)

    def generate_step(self, timestamp):
        """Generate one step of wind turbine data."""
        hour = timestamp.hour + timestamp.minute / 60.0
        
        # 1. Wind Speed Model (Base wind + Gusts)
        # Often higher at night/evening
        base_wind = 7 + 3 * np.cos(np.pi * (hour - 2) / 12)
        wind_speed = base_wind + np.random.normal(0, 1.5)
        wind_speed = max(0, wind_speed)
        
        # 2. Power Output Calculation
        if wind_speed < self.config['cut_in_speed'] or wind_speed > self.config['cut_out_speed']:
            power = 0
            rotor_speed = 0
        elif wind_speed < self.config['rated_speed']:
            # Power follows cube of wind speed roughly
            efficiency = (wind_speed - self.config['cut_in_speed']) / (self.config['rated_speed'] - self.config['cut_in_speed'])
            power = self.config['rated_power'] * (efficiency ** 3)
            rotor_speed = 10 + 5 * efficiency
        else:
            # Rated power above rated speed (pitch control)
            power = self.config['rated_power'] * (1 - 0.02 * np.random.random())
            rotor_speed = 15 + np.random.normal(0, 0.2)
            
        # 3. Mechanical/Health Metrics
        # Vibration increases with load and wind turbulence
        vibration = 0.05 + (power / self.config['rated_power']) * 0.15 + np.random.normal(0, 0.02)
        vibration = max(0.01, vibration)
        
        # 2. Temperature Simulation (Diurnal & Seasonal)
        season = self.state.get('season_mode', 'spring')
        season_offset = 0.0
        if season == 'summer': season_offset = 15.0
        elif season == 'winter': season_offset = -10.0
        
        ambient_temp = 25 + season_offset + 5 * np.sin(np.pi * (hour - 12) / 12) + np.random.normal(0, 1)
        gen_temp = ambient_temp + (power / self.config['rated_power']) * 40 + np.random.normal(0, 2)
        gearbox_temp = gen_temp * 1.1 + np.random.normal(0, 3)
        
        # Pitch angle (controlled above rated speed)
        if wind_speed > self.config['rated_speed']:
            pitch_angle = (wind_speed - self.config['rated_speed']) * 2.5
        else:
            pitch_angle = 0
            
        # --- APPLIED ANOMALIES ---
        anomaly_mode = self.state.get('anomaly_mode', 'normal')
        is_fault = 0
        
        if anomaly_mode == 'gearbox_fault':
            vibration *= 4.5  # Critical vibration
            gearbox_temp += 35.0  # Overheating
            is_fault = 1
        elif anomaly_mode == 'sensor_drift':
            vibration += 0.5 + np.random.normal(0, 0.1) # Unnatural drift
            gen_temp -= 20.0 # Crazy unnatural temp reading
            is_fault = 1
        elif anomaly_mode == 'curtailment':
            power *= 0.2
            rotor_speed *= 0.6
            is_fault = 1

        return {
            'asset_id': self.asset_id,
            'timestamp': timestamp.isoformat(),
            'wind_speed': round(float(wind_speed), 2),
            'power_output': round(float(power), 2),
            'rotor_speed': round(float(rotor_speed), 2),
            'pitch_angle': round(float(pitch_angle), 2),
            'generator_temperature': round(float(gen_temp), 1),
            'gearbox_temperature': round(float(gearbox_temp), 1),
            'vibration_level': round(float(vibration), 3),
            'ambient_temperature': round(float(ambient_temp), 1),
            'is_fault': is_fault
        }
