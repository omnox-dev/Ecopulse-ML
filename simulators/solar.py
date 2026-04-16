import numpy as np
from datetime import datetime
from .base import BaseSimulator

class SolarPanelSimulator(BaseSimulator):
    """Simulates a Solar Panel Grid Asset."""
    
    def __init__(self, asset_id, config=None):
        default_config = {
            'num_panels': 10,
            'rated_power': 400,
            'efficiency': 0.20,
            'base_temp': 25,
            'degradation_rate': 0.0001,  # daily
        }
        if config:
            default_config.update(config)
        super().__init__(asset_id, default_config)

    def generate_step(self, timestamp):
        """Generate one step of solar data."""
        hour = timestamp.hour + timestamp.minute / 60.0
        
        # 1. Simple Irradiance Model (Sinusoidal)
        # Peak at 13:00 (1 PM)
        irradiance = max(0, 1000 * np.sin(np.pi * (hour - 6) / 12))
        # Add random cloud cover noise (10-20% flicker)
        irradiance *= (0.85 + 0.15 * np.random.random())
        
        # 2. Temperature Model
        ambient_temp = 28 + 7 * np.sin(np.pi * (hour - 9) / 12) + np.random.normal(0, 1)
        # Panels heat up with irradiance
        panel_temp = ambient_temp + (irradiance / 800) * 25 + np.random.normal(0, 2)
        inverter_temp = ambient_temp + (irradiance / 800) * 15 + np.random.normal(0, 1)
        
        # 3. Electrical Model
        # Voltage decreases with temperature
        voltage = 40 * (1 - 0.004 * (panel_temp - 25)) + np.random.normal(0, 0.5)
        
        # Expected Power
        area = 2.0 # m2
        expected_power = irradiance * area * self.config['efficiency'] * self.config['num_panels']
        
        # Actual Power with degradation and variation
        days_since_start = (timestamp - datetime(2025, 1, 1)).days
        degradation = 1 - (self.config['degradation_rate'] * days_since_start)
        degradation = max(0.5, degradation)
        
        power_output = expected_power * degradation * (0.95 + 0.1 * np.random.random())
        
        # --- APPLIED ANOMALIES ---
        anomaly_mode = self.state.get('anomaly_mode', 'normal')
        is_fault = 0
        
        if anomaly_mode == 'inverter_overheat':
            inverter_temp += 45.0 + np.random.normal(0, 5)  # Spike temperature
            power_output *= 0.6  # Efficiency throttling
            is_fault = 1
        elif anomaly_mode == 'soiling':
            # Severe dust storm coverage
            power_output *= 0.5  
            is_fault = 1
        elif anomaly_mode == 'offline':
            power_output = 0.0
            is_fault = 1
            
        current = power_output / max(1.0, voltage)
        pr = power_output / max(1.0, expected_power)

        return {
            'asset_id': self.asset_id,
            'timestamp': timestamp.isoformat(),
            'voltage': round(float(voltage), 2),
            'current': round(float(current), 2),
            'power_output': round(float(power_output), 2),
            'expected_power': round(float(expected_power), 2),
            'performance_ratio': round(float(pr), 4),
            'panel_temperature': round(float(panel_temp), 1),
            'inverter_temperature': round(float(inverter_temp), 1),
            'ambient_temperature': round(float(ambient_temp), 1),
            'irradiance': round(float(irradiance), 0),
            'is_fault': is_fault
        }
