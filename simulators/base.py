import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from abc import ABC, abstractmethod
from pathlib import Path

# Directory where this file lives — always the simulators/ folder
_SIMULATORS_DIR = Path(__file__).resolve().parent

class BaseSimulator(ABC):
    """Base class for all asset simulators."""
    
    def __init__(self, asset_id, config=None, state_file=None):
        self.asset_id = asset_id
        self.config = config or {}
        # Anchor state files to the simulators/ dir — works from any CWD
        self.state_file = state_file or str(_SIMULATORS_DIR / f"state_{asset_id}.json")
        self.state = self.load_state()
        
        # Default start time if no state
        if 'last_timestamp' not in self.state:
            self.state['last_timestamp'] = (datetime.now() - timedelta(days=1)).isoformat()
            
    def load_state(self):
        """Load simulator state from file."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}

    def save_state(self):
        """Save current simulator state."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    @abstractmethod
    def generate_step(self, timestamp):
        """Generate a single data point for a given timestamp."""
        pass

    def simulate_until(self, end_time=None, interval_minutes=15):
        """Simulate asset behavior from last timestamp until end_time."""
        if end_time is None:
            end_time = datetime.now()
            
        last_time = datetime.fromisoformat(self.state['last_timestamp'])
        current_time = last_time + timedelta(minutes=interval_minutes)
        
        new_data = []
        
        while current_time <= end_time:
            point = self.generate_step(current_time)
            new_data.append(point)
            current_time += timedelta(minutes=interval_minutes)
            
        if new_data:
            df = pd.DataFrame(new_data)
            self.state['last_timestamp'] = current_time.isoformat()
            self.save_state()
            return df
        
        return pd.DataFrame()
