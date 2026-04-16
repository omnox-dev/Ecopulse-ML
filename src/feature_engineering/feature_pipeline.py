"""
Feature Engineering Pipeline for Predictive Maintenance

Creates features for ML models including:
- Rolling statistics (mean, std, min, max)
- Rate of change features
- Domain-specific features (Performance Ratio, Expected Power)
- Lag features
- Time-based features
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from datetime import datetime


class FeaturePipeline:
    """Feature engineering pipeline for solar panel sensor data."""
    
    # Rolling window sizes in hours
    WINDOW_SIZES = [1, 6, 24]
    
    # Features to compute rolling stats for
    ROLLING_FEATURES = [
        'power_output', 'voltage', 'current',
        'panel_temperature', 'inverter_temperature',
        'performance_ratio', 'irradiance'
    ]
    
    def __init__(self, sampling_minutes: int = 15):
        """
        Initialize the feature pipeline.
        
        Args:
            sampling_minutes: Sampling interval in minutes
        """
        self.sampling_minutes = sampling_minutes
        self.samples_per_hour = 60 // sampling_minutes
        
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with time features added
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if df['timestamp'].dtype != 'datetime64[ns]':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Is daytime (6 AM to 7 PM)
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] < 19)).astype(int)
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Add rolling statistics features.
        
        Args:
            df: DataFrame with sensor data
            features: List of features to compute rolling stats for
            windows: List of window sizes in hours
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        features = features or self.ROLLING_FEATURES
        windows = windows or self.WINDOW_SIZES
        
        for feature in features:
            if feature not in df.columns:
                continue
                
            for window_hours in windows:
                window_samples = window_hours * self.samples_per_hour
                
                # Rolling mean
                df[f'{feature}_mean_{window_hours}h'] = (
                    df[feature].rolling(window=window_samples, min_periods=1).mean()
                )
                
                # Rolling std
                df[f'{feature}_std_{window_hours}h'] = (
                    df[feature].rolling(window=window_samples, min_periods=1).std()
                )
                
                # Rolling min
                df[f'{feature}_min_{window_hours}h'] = (
                    df[feature].rolling(window=window_samples, min_periods=1).min()
                )
                
                # Rolling max
                df[f'{feature}_max_{window_hours}h'] = (
                    df[feature].rolling(window=window_samples, min_periods=1).max()
                )
        
        return df
    
    def add_rate_of_change_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Add rate of change features.
        
        Args:
            df: DataFrame with sensor data
            features: List of features to compute rate of change for
            
        Returns:
            DataFrame with rate of change features added
        """
        df = df.copy()
        features = features or ['power_output', 'panel_temperature', 'performance_ratio']
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            # Absolute change
            df[f'{feature}_diff_15m'] = df[feature].diff()
            df[f'{feature}_diff_1h'] = df[feature].diff(periods=self.samples_per_hour)
            
            # Percentage change
            df[f'{feature}_pct_change_15m'] = df[feature].pct_change()
            df[f'{feature}_pct_change_1h'] = df[feature].pct_change(periods=self.samples_per_hour)
        
        return df
    
    def add_lag_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Add lagged features for time series modeling.
        
        Args:
            df: DataFrame with sensor data
            features: List of features to create lags for
            lags: List of lag periods in hours
            
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        features = features or ['power_output', 'performance_ratio']
        lags = lags or [1, 6, 24]
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            for lag_hours in lags:
                lag_samples = lag_hours * self.samples_per_hour
                df[f'{feature}_lag_{lag_hours}h'] = df[feature].shift(lag_samples)
        
        return df
    
    def add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add domain-specific features for solar panels.
        
        Args:
            df: DataFrame with sensor data
            
        Returns:
            DataFrame with domain features added
        """
        df = df.copy()
        
        # Temperature stress indicator
        # Panels are less efficient when too hot (>35°C) or too cold (<15°C)
        df['temp_stress'] = np.where(
            df['panel_temperature'] > 35,
            (df['panel_temperature'] - 35) / 10,
            np.where(
                df['panel_temperature'] < 15,
                (15 - df['panel_temperature']) / 10,
                0
            )
        )
        df['temp_stress'] = np.clip(df['temp_stress'], 0, 1)
        
        # Inverter stress (temperature difference from panel)
        if 'inverter_temperature' in df.columns:
            df['inverter_stress'] = np.maximum(0, 
                (df['inverter_temperature'] - 50) / 20
            )
        
        # Power efficiency (actual vs expected, already as PR)
        if 'expected_power' in df.columns and 'power_output' in df.columns:
            df['power_deficit'] = df['expected_power'] - df['power_output']
            df['power_deficit_pct'] = df['power_deficit'] / (df['expected_power'] + 1e-6)
        
        # Dust/soiling impact estimate
        if 'dust_index' in df.columns:
            df['dust_impact'] = df['dust_index'] / 100
        
        # Combined weather stress
        if all(col in df.columns for col in ['humidity', 'wind_speed', 'dust_index']):
            df['weather_stress'] = (
                0.3 * (df['humidity'] / 100) +
                0.2 * np.minimum(1, df['wind_speed'] / 15) +
                0.5 * (df['dust_index'] / 100)
            )
        
        # Cumulative operating hours
        if 'is_daytime' in df.columns:
            df['cumulative_operating_hours'] = (
                df['is_daytime'].cumsum() * (self.sampling_minutes / 60)
            )
        
        return df
    
    def add_anomaly_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic anomaly indicator features.
        
        Args:
            df: DataFrame with sensor data
            
        Returns:
            DataFrame with anomaly indicator features
        """
        df = df.copy()
        
        # Z-score based indicators
        for feature in ['power_output', 'performance_ratio', 'panel_temperature']:
            if feature not in df.columns:
                continue
            
            # Calculate z-score using rolling window
            window = 24 * self.samples_per_hour  # 24 hours
            rolling_mean = df[feature].rolling(window=window, min_periods=1).mean()
            rolling_std = df[feature].rolling(window=window, min_periods=1).std()
            
            df[f'{feature}_zscore'] = (df[feature] - rolling_mean) / (rolling_std + 1e-6)
            
            # Flag extreme values
            df[f'{feature}_extreme'] = (np.abs(df[f'{feature}_zscore']) > 2).astype(int)
        
        # Performance ratio deviation from expected
        if 'performance_ratio' in df.columns:
            # During daytime, PR should be around 0.8-1.0
            expected_pr = 0.85
            df['pr_deviation'] = np.abs(df['performance_ratio'] - expected_pr)
            df['low_pr_flag'] = (df['performance_ratio'] < 0.6).astype(int)
        
        return df
    
    def transform(
        self,
        df: pd.DataFrame,
        include_time: bool = True,
        include_rolling: bool = True,
        include_rate_of_change: bool = True,
        include_lags: bool = True,
        include_domain: bool = True,
        include_anomaly: bool = True
    ) -> pd.DataFrame:
        """
        Apply full feature engineering pipeline.
        
        Args:
            df: Raw sensor/weather DataFrame
            include_*: Flags to include each feature category
            
        Returns:
            DataFrame with all features added
        """
        df = df.copy()
        
        if include_time:
            df = self.add_time_features(df)
        
        if include_rolling:
            df = self.add_rolling_features(df)
        
        if include_rate_of_change:
            df = self.add_rate_of_change_features(df)
        
        if include_lags:
            df = self.add_lag_features(df)
        
        if include_domain:
            df = self.add_domain_features(df)
        
        if include_anomaly:
            df = self.add_anomaly_indicators(df)
        
        # Handle infinite values (replace with NaN)
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values created by rolling/lag operations
        # Use ffill/bfill directly as method kwarg is deprecated
        df = df.ffill().bfill()
        
        # Fill any remaining NaNs with 0 (e.g. if entire column is NaN)
        df = df.fillna(0)
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature column names (excluding metadata).
        
        Args:
            df: Transformed DataFrame
            
        Returns:
            List of feature column names
        """
        exclude_cols = [
            'timestamp', 'is_fault', 'fault_id',
            'failure_within_7d', 'efficiency_dropping'
        ]
        return [col for col in df.columns if col not in exclude_cols]


def create_ml_datasets(
    df: pd.DataFrame,
    target_column: str = 'is_fault',
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    Time-series aware split (no shuffling).
    
    Args:
        df: Feature-engineered DataFrame
        target_column: Name of target column
        test_size: Fraction for test set
        val_size: Fraction for validation set
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    n = len(df)
    test_start = int(n * (1 - test_size))
    val_start = int(n * (1 - test_size - val_size))
    
    train_df = df.iloc[:val_start].copy()
    val_df = df.iloc[val_start:test_start].copy()
    test_df = df.iloc[test_start:].copy()
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    import os
    
    # Load combined data
    input_path = "data/raw/combined_data.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run generate_all.py first.")
        exit(1)
    
    print("Loading data...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} samples with {len(df.columns)} columns")
    
    # Apply feature engineering
    print("\nApplying feature engineering...")
    pipeline = FeaturePipeline(sampling_minutes=15)
    df_features = pipeline.transform(df)
    
    print(f"Generated {len(df_features.columns)} features")
    
    # Save processed data
    output_path = "data/processed/features.csv"
    os.makedirs("data/processed", exist_ok=True)
    df_features.to_csv(output_path, index=False)
    print(f"\nSaved features to {output_path}")
    
    # Print feature summary
    print("\nFeature Categories:")
    time_features = [c for c in df_features.columns if any(x in c for x in ['hour', 'day', 'month', 'is_day', 'is_week'])]
    rolling_features = [c for c in df_features.columns if any(x in c for x in ['_mean_', '_std_', '_min_', '_max_'])]
    change_features = [c for c in df_features.columns if '_diff_' in c or '_pct_change_' in c]
    lag_features = [c for c in df_features.columns if '_lag_' in c]
    
    print(f"  - Time features: {len(time_features)}")
    print(f"  - Rolling features: {len(rolling_features)}")
    print(f"  - Rate of change features: {len(change_features)}")
    print(f"  - Lag features: {len(lag_features)}")
