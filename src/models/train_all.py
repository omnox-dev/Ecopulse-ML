"""
Training Script for All ML Models

This script:
1. Generates/loads data
2. Engineers features
3. Trains anomaly detection, failure prediction, and efficiency forecasting models
4. Saves trained models
5. Evaluates and reports metrics
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from datetime import datetime

from src.feature_engineering.feature_pipeline import FeaturePipeline, create_ml_datasets
from src.models.anomaly_detector import EnsembleAnomalyDetector
from src.models.failure_predictor import FailurePredictor
from src.models.efficiency_forecaster import EfficiencyForecaster


def load_data(data_path: str = "data/raw/combined_data.csv") -> pd.DataFrame:
    """Load and validate data."""
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        print("Running data generation...")
        
        # Import and run data generation
        from src.data_generation.generate_all import generate_all_data, create_training_labels
        
        data = generate_all_data(
            days=365,
            sampling_minutes=15,
            output_dir="data/raw"
        )
        
        df = create_training_labels(data['combined_df'])
        df.to_csv("data/processed/labeled_data.csv", index=False)
        
        return df
    else:
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Add labels if not present
        if 'failure_within_7d' not in df.columns:
            from src.data_generation.generate_all import create_training_labels
            df = create_training_labels(df)
        
        return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering pipeline."""
    print("\nEngineering features...")
    
    pipeline = FeaturePipeline(sampling_minutes=15)
    df_features = pipeline.transform(df)
    
    print(f"Created {len(df_features.columns)} features")
    
    return df_features


def train_anomaly_detector(
    df: pd.DataFrame,
    model_dir: str = "models"
) -> EnsembleAnomalyDetector:
    """Train anomaly detection models."""
    print("\n" + "=" * 50)
    print("TRAINING ANOMALY DETECTION MODELS")
    print("=" * 50)
    
    # Use only normal data for training (unsupervised)
    normal_mask = df['is_fault'] == 0
    df_normal = df[normal_mask]
    
    print(f"Training on {len(df_normal):,} normal samples")
    
    # Prepare features
    exclude_cols = [
        'timestamp', 'is_fault', 'fault_id',
        'failure_within_7d', 'efficiency_dropping',
        'pr_rolling_mean', 'pr_change'
    ]
    feature_cols = [c for c in df_normal.columns if c not in exclude_cols]
    
    X_train = df_normal[feature_cols].values
    
    # Train ensemble
    detector = EnsembleAnomalyDetector()
    detector.fit(X_train, feature_names=feature_cols, verbose=True)
    
    # Evaluate on full data
    X_full = df[feature_cols].values
    y_true = df['is_fault'].values
    
    scores = detector.score_samples(X_full)
    predictions = detector.predict(X_full, threshold=0.5)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    y_pred = (predictions == -1).astype(int)  # -1 = anomaly
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\nAnomaly Detection Results:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Save model
    model_path = os.path.join(model_dir, "anomaly_detector")
    os.makedirs(model_path, exist_ok=True)
    detector.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    return detector


def train_failure_predictor(
    df: pd.DataFrame,
    model_dir: str = "models"
) -> FailurePredictor:
    """Train failure prediction models."""
    print("\n" + "=" * 50)
    print("TRAINING FAILURE PREDICTION MODELS")
    print("=" * 50)
    
    # Split data
    train_df, val_df, test_df = create_ml_datasets(
        df,
        target_column='failure_within_7d',
        test_size=0.15,
        val_size=0.1
    )
    
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    # Prepare features
    exclude_cols = [
        'timestamp', 'is_fault', 'fault_id',
        'failure_within_7d', 'efficiency_dropping',
        'pr_rolling_mean', 'pr_change'
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['failure_within_7d'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['failure_within_7d'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['failure_within_7d'].values
    
    # Train model
    predictor = FailurePredictor()
    predictor.fit(X_train, y_train, feature_names=feature_cols, X_val=X_val, y_val=y_val)
    
    # Evaluate on test set
    metrics = predictor.evaluate(X_test, y_test)
    
    print(f"\nTest Set Results:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Save model
    model_path = os.path.join(model_dir, "failure_predictor.joblib")
    predictor.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    return predictor


def train_efficiency_forecaster(
    df: pd.DataFrame,
    model_dir: str = "models"
) -> EfficiencyForecaster:
    """Train efficiency forecasting model."""
    print("\n" + "=" * 50)
    print("TRAINING EFFICIENCY FORECASTING MODEL")
    print("=" * 50)
    
    # Prepare data (filter daytime only for meaningful efficiency)
    if 'is_daytime' in df.columns:
        df_day = df[df['is_daytime'] == 1].copy()
    else:
        df_day = df.copy()
    
    # Use 80% for training
    train_size = int(0.8 * len(df_day))
    df_train = df_day.iloc[:train_size]
    df_test = df_day.iloc[train_size:]
    
    print(f"Train: {len(df_train):,} | Test: {len(df_test):,}")
    
    # Prepare features
    exclude_cols = [
        'timestamp', 'is_fault', 'fault_id',
        'failure_within_7d', 'efficiency_dropping',
        'pr_rolling_mean', 'pr_change', 'performance_ratio'
    ]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    
    X_train = df_train[feature_cols].values
    y_train = df_train['performance_ratio'].values
    
    X_test = df_test[feature_cols].values
    y_test = df_test['performance_ratio'].values
    
    # Train model (reduced sequence length for faster training)
    forecaster = EfficiencyForecaster(
        model_type='lstm',
        sequence_length=96,   # 24 hours at 15-min intervals
        forecast_horizon=24,  # 6 hours ahead
        epochs=30,
        hidden_dim=32
    )
    
    forecaster.fit(
        X_train, y_train,
        feature_names=feature_cols,
        target_name='performance_ratio',
        verbose=True
    )
    
    # Evaluate
    metrics = forecaster.evaluate(X_test, y_test)
    
    print(f"\nTest Set Results:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    # Save model
    model_path = os.path.join(model_dir, "efficiency_forecaster.pt")
    forecaster.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    return forecaster


def main():
    """Run full training pipeline."""
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE - MODEL TRAINING")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Engineer features
    df_features = engineer_features(df)
    
    # Save processed features
    df_features.to_csv("data/processed/features.csv", index=False)
    print("Features saved to data/processed/features.csv")
    
    # Train models
    anomaly_detector = train_anomaly_detector(df_features)
    failure_predictor = train_failure_predictor(df_features)
    efficiency_forecaster = train_efficiency_forecaster(df_features)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nSaved Models:")
    print("  - models/anomaly_detector/")
    print("  - models/failure_predictor.joblib")
    print("  - models/efficiency_forecaster.pt")


if __name__ == "__main__":
    main()
