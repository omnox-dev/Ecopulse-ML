"""
Training Script for Failure Prediction Model Only
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary modules
from src.feature_engineering.feature_pipeline import FeaturePipeline, create_ml_datasets
from src.models.failure_predictor import FailurePredictor

def load_processed_data(features_path: str = "data/processed/features.csv"):
    """Load pre-computed features if available."""
    if os.path.exists(features_path):
        print(f"Loading features from {features_path}...")
        return pd.read_csv(features_path)
    else:
        print(f"Features file not found at {features_path}")
        print("Please run src/models/train_all.py first or ensure data exists.")
        sys.exit(1)

def train_failure_predictor(df: pd.DataFrame, model_dir: str = "models"):
    """Train failure prediction models."""
    print("\n" + "=" * 50)
    print("TRAINING FAILURE PREDICTION MODELS (Dedicated Script)")
    print("=" * 50)
    
    # Check target distribution
    target_col = 'failure_within_7d'
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in data.")
        return

    total_failures = df[target_col].sum()
    print(f"Total samples: {len(df):,}")
    print(f"Total failures: {total_failures:,} ({total_failures/len(df):.2%})")

    # Split data
    # We might need to adjust split sizes if test set has no failures
    train_df, val_df, test_df = create_ml_datasets(
        df,
        target_column=target_col,
        test_size=0.15,
        val_size=0.1
    )
    
    # Check distribution in splits
    train_failures = train_df[target_col].sum()
    val_failures = val_df[target_col].sum()
    test_failures = test_df[target_col].sum()

    print(f"\nSplit Distribution:")
    print(f"Train: {len(train_df):,} samples | {train_failures:,} failures ({train_failures/len(train_df):.2%})")
    print(f"Val:   {len(val_df):,} samples | {val_failures:,} failures ({val_failures/len(val_df):.2%})")
    print(f"Test:  {len(test_df):,} samples | {test_failures:,} failures ({test_failures/len(test_df):.2%})")

    if test_failures == 0:
        print("\nWARNING: Test set has no failures. Evaluation metrics (Precision/Recall/F1) will be 0.0.")
        print("Consider checking your data generation or time-split strategy.")
    
    # Prepare features
    exclude_cols = [
        'timestamp', 'is_fault', 'fault_id',
        'failure_within_7d', 'efficiency_dropping',
        'pr_rolling_mean', 'pr_change'
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Train model
    print("\nTraining model...")
    predictor = FailurePredictor()
    predictor.fit(X_train, y_train, feature_names=feature_cols, X_val=X_val, y_val=y_val)
    
    # Evaluate on test set
    print("\nEvaluating on Test Set...")
    metrics = predictor.evaluate(X_test, y_test)
    
    print(f"\nTest Set Results:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "failure_predictor.joblib")
    predictor.save(model_path)
    print(f"\nModel saved to: {model_path}")

def main():
    # Load data
    df = load_processed_data()
    
    # Train
    train_failure_predictor(df)

if __name__ == "__main__":
    main()
