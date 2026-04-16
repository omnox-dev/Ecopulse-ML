"""
Failure Prediction Model for Predictive Maintenance

Models:
1. Random Forest - Interpretable, robust to outliers
2. XGBoost - High performance gradient boosting
3. Ensemble - Weighted combination

Target: Predict if equipment will fail within next 7 days
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
import xgboost as xgb
import joblib
import os


class FailurePredictor:
    """Failure prediction using Random Forest and XGBoost ensemble."""
    
    def __init__(
        self,
        rf_weight: float = 0.4,
        xgb_weight: float = 0.6,
        random_state: int = 42
    ):
        """
        Initialize failure predictor.
        
        Args:
            rf_weight: Weight for Random Forest predictions
            xgb_weight: Weight for XGBoost predictions
            random_state: Random seed
        """
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.rf_model: Optional[RandomForestClassifier] = None
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.feature_importances_: Dict[str, float] = {}
        
    def _create_rf_model(self) -> RandomForestClassifier:
        """Create Random Forest model."""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def _create_xgb_model(self, scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
        """Create XGBoost model."""
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> 'FailurePredictor':
        """
        Train the failure prediction models.
        
        Args:
            X: Training features (N, features)
            y: Training labels (N,) - 1 for failure, 0 for normal
            feature_names: Optional feature names
            X_val: Validation features
            y_val: Validation labels
            verbose: Print training info
            
        Returns:
            Self
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate class weight for imbalanced data
        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)
        
        if verbose:
            print(f"Training data: {len(X)} samples")
            print(f"Class distribution: {n_neg} normal, {n_pos} failures ({n_pos/len(y):.2%})")
        
        # Train Random Forest
        if verbose:
            print("\nTraining Random Forest...")
        self.rf_model = self._create_rf_model()
        self.rf_model.fit(X_scaled, y)
        
        # Train XGBoost
        if verbose:
            print("Training XGBoost...")
        self.xgb_model = self._create_xgb_model(scale_pos_weight)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.xgb_model.fit(
                X_scaled, y,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        else:
            self.xgb_model.fit(X_scaled, y)
        
        # Calculate feature importances (average of both models)
        rf_importance = self.rf_model.feature_importances_
        xgb_importance = self.xgb_model.feature_importances_
        
        avg_importance = (rf_importance + xgb_importance) / 2
        self.feature_importances_ = dict(zip(self.feature_names, avg_importance))
        
        if verbose:
            print("\nTop 10 Important Features:")
            top_features = sorted(
                self.feature_importances_.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for name, importance in top_features:
                print(f"  {name}: {importance:.4f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict failure probability.
        
        Args:
            X: Input features (N, features)
            
        Returns:
            Failure probabilities (N,)
        """
        if self.rf_model is None or self.xgb_model is None:
            raise ValueError("Models not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities from both models
        rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X_scaled)[:, 1]
        
        # Weighted ensemble
        ensemble_proba = (
            self.rf_weight * rf_proba +
            self.xgb_weight * xgb_proba
        )
        
        return ensemble_proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict failure labels.
        
        Args:
            X: Input features (N, features)
            threshold: Probability threshold for failure
            
        Returns:
            Binary labels (N,) - 1 for failure, 0 for normal
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_predictions_breakdown(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual model predictions.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with probabilities from each model
        """
        X_scaled = self.scaler.transform(X)
        
        return {
            'random_forest': self.rf_model.predict_proba(X_scaled)[:, 1],
            'xgboost': self.xgb_model.predict_proba(X_scaled)[:, 1],
            'ensemble': self.predict_proba(X)
        }
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True labels
            threshold: Classification threshold
            
        Returns:
            Dictionary with evaluation metrics
        """
        proba = self.predict_proba(X)
        predictions = (proba >= threshold).astype(int)
        
        metrics = {
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1': f1_score(y, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0,
            'confusion_matrix': confusion_matrix(y, predictions),
            'classification_report': classification_report(y, predictions, zero_division=0)
        }
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance DataFrame.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importances_.items()
        ])
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save(self, path: str) -> None:
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances_,
            'rf_weight': self.rf_weight,
            'xgb_weight': self.xgb_weight
        }, path)
    
    def load(self, path: str) -> 'FailurePredictor':
        """Load model from file."""
        data = joblib.load(path)
        self.rf_model = data['rf_model']
        self.xgb_model = data['xgb_model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.feature_importances_ = data['feature_importances']
        self.rf_weight = data['rf_weight']
        self.xgb_weight = data['xgb_weight']
        return self


def train_failure_predictor(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_column: str = 'failure_within_7d',
    exclude_columns: Optional[List[str]] = None
) -> Tuple[FailurePredictor, Dict[str, Any]]:
    """
    Train failure predictor on prepared data.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        target_column: Name of target column
        exclude_columns: Columns to exclude from features
        
    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    exclude_cols = exclude_columns or [
        'timestamp', 'is_fault', 'fault_id',
        'failure_within_7d', 'efficiency_dropping'
    ]
    
    # Prepare features
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_column].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df[target_column].values
    
    # Train model
    predictor = FailurePredictor()
    predictor.fit(
        X_train, y_train,
        feature_names=feature_cols,
        X_val=X_val, y_val=y_val,
        verbose=True
    )
    
    # Evaluate
    metrics = predictor.evaluate(X_val, y_val)
    
    print("\nValidation Results:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    return predictor, metrics


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Failure Prediction Model...")
    
    np.random.seed(42)
    n_samples = 2000
    n_features = 20
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic labels (10% failure rate)
    y = np.zeros(n_samples)
    failure_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    y[failure_indices] = 1
    
    # Make failures slightly distinguishable
    X[failure_indices, 0] += 2
    X[failure_indices, 1] -= 1.5
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train and evaluate
    predictor = FailurePredictor()
    predictor.fit(X_train, y_train, verbose=True)
    
    metrics = predictor.evaluate(X_test, y_test)
    
    print("\nTest Results:")
    print(metrics['classification_report'])
    
    print("\n✓ Failure Predictor working correctly!")
