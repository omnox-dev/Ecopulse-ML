"""
Unit Tests for ML Models
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.anomaly_detector import (
    IsolationForestDetector, 
    AutoencoderDetector, 
    EnsembleAnomalyDetector
)
from src.models.failure_predictor import FailurePredictor


class TestIsolationForestDetector:
    """Tests for Isolation Forest anomaly detector."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        X_normal = np.random.randn(500, 10)
        X_anomaly = np.random.randn(50, 10) * 3 + 5
        return X_normal, X_anomaly
    
    def test_fit(self, sample_data):
        """Test model fitting."""
        X_normal, _ = sample_data
        
        detector = IsolationForestDetector()
        detector.fit(X_normal)
        
        assert detector.model is not None
        assert detector.scaler is not None
    
    def test_score_samples(self, sample_data):
        """Test anomaly scoring."""
        X_normal, X_anomaly = sample_data
        
        detector = IsolationForestDetector()
        detector.fit(X_normal)
        
        normal_scores = detector.score_samples(X_normal[:50])
        anomaly_scores = detector.score_samples(X_anomaly)
        
        # Anomalies should have higher scores
        assert anomaly_scores.mean() > normal_scores.mean()
        
        # Scores should be in [0, 1]
        assert (normal_scores >= 0).all() and (normal_scores <= 1).all()
        assert (anomaly_scores >= 0).all() and (anomaly_scores <= 1).all()
    
    def test_predict(self, sample_data):
        """Test anomaly prediction."""
        X_normal, X_anomaly = sample_data
        
        detector = IsolationForestDetector(contamination=0.1)
        detector.fit(X_normal)
        
        predictions = detector.predict(X_anomaly)
        
        # Most anomalies should be detected
        anomaly_detected = (predictions == -1).mean()
        assert anomaly_detected > 0.5


class TestAutoencoderDetector:
    """Tests for Autoencoder anomaly detector."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        X_normal = np.random.randn(500, 10)
        X_anomaly = np.random.randn(50, 10) * 3 + 5
        return X_normal, X_anomaly
    
    def test_fit(self, sample_data):
        """Test model fitting."""
        X_normal, _ = sample_data
        
        detector = AutoencoderDetector(epochs=5)
        detector.fit(X_normal, verbose=False)
        
        assert detector.model is not None
        assert detector.threshold > 0
    
    def test_score_samples(self, sample_data):
        """Test anomaly scoring."""
        X_normal, X_anomaly = sample_data
        
        detector = AutoencoderDetector(epochs=10)
        detector.fit(X_normal, verbose=False)
        
        normal_scores = detector.score_samples(X_normal[:50])
        anomaly_scores = detector.score_samples(X_anomaly)
        
        # Anomalies should generally have higher scores
        assert anomaly_scores.mean() > normal_scores.mean()


class TestEnsembleAnomalyDetector:
    """Tests for ensemble anomaly detector."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        X_normal = np.random.randn(500, 10)
        X_anomaly = np.random.randn(50, 10) * 3 + 5
        return X_normal, X_anomaly
    
    def test_fit_and_score(self, sample_data):
        """Test ensemble fitting and scoring."""
        X_normal, X_anomaly = sample_data
        
        ensemble = EnsembleAnomalyDetector()
        ensemble.autoencoder.epochs = 5  # Reduce for faster test
        ensemble.fit(X_normal, verbose=False)
        
        normal_scores = ensemble.score_samples(X_normal[:50])
        anomaly_scores = ensemble.score_samples(X_anomaly)
        
        assert anomaly_scores.mean() > normal_scores.mean()
    
    def test_get_scores_breakdown(self, sample_data):
        """Test individual model scores."""
        X_normal, X_anomaly = sample_data
        
        ensemble = EnsembleAnomalyDetector()
        ensemble.autoencoder.epochs = 5
        ensemble.fit(X_normal, verbose=False)
        
        breakdown = ensemble.get_scores_breakdown(X_anomaly[:10])
        
        assert 'isolation_forest' in breakdown
        assert 'autoencoder' in breakdown
        assert 'ensemble' in breakdown


class TestFailurePredictor:
    """Tests for failure prediction model."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 15
        
        X = np.random.randn(n_samples, n_features)
        y = np.zeros(n_samples)
        
        # Create some positive samples
        failure_idx = np.random.choice(n_samples, size=100, replace=False)
        y[failure_idx] = 1
        X[failure_idx, 0] += 2
        X[failure_idx, 1] -= 1.5
        
        return X, y
    
    def test_fit(self, sample_data):
        """Test model fitting."""
        X, y = sample_data
        
        predictor = FailurePredictor()
        predictor.fit(X, y, verbose=False)
        
        assert predictor.rf_model is not None
        assert predictor.xgb_model is not None
    
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        
        predictor = FailurePredictor()
        predictor.fit(X, y, verbose=False)
        
        proba = predictor.predict_proba(X[:100])
        
        assert len(proba) == 100
        assert (proba >= 0).all() and (proba <= 1).all()
    
    def test_predict(self, sample_data):
        """Test label prediction."""
        X, y = sample_data
        
        predictor = FailurePredictor()
        predictor.fit(X, y, verbose=False)
        
        predictions = predictor.predict(X[:100])
        
        assert len(predictions) == 100
        assert set(predictions).issubset({0, 1})
    
    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        
        # Split data
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        predictor = FailurePredictor()
        predictor.fit(X_train, y_train, verbose=False)
        
        metrics = predictor.evaluate(X_test, y_test)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
    
    def test_feature_importance(self, sample_data):
        """Test feature importance retrieval."""
        X, y = sample_data
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        predictor = FailurePredictor()
        predictor.fit(X, y, feature_names=feature_names, verbose=False)
        
        importance_df = predictor.get_feature_importance(top_n=5)
        
        assert len(importance_df) == 5
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
