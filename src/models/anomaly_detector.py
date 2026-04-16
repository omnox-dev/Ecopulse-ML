"""
Anomaly Detection Models for Predictive Maintenance

Models:
1. Isolation Forest - Fast, interpretable outlier detection
2. Autoencoder - Deep learning based reconstruction anomaly detection

Output: Anomaly scores (0-1) where higher = more anomalous
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class IsolationForestDetector:
    """Isolation Forest based anomaly detector."""
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of outliers
            n_estimators: Number of trees
            random_state: Random seed
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'IsolationForestDetector':
        """
        Fit the model on normal data.
        
        Args:
            X: Training data (N, features)
            feature_names: Optional feature names
            
        Returns:
            Self
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit isolation forest
        self.model.fit(X_scaled)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Input data (N, features)
            
        Returns:
            Binary labels (1 = normal, -1 = anomaly)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (higher = more anomalous).
        
        Args:
            X: Input data (N, features)
            
        Returns:
            Anomaly scores (0-1)
        """
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest returns negative scores (lower = more anomalous)
        raw_scores = self.model.score_samples(X_scaled)
        
        # Normalize to 0-1 range (higher = more anomalous)
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        
        if max_score - min_score > 0:
            normalized = 1 - (raw_scores - min_score) / (max_score - min_score)
        else:
            normalized = np.zeros_like(raw_scores)
        
        return normalized
    
    def save(self, path: str) -> None:
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)
        
    def load(self, path: str) -> 'IsolationForestDetector':
        """Load model from file."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        return self


class Autoencoder(nn.Module):
    """PyTorch Autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int, encoding_dim: int = 16):
        """
        Initialize autoencoder architecture.
        
        Args:
            input_dim: Number of input features
            encoding_dim: Size of latent space
        """
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (encoded, reconstructed)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AutoencoderDetector:
    """Autoencoder-based anomaly detector."""
    
    def __init__(
        self,
        encoding_dim: int = 16,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Initialize autoencoder detector.
        
        Args:
            encoding_dim: Latent space dimension
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            device: 'cuda' or 'cpu'
        """
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model: Optional[Autoencoder] = None
        self.scaler = StandardScaler()
        self.threshold: float = 0.0
        self.feature_names: List[str] = []
        
    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.1,
        verbose: bool = True
    ) -> 'AutoencoderDetector':
        """
        Train the autoencoder on normal data.
        
        Args:
            X: Training data (N, features)
            feature_names: Optional feature names
            validation_split: Fraction for validation
            verbose: Print training progress
            
        Returns:
            Self
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split validation
        n_val = int(len(X_scaled) * validation_split)
        X_train = X_scaled[:-n_val]
        X_val = X_scaled[-n_val:]
        
        # Convert to tensors
        train_tensor = torch.FloatTensor(X_train)
        val_tensor = torch.FloatTensor(X_val)
        
        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Initialize model
        input_dim = X.shape[1]
        self.model = Autoencoder(input_dim, self.encoding_dim).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                _, reconstructed = self.model(x)
                loss = criterion(reconstructed, x)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_input = val_tensor.to(self.device)
                _, val_reconstructed = self.model(val_input)
                val_loss = criterion(val_reconstructed, val_input).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Train Loss: {train_loss:.6f} - "
                      f"Val Loss: {val_loss:.6f}")
        
        # Set threshold based on training reconstruction error
        self.model.eval()
        with torch.no_grad():
            train_input = train_tensor.to(self.device)
            _, train_reconstructed = self.model(train_input)
            reconstruction_errors = torch.mean((train_reconstructed - train_input) ** 2, dim=1)
            
            # 95th percentile as threshold
            self.threshold = torch.quantile(reconstruction_errors, 0.95).item()
        
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores based on reconstruction error.
        
        Args:
            X: Input data (N, features)
            
        Returns:
            Anomaly scores (0-1)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            _, reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((reconstructed - X_tensor) ** 2, dim=1)
        
        errors = reconstruction_errors.cpu().numpy()
        
        # Normalize to 0-1 (using training threshold)
        normalized = np.clip(errors / (2 * self.threshold), 0, 1)
        
        return normalized
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Input data (N, features)
            
        Returns:
            Binary labels (1 = normal, -1 = anomaly)
        """
        scores = self.score_samples(X)
        return np.where(scores > 0.5, -1, 1)
    
    def save(self, path: str) -> None:
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict() if self.model else None,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'encoding_dim': self.encoding_dim,
            'input_dim': len(self.feature_names)
        }, path)
    
    def load(self, path: str) -> 'AutoencoderDetector':
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.scaler = checkpoint['scaler']
        self.threshold = checkpoint['threshold']
        self.feature_names = checkpoint['feature_names']
        self.encoding_dim = checkpoint['encoding_dim']
        
        input_dim = checkpoint['input_dim']
        self.model = Autoencoder(input_dim, self.encoding_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        
        return self


class EnsembleAnomalyDetector:
    """Ensemble of anomaly detection models."""
    
    def __init__(
        self,
        isolation_weight: float = 0.5,
        autoencoder_weight: float = 0.5
    ):
        """
        Initialize ensemble detector.
        
        Args:
            isolation_weight: Weight for Isolation Forest scores
            autoencoder_weight: Weight for Autoencoder scores
        """
        self.isolation_weight = isolation_weight
        self.autoencoder_weight = autoencoder_weight
        
        self.isolation_forest = IsolationForestDetector()
        self.autoencoder = AutoencoderDetector()
        self.feature_names: List[str] = []
        
    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True
    ) -> 'EnsembleAnomalyDetector':
        """
        Fit all models.
        
        Args:
            X: Training data
            feature_names: Feature names
            verbose: Print progress
            
        Returns:
            Self
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        if verbose:
            print("Training Isolation Forest...")
        self.isolation_forest.fit(X, self.feature_names)
        
        if verbose:
            print("Training Autoencoder...")
        self.autoencoder.fit(X, self.feature_names, verbose=verbose)
        
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble anomaly scores.
        
        Args:
            X: Input data
            
        Returns:
            Weighted average anomaly scores (0-1)
        """
        if_scores = self.isolation_forest.score_samples(X)
        ae_scores = self.autoencoder.score_samples(X)
        
        ensemble_scores = (
            self.isolation_weight * if_scores +
            self.autoencoder_weight * ae_scores
        )
        
        return ensemble_scores
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Input data
            threshold: Score threshold for anomaly
            
        Returns:
            Binary labels (1 = normal, -1 = anomaly)
        """
        scores = self.score_samples(X)
        return np.where(scores > threshold, -1, 1)
    
    def get_scores_breakdown(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual model scores.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary with scores from each model
        """
        return {
            'isolation_forest': self.isolation_forest.score_samples(X),
            'autoencoder': self.autoencoder.score_samples(X),
            'ensemble': self.score_samples(X)
        }
    
    def save(self, model_dir: str) -> None:
        """Save all models to directory."""
        os.makedirs(model_dir, exist_ok=True)
        self.isolation_forest.save(os.path.join(model_dir, 'isolation_forest.joblib'))
        self.autoencoder.save(os.path.join(model_dir, 'autoencoder.pt'))
    
    def load(self, model_dir: str) -> 'EnsembleAnomalyDetector':
        """Load all models from directory."""
        self.isolation_forest.load(os.path.join(model_dir, 'isolation_forest.joblib'))
        self.autoencoder.load(os.path.join(model_dir, 'autoencoder.pt'))
        self.feature_names = self.isolation_forest.feature_names
        return self


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Anomaly Detection Models...")
    
    # Generate synthetic normal data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X_normal = np.random.randn(n_samples, n_features)
    
    # Add some anomalies
    n_anomalies = 50
    X_anomalies = np.random.randn(n_anomalies, n_features) * 3 + 5
    
    X_test = np.vstack([X_normal[:100], X_anomalies])
    y_true = np.array([0] * 100 + [1] * n_anomalies)
    
    # Train and test Isolation Forest
    print("\n1. Isolation Forest")
    if_detector = IsolationForestDetector()
    if_detector.fit(X_normal)
    if_scores = if_detector.score_samples(X_test)
    print(f"   Mean score (normal): {if_scores[:100].mean():.3f}")
    print(f"   Mean score (anomaly): {if_scores[100:].mean():.3f}")
    
    # Train and test Autoencoder
    print("\n2. Autoencoder")
    ae_detector = AutoencoderDetector(epochs=30)
    ae_detector.fit(X_normal, verbose=False)
    ae_scores = ae_detector.score_samples(X_test)
    print(f"   Mean score (normal): {ae_scores[:100].mean():.3f}")
    print(f"   Mean score (anomaly): {ae_scores[100:].mean():.3f}")
    
    # Test ensemble
    print("\n3. Ensemble")
    ensemble = EnsembleAnomalyDetector()
    ensemble.fit(X_normal, verbose=False)
    ens_scores = ensemble.score_samples(X_test)
    print(f"   Mean score (normal): {ens_scores[:100].mean():.3f}")
    print(f"   Mean score (anomaly): {ens_scores[100:].mean():.3f}")
    
    print("\n✓ All models working correctly!")
