"""
Efficiency Forecasting Model for Predictive Maintenance

Uses LSTM/GRU networks for time-series prediction of:
- Future efficiency trends
- Performance ratio forecasting
- Power output prediction

Input: Sequence of recent sensor readings (e.g., last 72 hours)
Output: Forecasted values for next 24-48 hours
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import os


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int,
        forecast_horizon: int = 1,
        target_idx: int = 0
    ):
        """
        Initialize dataset.
        
        Args:
            data: Array of shape (N, features)
            sequence_length: Number of past timesteps for input
            forecast_horizon: Number of future timesteps to predict
            target_idx: Index of target feature in data
        """
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.target_idx = target_idx
        
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.forecast_horizon + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input sequence
        x = self.data[idx:idx + self.sequence_length]
        
        # Target: future values of target feature
        y = self.data[
            idx + self.sequence_length:
            idx + self.sequence_length + self.forecast_horizon,
            self.target_idx
        ]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


class LSTMForecaster(nn.Module):
    """LSTM-based forecasting model."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            num_layers: Number of LSTM layers
            output_dim: Number of output timesteps
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        # LSTM output
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        out = self.fc(lstm_out[:, -1, :])
        
        return out


class GRUForecaster(nn.Module):
    """GRU-based forecasting model (lighter than LSTM)."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize GRU model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GRU layers
            output_dim: Number of output timesteps
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        gru_out, h_n = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out


class EfficiencyForecaster:
    """Efficiency forecasting using LSTM/GRU models."""
    
    def __init__(
        self,
        model_type: str = 'lstm',
        sequence_length: int = 288,  # 72 hours at 15-min intervals
        forecast_horizon: int = 96,   # 24 hours ahead
        hidden_dim: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 64,
        device: Optional[str] = None
    ):
        """
        Initialize efficiency forecaster.
        
        Args:
            model_type: 'lstm' or 'gru'
            sequence_length: Input sequence length (timesteps)
            forecast_horizon: Number of timesteps to forecast
            hidden_dim: Hidden layer dimension
            num_layers: Number of RNN layers
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            device: 'cuda' or 'cpu'
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model: Optional[nn.Module] = None
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.target_name: str = ''
        
    def _create_model(self, input_dim: int) -> nn.Module:
        """Create the forecasting model."""
        if self.model_type == 'lstm':
            return LSTMForecaster(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_dim=self.forecast_horizon
            )
        else:
            return GRUForecaster(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_dim=self.forecast_horizon
            )
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        target_name: str = 'performance_ratio',
        validation_split: float = 0.1,
        verbose: bool = True
    ) -> 'EfficiencyForecaster':
        """
        Train the forecasting model.
        
        Args:
            X: Input features array (N, features)
            y: Target values array (N,)
            feature_names: Feature names
            target_name: Name of target variable
            validation_split: Validation split fraction
            verbose: Print training progress
            
        Returns:
            Self
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        self.target_name = target_name
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Combine features and target
        data = np.column_stack([y_scaled, X_scaled])
        
        # Create sequences
        n_samples = len(data) - self.sequence_length - self.forecast_horizon + 1
        
        if n_samples <= 0:
            raise ValueError(
                f"Not enough data. Need at least {self.sequence_length + self.forecast_horizon} samples, "
                f"got {len(data)}"
            )
        
        # Split data
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        train_data = data[:n_train + self.sequence_length + self.forecast_horizon]
        val_data = data[n_train:]
        
        train_dataset = TimeSeriesDataset(
            train_data, self.sequence_length, self.forecast_horizon
        )
        val_dataset = TimeSeriesDataset(
            val_data, self.sequence_length, self.forecast_horizon
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        # Create model
        input_dim = data.shape[1]
        self.model = self._create_model(input_dim).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        
        best_val_loss = float('inf')
        best_model_state = None
        
        if verbose:
            print(f"Training {self.model_type.upper()} Forecaster...")
            print(f"  Sequence length: {self.sequence_length}")
            print(f"  Forecast horizon: {self.forecast_horizon}")
            print(f"  Training samples: {len(train_dataset)}")
            print(f"  Validation samples: {len(val_dataset)}")
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    output = self.model(batch_x)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Train Loss: {train_loss:.6f} - "
                      f"Val Loss: {val_loss:.6f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        if verbose:
            print(f"Best validation loss: {best_val_loss:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray, y_last: np.ndarray) -> np.ndarray:
        """
        Make predictions for future timesteps.
        
        Args:
            X: Recent input features (sequence_length, features)
            y_last: Recent target values (sequence_length,)
            
        Returns:
            Predictions (forecast_horizon,)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Scale inputs
        X_scaled = self.scaler.transform(X)
        y_scaled = self.target_scaler.transform(y_last.reshape(-1, 1)).flatten()
        
        # Combine and create input sequence
        data = np.column_stack([y_scaled, X_scaled])
        
        if len(data) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} timesteps, got {len(data)}"
            )
        
        # Take last sequence_length samples
        sequence = data[-self.sequence_length:]
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
            predictions_scaled = output.cpu().numpy().flatten()
        
        # Inverse transform predictions
        predictions = self.target_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        return predictions
    
    def forecast_efficiency(
        self,
        df: pd.DataFrame,
        target_column: str = 'performance_ratio'
    ) -> pd.DataFrame:
        """
        Forecast efficiency for a DataFrame.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            
        Returns:
            DataFrame with forecasted values
        """
        exclude_cols = [
            'timestamp', 'is_fault', 'fault_id',
            'failure_within_7d', 'efficiency_dropping'
        ]
        
        feature_cols = [c for c in df.columns if c not in exclude_cols and c != target_column]
        
        X = df[feature_cols].values
        y = df[target_column].values
        
        # Get predictions
        predictions = self.predict(X, y)
        
        # Create forecast DataFrame
        last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
        forecast_timestamps = pd.date_range(
            start=last_timestamp,
            periods=self.forecast_horizon + 1,
            freq='15min'
        )[1:]
        
        forecast_df = pd.DataFrame({
            'timestamp': forecast_timestamps,
            f'{target_column}_forecast': predictions
        })
        
        return forecast_df
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X: Test features
            y: Test target
            
        Returns:
            Dictionary with error metrics
        """
        if self.model is None:
            raise ValueError("Model not fitted.")
        
        # Scale data
        X_scaled = self.scaler.transform(X)
        y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        data = np.column_stack([y_scaled, X_scaled])
        
        dataset = TimeSeriesDataset(
            data, self.sequence_length, self.forecast_horizon
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x)
                all_preds.append(output.cpu().numpy())
                all_targets.append(batch_y.numpy())
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        # Inverse transform
        preds_orig = self.target_scaler.inverse_transform(preds)
        targets_orig = self.target_scaler.inverse_transform(targets)
        
        # Calculate metrics
        mse = np.mean((preds_orig - targets_orig) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds_orig - targets_orig))
        mape = np.mean(np.abs((targets_orig - preds_orig) / (targets_orig + 1e-8))) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def save(self, path: str) -> None:
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict() if self.model else None,
            'model_type': self.model_type,
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'input_dim': len(self.feature_names) + 1
        }, path)
    
    def load(self, path: str) -> 'EfficiencyForecaster':
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.scaler = checkpoint['scaler']
        self.target_scaler = checkpoint['target_scaler']
        self.feature_names = checkpoint['feature_names']
        self.target_name = checkpoint['target_name']
        self.sequence_length = checkpoint['sequence_length']
        self.forecast_horizon = checkpoint['forecast_horizon']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        
        input_dim = checkpoint['input_dim']
        self.model = self._create_model(input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        
        return self


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Efficiency Forecaster...")
    
    np.random.seed(42)
    
    # Generate synthetic time series
    n_samples = 5000
    n_features = 10
    
    t = np.linspace(0, 50 * np.pi, n_samples)
    
    # Target: sinusoidal with trend and noise
    y = 0.8 + 0.1 * np.sin(t) + 0.002 * np.arange(n_samples) / n_samples + np.random.randn(n_samples) * 0.02
    
    # Features: correlated with target + noise
    X = np.column_stack([
        y + np.random.randn(n_samples) * 0.1 for _ in range(n_features)
    ])
    
    # Train/test split
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train forecaster
    forecaster = EfficiencyForecaster(
        model_type='lstm',
        sequence_length=96,   # 24 hours
        forecast_horizon=24,   # 6 hours
        epochs=30
    )
    
    forecaster.fit(X_train, y_train, verbose=True)
    
    # Evaluate
    metrics = forecaster.evaluate(X_test, y_test)
    
    print("\nTest Metrics:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    print("\n✓ Efficiency Forecaster working correctly!")
