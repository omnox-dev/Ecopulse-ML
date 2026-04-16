"""
Utility functions for the predictive maintenance system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta


def calculate_health_score(
    anomaly_score: float,
    failure_probability: float,
    performance_ratio: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate overall asset health score (0-100).
    
    Args:
        anomaly_score: Anomaly detection score (0-1, higher = worse)
        failure_probability: Failure probability (0-1, higher = worse)
        performance_ratio: Performance ratio (0-1, higher = better)
        weights: Optional custom weights
        
    Returns:
        Health score (0-100, higher = healthier)
    """
    weights = weights or {
        'anomaly': 0.25,
        'failure': 0.35,
        'performance': 0.40
    }
    
    # Convert scores to health contributions (0-100 scale)
    anomaly_health = (1 - anomaly_score) * 100
    failure_health = (1 - failure_probability) * 100
    performance_health = performance_ratio * 100
    
    # Weighted average
    health_score = (
        weights['anomaly'] * anomaly_health +
        weights['failure'] * failure_health +
        weights['performance'] * performance_health
    )
    
    return np.clip(health_score, 0, 100)


def get_health_status(health_score: float) -> Dict[str, Any]:
    """
    Get health status category and color.
    
    Args:
        health_score: Health score (0-100)
        
    Returns:
        Dictionary with status info
    """
    if health_score >= 85:
        return {
            'status': 'Excellent',
            'color': '#22c55e',  # Green
            'icon': '✓',
            'recommendation': 'System operating normally. Continue regular monitoring.'
        }
    elif health_score >= 70:
        return {
            'status': 'Good',
            'color': '#84cc16',  # Lime
            'icon': '●',
            'recommendation': 'Minor variations detected. Consider inspection during next maintenance window.'
        }
    elif health_score >= 50:
        return {
            'status': 'Warning',
            'color': '#eab308',  # Yellow
            'icon': '⚠',
            'recommendation': 'Performance degradation detected. Schedule maintenance soon.'
        }
    elif health_score >= 30:
        return {
            'status': 'Poor',
            'color': '#f97316',  # Orange
            'icon': '!',
            'recommendation': 'Significant issues detected. Immediate inspection recommended.'
        }
    else:
        return {
            'status': 'Critical',
            'color': '#ef4444',  # Red
            'icon': '✗',
            'recommendation': 'Critical failure risk. Immediate maintenance required.'
        }


def generate_alert(
    asset_id: str,
    alert_type: str,
    severity: str,
    value: float,
    threshold: float,
    message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate an alert object.
    
    Args:
        asset_id: ID of the asset
        alert_type: Type of alert (anomaly, failure, efficiency)
        severity: Alert severity (info, warning, critical)
        value: Current value that triggered alert
        threshold: Threshold that was exceeded
        message: Optional custom message
        
    Returns:
        Alert dictionary
    """
    severity_config = {
        'info': {'color': '#3b82f6', 'priority': 1},
        'warning': {'color': '#f59e0b', 'priority': 2},
        'critical': {'color': '#ef4444', 'priority': 3}
    }
    
    default_messages = {
        'anomaly': f"Anomaly detected with score {value:.2f} (threshold: {threshold:.2f})",
        'failure': f"Failure probability {value:.1%} exceeds threshold {threshold:.1%}",
        'efficiency': f"Efficiency dropped to {value:.1%} (threshold: {threshold:.1%})"
    }
    
    return {
        'id': f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_{asset_id}",
        'asset_id': asset_id,
        'type': alert_type,
        'severity': severity,
        'priority': severity_config.get(severity, {}).get('priority', 1),
        'color': severity_config.get(severity, {}).get('color', '#6b7280'),
        'value': value,
        'threshold': threshold,
        'message': message or default_messages.get(alert_type, 'Alert triggered'),
        'timestamp': datetime.now().isoformat(),
        'acknowledged': False
    }


def check_alert_conditions(
    anomaly_score: float,
    failure_probability: float,
    efficiency: float,
    asset_id: str = "panel_001"
) -> List[Dict[str, Any]]:
    """
    Check all alert conditions and generate alerts.
    
    Args:
        anomaly_score: Anomaly score (0-1)
        failure_probability: Failure probability (0-1)
        efficiency: Current efficiency/performance ratio (0-1)
        asset_id: Asset identifier
        
    Returns:
        List of triggered alerts
    """
    alerts = []
    
    # Anomaly alerts
    if anomaly_score > 0.8:
        alerts.append(generate_alert(
            asset_id, 'anomaly', 'critical',
            anomaly_score, 0.8,
            "Critical anomaly detected! Immediate inspection required."
        ))
    elif anomaly_score > 0.6:
        alerts.append(generate_alert(
            asset_id, 'anomaly', 'warning',
            anomaly_score, 0.6
        ))
    
    # Failure probability alerts
    if failure_probability > 0.7:
        alerts.append(generate_alert(
            asset_id, 'failure', 'critical',
            failure_probability, 0.7,
            "High failure probability! Schedule immediate maintenance."
        ))
    elif failure_probability > 0.4:
        alerts.append(generate_alert(
            asset_id, 'failure', 'warning',
            failure_probability, 0.4
        ))
    
    # Efficiency alerts
    if efficiency < 0.5:
        alerts.append(generate_alert(
            asset_id, 'efficiency', 'critical',
            efficiency, 0.5,
            "Severe efficiency degradation detected!"
        ))
    elif efficiency < 0.7:
        alerts.append(generate_alert(
            asset_id, 'efficiency', 'warning',
            efficiency, 0.7
        ))
    
    return alerts


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    else:
        return f"{seconds / 86400:.1f} days"


def resample_data(
    df: pd.DataFrame,
    freq: str = '1H',
    agg_funcs: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Resample time series data to different frequency.
    
    Args:
        df: DataFrame with timestamp column
        freq: Target frequency (e.g., '1H', '1D')
        agg_funcs: Aggregation functions per column
        
    Returns:
        Resampled DataFrame
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Default aggregations
    default_agg = {
        'power_output': 'mean',
        'voltage': 'mean',
        'current': 'mean',
        'panel_temperature': 'mean',
        'inverter_temperature': 'mean',
        'performance_ratio': 'mean',
        'irradiance': 'mean',
        'is_fault': 'max'
    }
    
    agg_funcs = agg_funcs or default_agg
    
    # Filter to available columns
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
    
    df_resampled = df[list(agg_funcs.keys())].resample(freq).agg(agg_funcs)
    
    return df_resampled.reset_index()


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, max_alerts: int = 100):
        self.alerts: List[Dict[str, Any]] = []
        self.max_alerts = max_alerts
    
    def add_alert(self, alert: Dict[str, Any]) -> None:
        """Add a new alert."""
        self.alerts.insert(0, alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[:self.max_alerts]
    
    def add_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """Add multiple alerts."""
        for alert in alerts:
            self.add_alert(alert)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all unacknowledged alerts."""
        return [a for a in self.alerts if not a['acknowledged']]
    
    def get_critical_alerts(self) -> List[Dict[str, Any]]:
        """Get critical severity alerts."""
        return [a for a in self.alerts if a['severity'] == 'critical' and not a['acknowledged']]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                return True
        return False
    
    def clear_old_alerts(self, hours: int = 24) -> int:
        """Clear alerts older than specified hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        original_count = len(self.alerts)
        
        self.alerts = [
            a for a in self.alerts
            if datetime.fromisoformat(a['timestamp']) > cutoff
        ]
        
        return original_count - len(self.alerts)
    
    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of alert counts by severity."""
        active = self.get_active_alerts()
        return {
            'total': len(self.alerts),
            'active': len(active),
            'critical': len([a for a in active if a['severity'] == 'critical']),
            'warning': len([a for a in active if a['severity'] == 'warning']),
            'info': len([a for a in active if a['severity'] == 'info'])
        }
