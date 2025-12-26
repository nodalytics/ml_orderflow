"""
Standalone Anomaly Detection Service

Provides real-time anomaly detection without LLM overhead.
Useful for automated systems, dashboards, and high-frequency monitoring.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from ml_orderflow.utils.initializer import logger_instance

logger = logger_instance.get_logger()


class AnomalyDetector:
    """
    Standalone anomaly detector using statistical methods.
    Detects price spikes, volume anomalies, volatility changes, and trend reversals.
    """
    
    def __init__(self, z_threshold: float = 3.0):
        """
        Initialize anomaly detector.
        
        Args:
            z_threshold: Z-score threshold for anomaly detection
        """
        self.z_threshold = z_threshold
    
    def detect_all(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect all types of anomalies in the data.
        
        Args:
            data: DataFrame with market data
        
        Returns:
            List of anomaly dictionaries with type, severity, and details
        """
        anomalies = []
        
        # Price spike detection
        price_anomalies = self._detect_price_spikes(data)
        anomalies.extend(price_anomalies)
        
        # Volume anomalies
        volume_anomalies = self._detect_volume_anomalies(data)
        anomalies.extend(volume_anomalies)
        
        # Volatility anomalies
        volatility_anomalies = self._detect_volatility_anomalies(data)
        anomalies.extend(volatility_anomalies)
        
        # Trend reversals
        reversal_anomalies = self._detect_trend_reversals(data)
        anomalies.extend(reversal_anomalies)
        
        # Sort by severity
        anomalies.sort(key=lambda x: x['severity'], reverse=True)
        
        return anomalies
    
    def _detect_price_spikes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect extreme price predictions"""
        anomalies = []
        
        if 'y_pred' not in df.columns or df['y_pred'].isna().all():
            return anomalies
        
        predictions = df['y_pred'].dropna()
        if len(predictions) < 3:
            return anomalies
        
        mean = predictions.mean()
        std = predictions.std()
        
        if std == 0:
            return anomalies
        
        z_scores = (predictions - mean) / std
        extreme_indices = z_scores.abs() > self.z_threshold
        
        if extreme_indices.any():
            for idx in predictions[extreme_indices].index:
                value = predictions.loc[idx]
                z_score = z_scores.loc[idx]
                
                anomalies.append({
                    'type': 'price_spike',
                    'severity': min(abs(z_score) / self.z_threshold, 3.0),  # Cap at 3.0
                    'direction': 'upward' if value > 0 else 'downward',
                    'value': float(value),
                    'z_score': float(z_score),
                    'timestamp': str(idx) if hasattr(idx, '__str__') else None,
                    'message': f"Extreme {('upward' if value > 0 else 'downward')} price prediction detected"
                })
        
        return anomalies
    
    def _detect_volume_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect unusual volume patterns"""
        anomalies = []
        
        if 'volume' not in df.columns or df['volume'].isna().all():
            return anomalies
        
        volumes = df['volume'].dropna()
        if len(volumes) < 3:
            return anomalies
        
        mean_vol = volumes.mean()
        std_vol = volumes.std()
        
        if std_vol == 0:
            return anomalies
        
        latest_vol = volumes.iloc[-1]
        z_score = (latest_vol - mean_vol) / std_vol
        
        if abs(z_score) > self.z_threshold:
            anomalies.append({
                'type': 'volume_anomaly',
                'severity': min(abs(z_score) / self.z_threshold, 3.0),
                'direction': 'spike' if z_score > 0 else 'drop',
                'value': float(latest_vol),
                'z_score': float(z_score),
                'deviation_percent': float((z_score * std_vol / mean_vol) * 100),
                'timestamp': str(volumes.index[-1]) if hasattr(volumes.index[-1], '__str__') else None,
                'message': f"Unusual {'high' if z_score > 0 else 'low'} volume detected"
            })
        
        return anomalies
    
    def _detect_volatility_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect elevated volatility"""
        anomalies = []
        
        vol_cols = [col for col in df.columns if 'volatility' in col.lower()]
        if not vol_cols:
            return anomalies
        
        volatilities = df[vol_cols[0]].dropna()
        if len(volatilities) < 3:
            return anomalies
        
        mean_volatility = volatilities.mean()
        std_volatility = volatilities.std()
        
        if std_volatility == 0:
            return anomalies
        
        latest_volatility = volatilities.iloc[-1]
        z_score = (latest_volatility - mean_volatility) / std_volatility
        
        # Use lower threshold for volatility (2.5 instead of 3.0)
        if z_score > 2.5:
            anomalies.append({
                'type': 'volatility_spike',
                'severity': min(z_score / 2.5, 3.0),
                'value': float(latest_volatility),
                'z_score': float(z_score),
                'timestamp': str(volatilities.index[-1]) if hasattr(volatilities.index[-1], '__str__') else None,
                'message': "Market volatility is significantly elevated"
            })
        
        return anomalies
    
    def _detect_trend_reversals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect sharp trend reversals"""
        anomalies = []
        
        if 'y_pred' not in df.columns or len(df) < 3:
            return anomalies
        
        recent_preds = df['y_pred'].dropna().tail(3)
        if len(recent_preds) < 3:
            return anomalies
        
        signs = (recent_preds > 0).astype(int)
        
        # Check for sign change with magnitude increase
        if signs.iloc[0] != signs.iloc[-1]:
            magnitude_ratio = abs(recent_preds.iloc[-1]) / (abs(recent_preds.iloc[0]) + 1e-9)
            
            if magnitude_ratio > 2.0:
                direction = "bullish_to_bearish" if signs.iloc[0] > signs.iloc[-1] else "bearish_to_bullish"
                
                anomalies.append({
                    'type': 'trend_reversal',
                    'severity': min(magnitude_ratio / 2.0, 3.0),
                    'direction': direction,
                    'magnitude_ratio': float(magnitude_ratio),
                    'from_value': float(recent_preds.iloc[0]),
                    'to_value': float(recent_preds.iloc[-1]),
                    'timestamp': str(recent_preds.index[-1]) if hasattr(recent_preds.index[-1], '__str__') else None,
                    'message': f"Sharp trend reversal detected ({direction.replace('_', ' ')})"
                })
        
        return anomalies
    
    def get_summary_stats(self, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics of detected anomalies"""
        if not anomalies:
            return {
                'total_anomalies': 0,
                'max_severity': 0.0,
                'types': {},
                'critical_count': 0
            }
        
        types_count = {}
        for anomaly in anomalies:
            atype = anomaly['type']
            types_count[atype] = types_count.get(atype, 0) + 1
        
        critical_count = sum(1 for a in anomalies if a['severity'] >= 2.0)
        
        return {
            'total_anomalies': len(anomalies),
            'max_severity': max(a['severity'] for a in anomalies),
            'types': types_count,
            'critical_count': critical_count,
            'timestamp': datetime.utcnow().isoformat()
        }
