"""
Performance Tracking and Model Monitoring Module
Implements comprehensive model performance tracking, drift detection, and monitoring systems.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import os
from dataclasses import dataclass, asdict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """Data class for model performance metrics"""
    model_name: str
    symbol: str
    timestamp: datetime
    mse: float
    mae: float
    rmse: float
    r2: float
    mape: float
    directional_accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    prediction_horizon: int
    data_points: int
    training_samples: int
    validation_samples: int

@dataclass
class ModelDriftMetrics:
    """Data class for model drift detection metrics"""
    model_name: str
    symbol: str
    timestamp: datetime
    feature_drift_score: float
    prediction_drift_score: float
    data_drift_score: float
    concept_drift_score: float
    drift_threshold: float
    drift_detected: bool
    drift_severity: str
    affected_features: List[str]

class ModelPerformanceTracker:
    """
    Tracks and monitors model performance over time
    """
    
    def __init__(self, storage_path: str = "data/models/performance"):
        self.storage_path = storage_path
        self.performance_history = {}
        self.drift_history = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Performance thresholds
        self.performance_thresholds = {
            'r2_min': 0.3,
            'mape_max': 0.15,
            'directional_accuracy_min': 0.55,
            'sharpe_ratio_min': 0.5
        }
        
        # Drift thresholds
        self.drift_thresholds = {
            'feature_drift': 0.3,
            'prediction_drift': 0.2,
            'data_drift': 0.25,
            'concept_drift': 0.4
        }
    
    def calculate_performance_metrics(self, 
                                    y_true: np.ndarray, 
                                    y_pred: np.ndarray,
                                    model_name: str,
                                    symbol: str,
                                    prediction_horizon: int = 1,
                                    training_samples: int = 0,
                                    validation_samples: int = 0) -> ModelPerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            symbol: Stock symbol
            prediction_horizon: Prediction horizon in days
            training_samples: Number of training samples
            validation_samples: Number of validation samples
        
        Returns:
            ModelPerformanceMetrics object
        """
        logger.info(f"Calculating performance metrics for {model_name} on {symbol}")
        
        # Basic regression metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Directional accuracy
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(true_direction == pred_direction)
        
        # Financial metrics
        returns_true = np.diff(y_true) / y_true[:-1]
        returns_pred = np.diff(y_pred) / y_pred[:-1]
        
        sharpe_ratio = np.mean(returns_true) / (np.std(returns_true) + 1e-8) * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns_true)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Volatility
        volatility = np.std(returns_true) * np.sqrt(252)
        
        return ModelPerformanceMetrics(
            model_name=model_name,
            symbol=symbol,
            timestamp=datetime.now(),
            mse=mse,
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            directional_accuracy=directional_accuracy,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            prediction_horizon=prediction_horizon,
            data_points=len(y_true),
            training_samples=training_samples,
            validation_samples=validation_samples
        )
    
    def detect_model_drift(self, 
                          current_features: np.ndarray,
                          historical_features: np.ndarray,
                          current_predictions: np.ndarray,
                          historical_predictions: np.ndarray,
                          model_name: str,
                          symbol: str) -> ModelDriftMetrics:
        """
        Detect various types of model drift
        
        Args:
            current_features: Current feature data
            historical_features: Historical feature data
            current_predictions: Current predictions
            historical_predictions: Historical predictions
            model_name: Name of the model
            symbol: Stock symbol
        
        Returns:
            ModelDriftMetrics object
        """
        logger.info(f"Detecting drift for {model_name} on {symbol}")
        
        # Feature drift detection (using KL divergence approximation)
        feature_drift_score = self._calculate_feature_drift(current_features, historical_features)
        
        # Prediction drift detection
        prediction_drift_score = self._calculate_prediction_drift(current_predictions, historical_predictions)
        
        # Data drift detection (statistical tests)
        data_drift_score = self._calculate_data_drift(current_features, historical_features)
        
        # Concept drift detection (performance degradation)
        concept_drift_score = self._calculate_concept_drift(current_predictions, historical_predictions)
        
        # Overall drift assessment
        drift_scores = {
            'feature': feature_drift_score,
            'prediction': prediction_drift_score,
            'data': data_drift_score,
            'concept': concept_drift_score
        }
        
        max_drift_score = max(drift_scores.values())
        drift_detected = max_drift_score > self.drift_thresholds['feature_drift']
        
        # Determine drift severity
        if max_drift_score > 0.6:
            drift_severity = 'high'
        elif max_drift_score > 0.4:
            drift_severity = 'medium'
        else:
            drift_severity = 'low'
        
        # Identify affected features
        affected_features = []
        for feature_idx in range(current_features.shape[1]):
            feature_drift = self._calculate_feature_drift(
                current_features[:, feature_idx:feature_idx+1],
                historical_features[:, feature_idx:feature_idx+1]
            )
            if feature_drift > self.drift_thresholds['feature_drift']:
                affected_features.append(f"feature_{feature_idx}")
        
        return ModelDriftMetrics(
            model_name=model_name,
            symbol=symbol,
            timestamp=datetime.now(),
            feature_drift_score=feature_drift_score,
            prediction_drift_score=prediction_drift_score,
            data_drift_score=data_drift_score,
            concept_drift_score=concept_drift_score,
            drift_threshold=self.drift_thresholds['feature_drift'],
            drift_detected=drift_detected,
            drift_severity=drift_severity,
            affected_features=affected_features
        )
    
    def _calculate_feature_drift(self, current: np.ndarray, historical: np.ndarray) -> float:
        """Calculate feature drift using statistical distance"""
        try:
            # Use Wasserstein distance as a proxy for drift
            from scipy.stats import wasserstein_distance
            
            drift_scores = []
            for i in range(current.shape[1]):
                if len(current[:, i]) > 0 and len(historical[:, i]) > 0:
                    drift_score = wasserstein_distance(current[:, i], historical[:, i])
                    drift_scores.append(drift_score)
            
            return np.mean(drift_scores) if drift_scores else 0.0
        except ImportError:
            # Fallback to simple statistical difference
            current_mean = np.mean(current, axis=0)
            historical_mean = np.mean(historical, axis=0)
            current_std = np.std(current, axis=0)
            historical_std = np.std(historical, axis=0)
            
            mean_diff = np.mean(np.abs(current_mean - historical_mean))
            std_diff = np.mean(np.abs(current_std - historical_std))
            
            return (mean_diff + std_diff) / 2
    
    def _calculate_prediction_drift(self, current: np.ndarray, historical: np.ndarray) -> float:
        """Calculate prediction drift"""
        try:
            from scipy.stats import wasserstein_distance
            return wasserstein_distance(current, historical)
        except ImportError:
            # Fallback to simple difference
            current_mean = np.mean(current)
            historical_mean = np.mean(historical)
            current_std = np.std(current)
            historical_std = np.std(historical)
            
            mean_diff = abs(current_mean - historical_mean)
            std_diff = abs(current_std - historical_std)
            
            return (mean_diff + std_diff) / 2
    
    def _calculate_data_drift(self, current: np.ndarray, historical: np.ndarray) -> float:
        """Calculate data drift using Kolmogorov-Smirnov test"""
        try:
            from scipy.stats import ks_2samp
            
            drift_scores = []
            for i in range(current.shape[1]):
                if len(current[:, i]) > 0 and len(historical[:, i]) > 0:
                    statistic, _ = ks_2samp(historical[:, i], current[:, i])
                    drift_scores.append(statistic)
            
            return np.mean(drift_scores) if drift_scores else 0.0
        except ImportError:
            # Fallback to simple statistical difference
            return self._calculate_feature_drift(current, historical)
    
    def _calculate_concept_drift(self, current: np.ndarray, historical: np.ndarray) -> float:
        """Calculate concept drift based on performance degradation"""
        # Simple concept drift detection based on prediction distribution changes
        current_mean = np.mean(current)
        historical_mean = np.mean(historical)
        current_std = np.std(current)
        historical_std = np.std(historical)
        
        # Calculate relative change in prediction characteristics
        mean_change = abs(current_mean - historical_mean) / (abs(historical_mean) + 1e-8)
        std_change = abs(current_std - historical_std) / (historical_std + 1e-8)
        
        return (mean_change + std_change) / 2
    
    def save_performance_metrics(self, metrics: ModelPerformanceMetrics):
        """Save performance metrics to storage"""
        try:
            model_key = f"{metrics.model_name}_{metrics.symbol}"
            
            if model_key not in self.performance_history:
                self.performance_history[model_key] = []
            
            self.performance_history[model_key].append(asdict(metrics))
            
            # Save to file
            file_path = os.path.join(self.storage_path, f"performance_{model_key}.json")
            with open(file_path, 'w') as f:
                json.dump(self.performance_history[model_key], f, default=str)
            
            logger.info(f"Performance metrics saved for {model_key}")
            
        except Exception as e:
            logger.error(f"Error saving performance metrics: {str(e)}")
    
    def save_drift_metrics(self, metrics: ModelDriftMetrics):
        """Save drift metrics to storage"""
        try:
            model_key = f"{metrics.model_name}_{metrics.symbol}"
            
            if model_key not in self.drift_history:
                self.drift_history[model_key] = []
            
            self.drift_history[model_key].append(asdict(metrics))
            
            # Save to file
            file_path = os.path.join(self.storage_path, f"drift_{model_key}.json")
            with open(file_path, 'w') as f:
                json.dump(self.drift_history[model_key], f, default=str)
            
            logger.info(f"Drift metrics saved for {model_key}")
            
        except Exception as e:
            logger.error(f"Error saving drift metrics: {str(e)}")
    
    def load_performance_history(self, model_name: str, symbol: str) -> List[Dict]:
        """Load performance history for a model"""
        try:
            model_key = f"{model_name}_{symbol}"
            file_path = os.path.join(self.storage_path, f"performance_{model_key}.json")
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error loading performance history: {str(e)}")
            return []
    
    def load_drift_history(self, model_name: str, symbol: str) -> List[Dict]:
        """Load drift history for a model"""
        try:
            model_key = f"{model_name}_{symbol}"
            file_path = os.path.join(self.storage_path, f"drift_{model_key}.json")
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error loading drift history: {str(e)}")
            return []
    
    def get_performance_summary(self, model_name: str, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for the last N days"""
        try:
            performance_history = self.load_performance_history(model_name, symbol)
            
            if not performance_history:
                return {}
            
            # Filter recent data
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_metrics = [
                m for m in performance_history 
                if datetime.fromisoformat(m['timestamp'].replace('Z', '+00:00')) > cutoff_date
            ]
            
            if not recent_metrics:
                return {}
            
            # Calculate summary statistics
            r2_scores = [m['r2'] for m in recent_metrics]
            mape_scores = [m['mape'] for m in recent_metrics]
            directional_accuracies = [m['directional_accuracy'] for m in recent_metrics]
            sharpe_ratios = [m['sharpe_ratio'] for m in recent_metrics]
            
            return {
                'model_name': model_name,
                'symbol': symbol,
                'period_days': days,
                'total_evaluations': len(recent_metrics),
                'avg_r2': np.mean(r2_scores),
                'min_r2': np.min(r2_scores),
                'max_r2': np.max(r2_scores),
                'avg_mape': np.mean(mape_scores),
                'max_mape': np.max(mape_scores),
                'avg_directional_accuracy': np.mean(directional_accuracies),
                'min_directional_accuracy': np.min(directional_accuracies),
                'avg_sharpe_ratio': np.mean(sharpe_ratios),
                'min_sharpe_ratio': np.min(sharpe_ratios),
                'performance_trend': self._calculate_performance_trend(r2_scores),
                'last_evaluation': recent_metrics[-1]['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {}
    
    def _calculate_performance_trend(self, scores: List[float]) -> str:
        """Calculate performance trend"""
        if len(scores) < 2:
            return 'stable'
        
        # Simple linear trend calculation
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def get_drift_summary(self, model_name: str, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get drift summary for the last N days"""
        try:
            drift_history = self.load_drift_history(model_name, symbol)
            
            if not drift_history:
                return {}
            
            # Filter recent data
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_drift = [
                d for d in drift_history 
                if datetime.fromisoformat(d['timestamp'].replace('Z', '+00:00')) > cutoff_date
            ]
            
            if not recent_drift:
                return {}
            
            # Calculate summary statistics
            feature_drifts = [d['feature_drift_score'] for d in recent_drift]
            prediction_drifts = [d['prediction_drift_score'] for d in recent_drift]
            data_drifts = [d['data_drift_score'] for d in recent_drift]
            concept_drifts = [d['concept_drift_score'] for d in recent_drift]
            
            drift_detected_count = sum(1 for d in recent_drift if d['drift_detected'])
            
            return {
                'model_name': model_name,
                'symbol': symbol,
                'period_days': days,
                'total_evaluations': len(recent_drift),
                'drift_detected_count': drift_detected_count,
                'drift_detection_rate': drift_detected_count / len(recent_drift),
                'avg_feature_drift': np.mean(feature_drifts),
                'max_feature_drift': np.max(feature_drifts),
                'avg_prediction_drift': np.mean(prediction_drifts),
                'max_prediction_drift': np.max(prediction_drifts),
                'avg_data_drift': np.mean(data_drifts),
                'max_data_drift': np.max(data_drifts),
                'avg_concept_drift': np.mean(concept_drifts),
                'max_concept_drift': np.max(concept_drifts),
                'last_evaluation': recent_drift[-1]['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error getting drift summary: {str(e)}")
            return {}
    
    def check_performance_alerts(self, model_name: str, symbol: str) -> List[Dict[str, Any]]:
        """Check for performance alerts and warnings"""
        alerts = []
        
        try:
            performance_summary = self.get_performance_summary(model_name, symbol, 7)  # Last 7 days
            
            if not performance_summary:
                return alerts
            
            # Check performance thresholds
            if performance_summary['avg_r2'] < self.performance_thresholds['r2_min']:
                alerts.append({
                    'type': 'performance',
                    'severity': 'high',
                    'message': f"Low R² score: {performance_summary['avg_r2']:.3f}",
                    'threshold': self.performance_thresholds['r2_min']
                })
            
            if performance_summary['avg_mape'] > self.performance_thresholds['mape_max']:
                alerts.append({
                    'type': 'performance',
                    'severity': 'high',
                    'message': f"High MAPE: {performance_summary['avg_mape']:.3f}",
                    'threshold': self.performance_thresholds['mape_max']
                })
            
            if performance_summary['avg_directional_accuracy'] < self.performance_thresholds['directional_accuracy_min']:
                alerts.append({
                    'type': 'performance',
                    'severity': 'medium',
                    'message': f"Low directional accuracy: {performance_summary['avg_directional_accuracy']:.3f}",
                    'threshold': self.performance_thresholds['directional_accuracy_min']
                })
            
            if performance_summary['avg_sharpe_ratio'] < self.performance_thresholds['sharpe_ratio_min']:
                alerts.append({
                    'type': 'performance',
                    'severity': 'medium',
                    'message': f"Low Sharpe ratio: {performance_summary['avg_sharpe_ratio']:.3f}",
                    'threshold': self.performance_thresholds['sharpe_ratio_min']
                })
            
            # Check drift alerts
            drift_summary = self.get_drift_summary(model_name, symbol, 7)
            
            if drift_summary and drift_summary['drift_detection_rate'] > 0.5:
                alerts.append({
                    'type': 'drift',
                    'severity': 'high',
                    'message': f"High drift detection rate: {drift_summary['drift_detection_rate']:.2f}",
                    'threshold': 0.5
                })
            
            if drift_summary and drift_summary['max_feature_drift'] > self.drift_thresholds['feature_drift']:
                alerts.append({
                    'type': 'drift',
                    'severity': 'medium',
                    'message': f"Feature drift detected: {drift_summary['max_feature_drift']:.3f}",
                    'threshold': self.drift_thresholds['feature_drift']
                })
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {str(e)}")
        
        return alerts
    
    def get_model_health_score(self, model_name: str, symbol: str) -> Dict[str, Any]:
        """Calculate overall model health score"""
        try:
            performance_summary = self.get_performance_summary(model_name, symbol, 30)
            drift_summary = self.get_drift_summary(model_name, symbol, 30)
            
            if not performance_summary:
                return {'health_score': 0, 'status': 'unknown', 'details': {}}
            
            # Calculate health score components
            r2_score = min(1.0, max(0.0, performance_summary['avg_r2']))
            mape_score = max(0.0, 1.0 - performance_summary['avg_mape'] / 0.2)  # Normalize MAPE
            directional_score = max(0.0, performance_summary['avg_directional_accuracy'])
            sharpe_score = max(0.0, min(1.0, performance_summary['avg_sharpe_ratio'] / 2.0))
            
            # Drift penalty
            drift_penalty = 0.0
            if drift_summary:
                drift_penalty = min(0.3, drift_summary['drift_detection_rate'] * 0.3)
            
            # Overall health score
            health_score = (r2_score * 0.3 + mape_score * 0.25 + 
                           directional_score * 0.25 + sharpe_score * 0.2) - drift_penalty
            
            # Determine status
            if health_score >= 0.8:
                status = 'excellent'
            elif health_score >= 0.6:
                status = 'good'
            elif health_score >= 0.4:
                status = 'fair'
            elif health_score >= 0.2:
                status = 'poor'
            else:
                status = 'critical'
            
            return {
                'health_score': health_score,
                'status': status,
                'details': {
                    'r2_score': r2_score,
                    'mape_score': mape_score,
                    'directional_score': directional_score,
                    'sharpe_score': sharpe_score,
                    'drift_penalty': drift_penalty,
                    'performance_summary': performance_summary,
                    'drift_summary': drift_summary
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating model health score: {str(e)}")
            return {'health_score': 0, 'status': 'error', 'details': {'error': str(e)}}

# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = ModelPerformanceTracker()
    
    # Create sample data
    np.random.seed(42)
    y_true = np.random.randn(100) * 10 + 100
    y_pred = y_true + np.random.randn(100) * 2
    
    # Calculate performance metrics
    metrics = tracker.calculate_performance_metrics(
        y_true, y_pred, 'lstm', 'RELIANCE', 1, 1000, 200
    )
    
    print(f"Performance Metrics: {metrics}")
    
    # Save metrics
    tracker.save_performance_metrics(metrics)
    
    # Get performance summary
    summary = tracker.get_performance_summary('lstm', 'RELIANCE', 30)
    print(f"Performance Summary: {summary}")
    
    # Get health score
    health = tracker.get_model_health_score('lstm', 'RELIANCE')
    print(f"Model Health: {health}")
    
    print("Performance tracking completed!")

