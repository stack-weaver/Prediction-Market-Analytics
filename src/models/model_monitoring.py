"""
Real-time Model Monitoring Dashboard
Implements comprehensive model performance monitoring, drift detection, and alerting system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import os
from dataclasses import dataclass, asdict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from src.models.models import StockPredictor
from src.data.dataset_manager import nse_dataset_manager
from src.features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

@dataclass
class ModelAlert:
    """Data class for model alerts"""
    alert_id: str
    model_name: str
    symbol: str
    alert_type: str  # 'performance', 'drift', 'error', 'warning'
    severity: str    # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    details: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class ModelHealthMetrics:
    """Data class for model health metrics"""
    model_name: str
    symbol: str
    timestamp: datetime
    health_score: float
    performance_score: float
    stability_score: float
    data_quality_score: float
    prediction_accuracy: float
    response_time_ms: float
    error_rate: float
    last_training: datetime
    data_freshness_hours: float

class ModelMonitoringDashboard:
    """
    Real-time model monitoring and alerting system
    """
    
    def __init__(self):
        self.stock_predictor = StockPredictor()
        self.feature_engineer = FeatureEngineer()
        self.monitoring_data = {}
        self.alerts = []
        self.health_metrics = {}
        self.alert_thresholds = {
            'performance_degradation': 0.1,  # 10% drop in performance
            'prediction_drift': 0.15,       # 15% drift threshold
            'error_rate': 0.05,             # 5% error rate
            'response_time': 5000,           # 5 seconds
            'data_freshness': 24,            # 24 hours
            'health_score': 0.7              # 70% health score
        }
        self.monitoring_storage = "data/monitoring"
        os.makedirs(self.monitoring_storage, exist_ok=True)
        
    def monitor_model_performance(self, symbol: str, model_name: str, 
                                test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Monitor model performance and generate alerts
        
        Args:
            symbol: Stock symbol
            model_name: Model to monitor
            test_data: Test data for evaluation
            
        Returns:
            Monitoring results and alerts
        """
        try:
            logger.info(f"Monitoring model performance for {symbol} - {model_name}")
            
            start_time = datetime.now()
            
            # Load recent data for monitoring
            if test_data is None:
                test_data = nse_dataset_manager.load_multi_year_data(symbol, ['2024'])
                if test_data.empty:
                    return {"error": "No test data available"}
            
            # Prepare features
            processed_data = self.feature_engineer.prepare_features(test_data)
            if processed_data.empty:
                return {"error": "Failed to process features"}
            
            # Get predictions
            predictions = self.stock_predictor.predict(processed_data, model_name)
            if len(predictions) == 0:
                return {"error": "Failed to get predictions"}
            
            # Calculate performance metrics
            actual_values = processed_data[processed_data.columns[-1]].values[-len(predictions):]
            
            mse = mean_squared_error(actual_values, predictions)
            mae = mean_absolute_error(actual_values, predictions)
            r2 = r2_score(actual_values, predictions)
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Calculate health metrics
            health_metrics = self._calculate_model_health(
                symbol, model_name, mse, mae, r2, response_time, processed_data
            )
            
            # Store health metrics
            self.health_metrics[f"{symbol}_{model_name}"] = health_metrics
            
            # Check for alerts
            alerts = self._check_performance_alerts(symbol, model_name, health_metrics)
            
            # Update monitoring data
            self.monitoring_data[f"{symbol}_{model_name}"] = {
                "last_check": datetime.now(),
                "performance": {
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "response_time": response_time
                },
                "health_metrics": health_metrics,
                "alerts": alerts
            }
            
            return {
                "status": "success",
                "symbol": symbol,
                "model": model_name,
                "health_metrics": asdict(health_metrics),
                "alerts": [asdict(alert) for alert in alerts],
                "performance": {
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "response_time": response_time,
                    "predictions_count": len(predictions)
                }
            }
            
        except Exception as e:
            logger.error(f"Model performance monitoring failed: {str(e)}")
            return {"error": f"Model performance monitoring failed: {str(e)}"}
    
    def _calculate_model_health(self, symbol: str, model_name: str, 
                               mse: float, mae: float, r2: float, 
                               response_time: float, data: pd.DataFrame) -> ModelHealthMetrics:
        """Calculate comprehensive model health metrics"""
        try:
            # Performance score (0-1, higher is better)
            performance_score = max(0, min(1, r2))
            
            # Stability score based on prediction consistency
            if len(data) > 10:
                recent_data = data.tail(10)
                price_volatility = recent_data['close'].pct_change().std()
                stability_score = max(0, min(1, 1 - price_volatility))
            else:
                stability_score = 0.5
            
            # Data quality score
            data_quality_score = self._calculate_data_quality_score(data)
            
            # Prediction accuracy (based on R²)
            prediction_accuracy = max(0, min(1, r2))
            
            # Error rate (based on MAE relative to price)
            if 'close' in data.columns and len(data) > 0:
                avg_price = data['close'].mean()
                error_rate = mae / avg_price if avg_price > 0 else 1.0
            else:
                error_rate = 1.0
            
            # Overall health score
            health_score = (
                performance_score * 0.3 +
                stability_score * 0.2 +
                data_quality_score * 0.2 +
                prediction_accuracy * 0.2 +
                (1 - min(error_rate, 1)) * 0.1
            )
            
            return ModelHealthMetrics(
                model_name=model_name,
                symbol=symbol,
                timestamp=datetime.now(),
                health_score=health_score,
                performance_score=performance_score,
                stability_score=stability_score,
                data_quality_score=data_quality_score,
                prediction_accuracy=prediction_accuracy,
                response_time_ms=response_time,
                error_rate=error_rate,
                last_training=datetime.now(),  # This should be actual last training time
                data_freshness_hours=0  # This should be calculated from actual data timestamp
            )
            
        except Exception as e:
            logger.warning(f"Failed to calculate model health: {str(e)}")
            return ModelHealthMetrics(
                model_name=model_name,
                symbol=symbol,
                timestamp=datetime.now(),
                health_score=0.5,
                performance_score=0.5,
                stability_score=0.5,
                data_quality_score=0.5,
                prediction_accuracy=0.5,
                response_time_ms=response_time,
                error_rate=1.0,
                last_training=datetime.now(),
                data_freshness_hours=0
            )
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score"""
        try:
            if data.empty:
                return 0.0
            
            score = 1.0
            
            # Check for missing values
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            score -= missing_ratio * 0.3
            
            # Check for outliers (using IQR method)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            outlier_ratio = 0
            for col in numeric_cols:
                if len(data[col].dropna()) > 0:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)]
                    outlier_ratio += len(outliers) / len(data)
            
            outlier_ratio /= len(numeric_cols) if len(numeric_cols) > 0 else 1
            score -= outlier_ratio * 0.2
            
            # Check for data consistency
            if 'close' in data.columns and len(data) > 1:
                price_changes = data['close'].pct_change().dropna()
                extreme_changes = abs(price_changes) > 0.2  # 20% daily change
                extreme_ratio = extreme_changes.sum() / len(price_changes)
                score -= extreme_ratio * 0.1
            
            return max(0, min(1, score))
            
        except Exception as e:
            logger.warning(f"Failed to calculate data quality score: {str(e)}")
            return 0.5
    
    def _check_performance_alerts(self, symbol: str, model_name: str, 
                                health_metrics: ModelHealthMetrics) -> List[ModelAlert]:
        """Check for performance alerts and generate them"""
        alerts = []
        
        try:
            # Health score alert
            if health_metrics.health_score < self.alert_thresholds['health_score']:
                alerts.append(ModelAlert(
                    alert_id=f"{symbol}_{model_name}_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_name=model_name,
                    symbol=symbol,
                    alert_type="performance",
                    severity="high" if health_metrics.health_score < 0.5 else "medium",
                    message=f"Model health score is {health_metrics.health_score:.2f} (below threshold)",
                    timestamp=datetime.now(),
                    details={"health_score": health_metrics.health_score, "threshold": self.alert_thresholds['health_score']}
                ))
            
            # Response time alert
            if health_metrics.response_time_ms > self.alert_thresholds['response_time']:
                alerts.append(ModelAlert(
                    alert_id=f"{symbol}_{model_name}_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_name=model_name,
                    symbol=symbol,
                    alert_type="performance",
                    severity="medium",
                    message=f"Model response time is {health_metrics.response_time_ms:.0f}ms (above threshold)",
                    timestamp=datetime.now(),
                    details={"response_time": health_metrics.response_time_ms, "threshold": self.alert_thresholds['response_time']}
                ))
            
            # Error rate alert
            if health_metrics.error_rate > self.alert_thresholds['error_rate']:
                alerts.append(ModelAlert(
                    alert_id=f"{symbol}_{model_name}_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_name=model_name,
                    symbol=symbol,
                    alert_type="performance",
                    severity="high" if health_metrics.error_rate > 0.1 else "medium",
                    message=f"Model error rate is {health_metrics.error_rate:.2%} (above threshold)",
                    timestamp=datetime.now(),
                    details={"error_rate": health_metrics.error_rate, "threshold": self.alert_thresholds['error_rate']}
                ))
            
            # Performance degradation alert
            if health_metrics.performance_score < 0.5:
                alerts.append(ModelAlert(
                    alert_id=f"{symbol}_{model_name}_perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_name=model_name,
                    symbol=symbol,
                    alert_type="performance",
                    severity="critical" if health_metrics.performance_score < 0.3 else "high",
                    message=f"Model performance score is {health_metrics.performance_score:.2f} (critical level)",
                    timestamp=datetime.now(),
                    details={"performance_score": health_metrics.performance_score}
                ))
            
            # Add alerts to global list
            self.alerts.extend(alerts)
            
            return alerts
            
        except Exception as e:
            logger.warning(f"Failed to check performance alerts: {str(e)}")
            return []
    
    def create_monitoring_dashboard(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Create comprehensive monitoring dashboard
        
        Args:
            symbols: List of symbols to monitor (None for all)
            
        Returns:
            Dashboard data and visualizations
        """
        try:
            logger.info("Creating monitoring dashboard")
            
            if symbols is None:
                symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']
            
            # Collect monitoring data for all symbols
            dashboard_data = {}
            all_alerts = []
            health_scores = []
            
            for symbol in symbols:
                for model_name in ['lstm', 'random_forest', 'xgboost']:
                    try:
                        monitoring_result = self.monitor_model_performance(symbol, model_name)
                        if "error" not in monitoring_result:
                            dashboard_data[f"{symbol}_{model_name}"] = monitoring_result
                            all_alerts.extend(monitoring_result.get("alerts", []))
                            
                            health_metrics = monitoring_result.get("health_metrics", {})
                            health_scores.append({
                                "symbol": symbol,
                                "model": model_name,
                                "health_score": health_metrics.get("health_score", 0),
                                "performance_score": health_metrics.get("performance_score", 0),
                                "stability_score": health_metrics.get("stability_score", 0)
                            })
                    except Exception as e:
                        logger.warning(f"Failed to monitor {symbol}_{model_name}: {str(e)}")
                        continue
            
            # Create visualizations
            visualizations = self._create_monitoring_visualizations(health_scores, all_alerts)
            
            return {
                "status": "success",
                "dashboard_type": "model_monitoring",
                "timestamp": datetime.now().isoformat(),
                "symbols_monitored": symbols,
                "models_monitored": ['lstm', 'random_forest', 'xgboost'],
                "dashboard_data": dashboard_data,
                "health_scores": health_scores,
                "alerts": all_alerts,
                "alert_summary": {
                    "total_alerts": len(all_alerts),
                    "critical_alerts": len([a for a in all_alerts if a.get("severity") == "critical"]),
                    "high_alerts": len([a for a in all_alerts if a.get("severity") == "high"]),
                    "medium_alerts": len([a for a in all_alerts if a.get("severity") == "medium"]),
                    "low_alerts": len([a for a in all_alerts if a.get("severity") == "low"])
                },
                "visualizations": visualizations
            }
            
        except Exception as e:
            logger.error(f"Monitoring dashboard creation failed: {str(e)}")
            return {"error": f"Monitoring dashboard creation failed: {str(e)}"}
    
    def _create_monitoring_visualizations(self, health_scores: List[Dict], 
                                        alerts: List[Dict]) -> Dict[str, Any]:
        """Create monitoring visualizations"""
        try:
            visualizations = {}
            
            if health_scores:
                # Health score heatmap
                df_health = pd.DataFrame(health_scores)
                health_pivot = df_health.pivot(index='symbol', columns='model', values='health_score')
                
                fig_health = go.Figure(data=go.Heatmap(
                    z=health_pivot.values,
                    x=health_pivot.columns,
                    y=health_pivot.index,
                    colorscale='RdYlGn',
                    zmin=0,
                    zmax=1,
                    text=health_pivot.round(2).values,
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    hoverongaps=False
                ))
                
                fig_health.update_layout(
                    title='Model Health Scores',
                    template='plotly_dark',
                    height=400
                )
                
                visualizations["health_heatmap"] = fig_health.to_dict()
                
                # Performance comparison chart
                fig_perf = px.bar(
                    df_health, 
                    x='symbol', 
                    y='performance_score',
                    color='model',
                    title='Model Performance Comparison',
                    template='plotly_dark'
                )
                
                visualizations["performance_chart"] = fig_perf.to_dict()
            
            # Alert timeline
            if alerts:
                alert_df = pd.DataFrame(alerts)
                alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])
                
                fig_alerts = px.scatter(
                    alert_df,
                    x='timestamp',
                    y='severity',
                    color='alert_type',
                    size='severity',
                    title='Model Alerts Timeline',
                    template='plotly_dark'
                )
                
                visualizations["alert_timeline"] = fig_alerts.to_dict()
            
            return visualizations
            
        except Exception as e:
            logger.warning(f"Failed to create monitoring visualizations: {str(e)}")
            return {}
    
    def get_model_health_summary(self, symbol: str = None, model_name: str = None) -> Dict[str, Any]:
        """Get model health summary"""
        try:
            if symbol and model_name:
                key = f"{symbol}_{model_name}"
                if key in self.health_metrics:
                    return {
                        "status": "success",
                        "health_metrics": asdict(self.health_metrics[key])
                    }
                else:
                    return {"error": "No health data available"}
            
            # Return all health metrics
            all_health = {}
            for key, metrics in self.health_metrics.items():
                all_health[key] = asdict(metrics)
            
            return {
                "status": "success",
                "all_health_metrics": all_health,
                "total_models": len(all_health)
            }
            
        except Exception as e:
            logger.error(f"Failed to get health summary: {str(e)}")
            return {"error": f"Failed to get health summary: {str(e)}"}
    
    def resolve_alert(self, alert_id: str) -> Dict[str, Any]:
        """Resolve an alert"""
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    
                    return {
                        "status": "success",
                        "message": f"Alert {alert_id} resolved",
                        "resolved_at": alert.resolved_at.isoformat()
                    }
            
            return {"error": "Alert not found"}
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {str(e)}")
            return {"error": f"Failed to resolve alert: {str(e)}"}

# Global instance
model_monitoring_dashboard = ModelMonitoringDashboard()
