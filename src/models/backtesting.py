"""
Backtesting Framework Module
Implements comprehensive backtesting for stock prediction models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.models import StockPredictor
from src.data.dataset_manager import nse_dataset_manager
from src.features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class BacktestingFramework:
    """
    Comprehensive backtesting framework for stock prediction models
    """
    
    def __init__(self):
        self.stock_predictor = StockPredictor()
        self.feature_engineer = FeatureEngineer()
        self.results_storage = "data/backtesting_results"
        import os
        os.makedirs(self.results_storage, exist_ok=True)
    
    def run_backtest(self, symbol: str, model_name: str = 'lstm', 
                    start_date: str = '2023-01-01', end_date: str = '2024-12-31',
                    train_window: int = 252, test_window: int = 30,
                    retrain_frequency: int = 30) -> Dict[str, Any]:
        """
        Run comprehensive backtest for a model
        
        Args:
            symbol: Stock symbol
            model_name: Model to backtest
            start_date: Start date for backtesting
            end_date: End date for backtesting
            train_window: Training window size in days
            test_window: Testing window size in days
            retrain_frequency: How often to retrain the model (days)
        
        Returns:
            Backtesting results
        """
        try:
            logger.info(f"Starting backtest for {symbol} using {model_name}")
            
            # Load data
            data = nse_dataset_manager.load_multi_year_data(symbol, ['2022', '2023', '2024'])
            if data.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Filter data by date range
            data['date'] = pd.to_datetime(data['date'])
            data = data[(data['date'] >= start_date) & (data['date'] <= end_date)].copy()
            
            if len(data) < train_window + test_window:
                return {"error": "Insufficient data for backtesting"}
            
            # Prepare features
            processed_data = self.feature_engineer.prepare_features(data)
            
            # Run rolling window backtest
            backtest_results = self._rolling_window_backtest(
                processed_data, symbol, model_name, train_window, test_window, retrain_frequency
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(backtest_results)
            
            # Generate trading signals and analyze
            trading_analysis = self._analyze_trading_performance(backtest_results)
            
            # Save results
            results = {
                "status": "success",
                "symbol": symbol,
                "model": model_name,
                "backtest_period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_days": len(data)
                },
                "parameters": {
                    "train_window": train_window,
                    "test_window": test_window,
                    "retrain_frequency": retrain_frequency
                },
                "backtest_results": backtest_results,
                "performance_metrics": performance_metrics,
                "trading_analysis": trading_analysis,
                "backtest_date": datetime.now().isoformat()
            }
            
            # Save to file
            self._save_backtest_results(symbol, model_name, results)
            
            logger.info(f"Backtest completed for {symbol}. Total predictions: {len(backtest_results)}")
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    def _rolling_window_backtest(self, data: pd.DataFrame, symbol: str, model_name: str,
                               train_window: int, test_window: int, retrain_frequency: int) -> List[Dict[str, Any]]:
        """Run rolling window backtest"""
        results = []
        
        # Sort data by date
        data = data.sort_values('date').reset_index(drop=True)
        
        # Start from train_window
        start_idx = train_window
        end_idx = len(data) - test_window
        
        for i in range(start_idx, end_idx, retrain_frequency):
            try:
                # Define training and testing periods
                train_start = max(0, i - train_window)
                train_end = i
                test_start = i
                test_end = min(len(data), i + test_window)
                
                # Get training data
                train_data = data.iloc[train_start:train_end].copy()
                test_data = data.iloc[test_start:test_end].copy()
                
                if len(train_data) < train_window * 0.8 or len(test_data) < test_window * 0.8:
                    continue
                
                # Train model
                training_result = self._train_model_for_backtest(train_data, symbol, model_name)
                if training_result.get("status") != "success":
                    continue
                
                # Make predictions
                predictions = self._make_backtest_predictions(test_data, symbol, model_name)
                
                # Calculate metrics for this window
                window_metrics = self._calculate_window_metrics(test_data, predictions)
                
                # Store results
                window_result = {
                    "window_id": len(results),
                    "train_period": {
                        "start": train_data['date'].min().isoformat(),
                        "end": train_data['date'].max().isoformat(),
                        "days": len(train_data)
                    },
                    "test_period": {
                        "start": test_data['date'].min().isoformat(),
                        "end": test_data['date'].max().isoformat(),
                        "days": len(test_data)
                    },
                    "predictions": predictions,
                    "metrics": window_metrics,
                    "training_result": training_result
                }
                
                results.append(window_result)
                
            except Exception as e:
                logger.warning(f"Window {i} failed: {str(e)}")
                continue
        
        return results
    
    def _train_model_for_backtest(self, train_data: pd.DataFrame, symbol: str, model_name: str) -> Dict[str, Any]:
        """Train model for backtesting"""
        try:
            if model_name == 'lstm':
                result = self.stock_predictor.train_lstm(train_data)
            elif model_name == 'random_forest':
                result = self.stock_predictor.train_random_forest(train_data)
            elif model_name == 'xgboost':
                result = self.stock_predictor.train_xgboost(train_data)
            elif model_name == 'arima':
                result = self.stock_predictor.train_arima(train_data)
            elif model_name == 'prophet':
                result = self.stock_predictor.train_prophet(train_data)
            else:
                return {"error": f"Unknown model: {model_name}"}
            
            return result
            
        except Exception as e:
            logger.warning(f"Model training failed: {str(e)}")
            return {"error": str(e)}
    
    def _make_backtest_predictions(self, test_data: pd.DataFrame, symbol: str, model_name: str) -> List[Dict[str, Any]]:
        """Make predictions for backtesting"""
        try:
            predictions = []
            
            for idx, row in test_data.iterrows():
                try:
                    # Get prediction for this point
                    pred_data = test_data.iloc[:idx+1]  # Use data up to current point
                    
                    if len(pred_data) < 10:  # Need minimum data
                        continue
                    
                    pred = self.stock_predictor.predict(pred_data, model_name)
                    
                    if len(pred) > 0:
                        predictions.append({
                            "date": row['date'].isoformat(),
                            "actual_price": float(row['close']),
                            "predicted_price": float(pred[-1]),
                            "prediction_error": float(pred[-1] - row['close']),
                            "relative_error": float((pred[-1] - row['close']) / row['close']) if row['close'] > 0 else 0
                        })
                
                except Exception as e:
                    logger.warning(f"Prediction failed for {row['date']}: {str(e)}")
                    continue
            
            return predictions
            
        except Exception as e:
            logger.warning(f"Backtest prediction failed: {str(e)}")
            return []
    
    def _calculate_window_metrics(self, test_data: pd.DataFrame, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for a backtest window"""
        try:
            if not predictions:
                return {"error": "No predictions available"}
            
            # Extract actual and predicted values
            actual_prices = [p['actual_price'] for p in predictions]
            predicted_prices = [p['predicted_price'] for p in predictions]
            
            # Calculate metrics
            mse = mean_squared_error(actual_prices, predicted_prices)
            mae = mean_absolute_error(actual_prices, predicted_prices)
            r2 = r2_score(actual_prices, predicted_prices)
            
            # Calculate directional accuracy
            actual_direction = [1 if actual_prices[i] > actual_prices[i-1] else 0 for i in range(1, len(actual_prices))]
            predicted_direction = [1 if predicted_prices[i] > predicted_prices[i-1] else 0 for i in range(1, len(predicted_prices))]
            
            directional_accuracy = sum(a == p for a, p in zip(actual_direction, predicted_direction)) / len(actual_direction) if actual_direction else 0
            
            # Calculate Sharpe ratio for predictions
            returns = [(predicted_prices[i] - predicted_prices[i-1]) / predicted_prices[i-1] for i in range(1, len(predicted_prices))]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            return {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2),
                "directional_accuracy": float(directional_accuracy),
                "sharpe_ratio": float(sharpe_ratio),
                "num_predictions": len(predictions),
                "avg_error": float(np.mean([abs(p['prediction_error']) for p in predictions])),
                "max_error": float(np.max([abs(p['prediction_error']) for p in predictions]))
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate window metrics: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_performance_metrics(self, backtest_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall performance metrics"""
        try:
            if not backtest_results:
                return {"error": "No backtest results available"}
            
            # Aggregate metrics across all windows
            all_metrics = []
            all_predictions = []
            
            for result in backtest_results:
                if "metrics" in result and "error" not in result["metrics"]:
                    all_metrics.append(result["metrics"])
                
                if "predictions" in result:
                    all_predictions.extend(result["predictions"])
            
            if not all_metrics:
                return {"error": "No valid metrics available"}
            
            # Calculate aggregate metrics
            aggregate_metrics = {
                "total_windows": len(backtest_results),
                "total_predictions": len(all_predictions),
                "avg_r2": float(np.mean([m.get("r2", 0) for m in all_metrics])),
                "avg_mse": float(np.mean([m.get("mse", 0) for m in all_metrics])),
                "avg_mae": float(np.mean([m.get("mae", 0) for m in all_metrics])),
                "avg_directional_accuracy": float(np.mean([m.get("directional_accuracy", 0) for m in all_metrics])),
                "avg_sharpe_ratio": float(np.mean([m.get("sharpe_ratio", 0) for m in all_metrics])),
                "consistency_score": float(1.0 - np.std([m.get("r2", 0) for m in all_metrics])),
                "best_window_r2": float(np.max([m.get("r2", 0) for m in all_metrics])),
                "worst_window_r2": float(np.min([m.get("r2", 0) for m in all_metrics]))
            }
            
            # Calculate overall prediction accuracy
            if all_predictions:
                overall_errors = [abs(p['prediction_error']) for p in all_predictions]
                aggregate_metrics.update({
                    "overall_avg_error": float(np.mean(overall_errors)),
                    "overall_max_error": float(np.max(overall_errors)),
                    "overall_error_std": float(np.std(overall_errors))
                })
            
            return aggregate_metrics
            
        except Exception as e:
            logger.warning(f"Failed to calculate performance metrics: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_trading_performance(self, backtest_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trading performance based on predictions"""
        try:
            if not backtest_results:
                return {"error": "No backtest results available"}
            
            # Collect all predictions
            all_predictions = []
            for result in backtest_results:
                if "predictions" in result:
                    all_predictions.extend(result["predictions"])
            
            if not all_predictions:
                return {"error": "No predictions available"}
            
            # Sort by date
            all_predictions.sort(key=lambda x: x['date'])
            
            # Simulate trading strategy
            trading_results = self._simulate_trading_strategy(all_predictions)
            
            return trading_results
            
        except Exception as e:
            logger.warning(f"Failed to analyze trading performance: {str(e)}")
            return {"error": str(e)}
    
    def _simulate_trading_strategy(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate trading strategy based on predictions"""
        try:
            if len(predictions) < 2:
                return {"error": "Insufficient predictions for trading simulation"}
            
            # Simple trading strategy: buy if prediction > actual, sell otherwise
            initial_capital = 10000  # Starting with $10,000
            capital = initial_capital
            position = 0  # Number of shares
            trades = []
            
            for i in range(1, len(predictions)):
                current_pred = predictions[i]
                previous_pred = predictions[i-1]
                
                # Calculate price change prediction
                pred_change = current_pred['predicted_price'] - previous_pred['predicted_price']
                actual_change = current_pred['actual_price'] - previous_pred['actual_price']
                
                # Trading signal
                if pred_change > 0 and position == 0:  # Buy signal
                    shares_to_buy = capital // current_pred['actual_price']
                    if shares_to_buy > 0:
                        position = shares_to_buy
                        capital -= shares_to_buy * current_pred['actual_price']
                        trades.append({
                            "date": current_pred['date'],
                            "action": "BUY",
                            "price": current_pred['actual_price'],
                            "shares": shares_to_buy,
                            "capital": capital
                        })
                
                elif pred_change < 0 and position > 0:  # Sell signal
                    capital += position * current_pred['actual_price']
                    trades.append({
                        "date": current_pred['date'],
                        "action": "SELL",
                        "price": current_pred['actual_price'],
                        "shares": position,
                        "capital": capital
                    })
                    position = 0
            
            # Final portfolio value
            final_value = capital + (position * predictions[-1]['actual_price']) if position > 0 else capital
            total_return = (final_value - initial_capital) / initial_capital
            
            # Calculate trading metrics
            trading_metrics = {
                "initial_capital": initial_capital,
                "final_value": final_value,
                "total_return": total_return,
                "total_trades": len(trades),
                "buy_trades": len([t for t in trades if t['action'] == 'BUY']),
                "sell_trades": len([t for t in trades if t['action'] == 'SELL']),
                "trades": trades
            }
            
            return trading_metrics
            
        except Exception as e:
            logger.warning(f"Trading simulation failed: {str(e)}")
            return {"error": str(e)}
    
    def _save_backtest_results(self, symbol: str, model_name: str, results: Dict[str, Any]):
        """Save backtest results to file"""
        try:
            import json
            
            filename = f"{self.results_storage}/backtest_{symbol}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save backtest results: {str(e)}")
    
    def compare_models_backtest(self, symbol: str, models: List[str] = None,
                              start_date: str = '2023-01-01', end_date: str = '2024-12-31') -> Dict[str, Any]:
        """
        Compare multiple models using backtesting
        
        Args:
            symbol: Stock symbol
            models: List of models to compare
            start_date: Start date for backtesting
            end_date: End date for backtesting
        
        Returns:
            Model comparison results
        """
        try:
            if models is None:
                models = ['lstm', 'random_forest', 'xgboost', 'arima', 'prophet']
            
            logger.info(f"Comparing models for {symbol}: {models}")
            
            comparison_results = {}
            
            for model_name in models:
                try:
                    result = self.run_backtest(symbol, model_name, start_date, end_date)
                    comparison_results[model_name] = result
                except Exception as e:
                    logger.warning(f"Backtest failed for {model_name}: {str(e)}")
                    comparison_results[model_name] = {"error": str(e)}
            
            # Find best performing model
            best_model = None
            best_r2 = -np.inf
            
            for model_name, result in comparison_results.items():
                if result.get("status") == "success":
                    r2 = result.get("performance_metrics", {}).get("avg_r2", -np.inf)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model_name
            
            return {
                "status": "success",
                "symbol": symbol,
                "models_compared": models,
                "comparison_results": comparison_results,
                "best_model": best_model,
                "best_r2": best_r2,
                "comparison_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}")
            return {"error": str(e)}

# Global backtesting instance
backtesting_framework = BacktestingFramework()
