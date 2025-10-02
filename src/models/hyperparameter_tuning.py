"""
Hyperparameter Tuning Module
Implements automated hyperparameter optimization using Optuna
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime

from src.models.models import StockPredictor, LSTM
from config.settings import settings

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization for ML models
    """
    
    def __init__(self):
        self.study_storage = "data/models/hyperparameter_studies"
        os.makedirs(self.study_storage, exist_ok=True)
        self.studies = {}
        
    def optimize_lstm(self, df: pd.DataFrame, symbol: str, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize LSTM hyperparameters using Optuna
        
        Args:
            df: Training DataFrame
            symbol: Stock symbol
            n_trials: Number of optimization trials
        
        Returns:
            Optimization results
        """
        try:
            logger.info(f"Starting LSTM hyperparameter optimization for {symbol}")
            
            # Prepare data
            X, y = self._prepare_lstm_data(df)
            if len(X) == 0:
                return {"error": "No valid data for LSTM optimization"}
            
            # Create study
            study_name = f"lstm_{symbol}_{datetime.now().strftime('%Y%m%d')}"
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=f"sqlite:///{self.study_storage}/{study_name}.db",
                load_if_exists=True
            )
            
            # Define objective function
            def objective(trial):
                # Suggest hyperparameters
                hidden_size = trial.suggest_int('hidden_size', 32, 256)
                num_layers = trial.suggest_int('num_layers', 1, 4)
                dropout = trial.suggest_float('dropout', 0.1, 0.5)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
                epochs = trial.suggest_int('epochs', 50, 200)
                
                # Train and evaluate model
                score = self._evaluate_lstm_config(
                    X, y, hidden_size, num_layers, dropout, 
                    learning_rate, batch_size, epochs
                )
                
                return score
            
            # Optimize
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value
            
            # Save study
            self.studies[f"lstm_{symbol}"] = study
            
            logger.info(f"LSTM optimization completed for {symbol}. Best score: {best_score:.4f}")
            
            return {
                "status": "success",
                "symbol": symbol,
                "model": "lstm",
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": n_trials,
                "study_name": study_name
            }
            
        except Exception as e:
            logger.error(f"LSTM hyperparameter optimization failed: {str(e)}")
            return {"error": str(e)}
    
    def optimize_random_forest(self, df: pd.DataFrame, symbol: str, n_trials: int = 30) -> Dict[str, Any]:
        """
        Optimize Random Forest hyperparameters
        
        Args:
            df: Training DataFrame
            symbol: Stock symbol
            n_trials: Number of optimization trials
        
        Returns:
            Optimization results
        """
        try:
            logger.info(f"Starting Random Forest hyperparameter optimization for {symbol}")
            
            # Prepare data
            X, y = self._prepare_sklearn_data(df)
            if len(X) == 0:
                return {"error": "No valid data for Random Forest optimization"}
            
            # Create study
            study_name = f"rf_{symbol}_{datetime.now().strftime('%Y%m%d')}"
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=f"sqlite:///{self.study_storage}/{study_name}.db",
                load_if_exists=True
            )
            
            # Define objective function
            def objective(trial):
                # Suggest hyperparameters
                n_estimators = trial.suggest_int('n_estimators', 50, 500)
                max_depth = trial.suggest_int('max_depth', 5, 30)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                
                # Train and evaluate model
                score = self._evaluate_rf_config(
                    X, y, n_estimators, max_depth, min_samples_split, 
                    min_samples_leaf, max_features
                )
                
                return score
            
            # Optimize
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value
            
            # Save study
            self.studies[f"rf_{symbol}"] = study
            
            logger.info(f"Random Forest optimization completed for {symbol}. Best score: {best_score:.4f}")
            
            return {
                "status": "success",
                "symbol": symbol,
                "model": "random_forest",
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": n_trials,
                "study_name": study_name
            }
            
        except Exception as e:
            logger.error(f"Random Forest hyperparameter optimization failed: {str(e)}")
            return {"error": str(e)}
    
    def optimize_xgboost(self, df: pd.DataFrame, symbol: str, n_trials: int = 30) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters
        
        Args:
            df: Training DataFrame
            symbol: Stock symbol
            n_trials: Number of optimization trials
        
        Returns:
            Optimization results
        """
        try:
            logger.info(f"Starting XGBoost hyperparameter optimization for {symbol}")
            
            # Prepare data
            X, y = self._prepare_sklearn_data(df)
            if len(X) == 0:
                return {"error": "No valid data for XGBoost optimization"}
            
            # Create study
            study_name = f"xgb_{symbol}_{datetime.now().strftime('%Y%m%d')}"
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=f"sqlite:///{self.study_storage}/{study_name}.db",
                load_if_exists=True
            )
            
            # Define objective function
            def objective(trial):
                # Suggest hyperparameters
                n_estimators = trial.suggest_int('n_estimators', 50, 500)
                max_depth = trial.suggest_int('max_depth', 3, 10)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                subsample = trial.suggest_float('subsample', 0.6, 1.0)
                colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
                reg_alpha = trial.suggest_float('reg_alpha', 0, 10)
                reg_lambda = trial.suggest_float('reg_lambda', 0, 10)
                
                # Train and evaluate model
                score = self._evaluate_xgb_config(
                    X, y, n_estimators, max_depth, learning_rate, 
                    subsample, colsample_bytree, reg_alpha, reg_lambda
                )
                
                return score
            
            # Optimize
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value
            
            # Save study
            self.studies[f"xgb_{symbol}"] = study
            
            logger.info(f"XGBoost optimization completed for {symbol}. Best score: {best_score:.4f}")
            
            return {
                "status": "success",
                "symbol": symbol,
                "model": "xgboost",
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": n_trials,
                "study_name": study_name
            }
            
        except Exception as e:
            logger.error(f"XGBoost hyperparameter optimization failed: {str(e)}")
            return {"error": str(e)}
    
    def optimize_all_models(self, df: pd.DataFrame, symbol: str, n_trials_per_model: int = 30) -> Dict[str, Any]:
        """
        Optimize hyperparameters for all models
        
        Args:
            df: Training DataFrame
            symbol: Stock symbol
            n_trials_per_model: Number of trials per model
        
        Returns:
            Combined optimization results
        """
        logger.info(f"Starting comprehensive hyperparameter optimization for {symbol}")
        
        results = {}
        
        # Optimize each model
        models_to_optimize = [
            ('lstm', self.optimize_lstm, n_trials_per_model * 2),  # More trials for LSTM
            ('random_forest', self.optimize_random_forest, n_trials_per_model),
            ('xgboost', self.optimize_xgboost, n_trials_per_model)
        ]
        
        for model_name, optimize_func, n_trials in models_to_optimize:
            try:
                result = optimize_func(df, symbol, n_trials)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Failed to optimize {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}
        
        # Find best overall model
        best_model = None
        best_score = -np.inf
        
        for model_name, result in results.items():
            if result.get("status") == "success" and result.get("best_score", -np.inf) > best_score:
                best_score = result["best_score"]
                best_model = model_name
        
        return {
            "status": "success",
            "symbol": symbol,
            "results": results,
            "best_model": best_model,
            "best_score": best_score,
            "optimization_date": datetime.now().isoformat()
        }
    
    def _prepare_lstm_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM optimization"""
        try:
            from src.features.feature_engineering import FeatureEngineer
            feature_engineer = FeatureEngineer()
            
            # Prepare features
            processed_df = feature_engineer.prepare_features(df)
            
            # Get feature columns
            feature_columns = [col for col in settings.FEATURE_COLUMNS if col in processed_df.columns]
            X = processed_df[feature_columns].fillna(0).values
            y = processed_df[settings.TARGET_COLUMN].values
            
            # Remove NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to prepare LSTM data: {str(e)}")
            return np.array([]), np.array([])
    
    def _prepare_sklearn_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for sklearn models optimization"""
        try:
            from src.features.feature_engineering import FeatureEngineer
            feature_engineer = FeatureEngineer()
            
            # Prepare features
            processed_df = feature_engineer.prepare_features(df)
            
            # Get feature columns
            feature_columns = [col for col in settings.FEATURE_COLUMNS if col in processed_df.columns]
            X = processed_df[feature_columns].fillna(0)
            y = processed_df[settings.TARGET_COLUMN]
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], 0)
            
            return X.values, y.values
            
        except Exception as e:
            logger.error(f"Failed to prepare sklearn data: {str(e)}")
            return np.array([]), np.array([])
    
    def _evaluate_lstm_config(self, X: np.ndarray, y: np.ndarray, hidden_size: int, 
                            num_layers: int, dropout: float, learning_rate: float, 
                            batch_size: int, epochs: int) -> float:
        """Evaluate LSTM configuration using cross-validation"""
        try:
            from sklearn.preprocessing import StandardScaler
            from torch.utils.data import DataLoader
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create datasets
                from src.models.models import StockDataset
                train_dataset = StockDataset(X_train, y_train, settings.LOOKBACK_DAYS)
                val_dataset = StockDataset(X_val, y_val, settings.LOOKBACK_DAYS)
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                # Initialize model
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = LSTM(input_size=X.shape[1], hidden_size=hidden_size, 
                           num_layers=num_layers, dropout=dropout).to(device)
                
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                
                # Training loop
                model.train()
                for epoch in range(min(epochs, 50)):  # Limit epochs for optimization
                    for batch_X, batch_y in train_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        loss.backward()
                        optimizer.step()
                
                # Evaluation
                model.eval()
                val_predictions = []
                val_targets = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(device)
                        outputs = model(batch_X)
                        val_predictions.extend(outputs.cpu().numpy())
                        val_targets.extend(batch_y.numpy())
                
                # Calculate R² score
                val_predictions = np.array(val_predictions).flatten()
                val_targets = np.array(val_targets)
                
                if len(val_predictions) > 0 and len(val_targets) > 0:
                    r2 = r2_score(val_targets, val_predictions)
                    scores.append(r2)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.warning(f"LSTM evaluation failed: {str(e)}")
            return 0.0
    
    def _evaluate_rf_config(self, X: np.ndarray, y: np.ndarray, n_estimators: int, 
                          max_depth: int, min_samples_split: int, min_samples_leaf: int, 
                          max_features: str) -> float:
        """Evaluate Random Forest configuration using cross-validation"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_val)
                r2 = r2_score(y_val, y_pred)
                scores.append(r2)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.warning(f"Random Forest evaluation failed: {str(e)}")
            return 0.0
    
    def _evaluate_xgb_config(self, X: np.ndarray, y: np.ndarray, n_estimators: int, 
                           max_depth: int, learning_rate: float, subsample: float, 
                           colsample_bytree: float, reg_alpha: float, reg_lambda: float) -> float:
        """Evaluate XGBoost configuration using cross-validation"""
        try:
            import xgboost as xgb
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_val)
                r2 = r2_score(y_val, y_pred)
                scores.append(r2)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.warning(f"XGBoost evaluation failed: {str(e)}")
            return 0.0
    
    def get_optimization_history(self, symbol: str, model_name: str) -> Dict[str, Any]:
        """Get optimization history for a model"""
        try:
            study_key = f"{model_name}_{symbol}"
            if study_key in self.studies:
                study = self.studies[study_key]
                
                # Get trial history
                trials = study.trials
                history = []
                
                for trial in trials:
                    if trial.state == optuna.trial.TrialState.COMPLETE:
                        history.append({
                            'trial_number': trial.number,
                            'value': trial.value,
                            'params': trial.params,
                            'datetime': trial.datetime_start.isoformat() if trial.datetime_start else None
                        })
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "model": model_name,
                    "best_value": study.best_value,
                    "best_params": study.best_params,
                    "n_trials": len(trials),
                    "history": history
                }
            else:
                return {"error": f"No optimization history found for {model_name}_{symbol}"}
                
        except Exception as e:
            logger.error(f"Failed to get optimization history: {str(e)}")
            return {"error": str(e)}

# Global optimizer instance
hyperparameter_optimizer = HyperparameterOptimizer()
