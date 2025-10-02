#!/usr/bin/env python3
"""
Bulk ML Training Script (Corrected & Refactored)

Trains a dedicated set of models for EACH stock individually, which is the
methodologically sound approach for financial time-series prediction.

Key Improvements in this Version:
- Fixed critical state management bug in StockPredictor.
- Centralized all settings into a `Config` class.
- Improved class structure and separation of concerns.
- Enhanced logging, type hinting, and documentation.
- Refactored for better maintainability and readability.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# --- Project Structure Setup ---
# In a real project, this would not be needed if the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import our enhanced models and features
from src.data.dataset_manager import nse_dataset_manager
from src.features.feature_engineering import FeatureEngineer
from src.models.models import StockPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Centralized configuration for the training pipeline."""
    # Data & Features - Now using dynamic feature selection
    TARGET_COLUMN: str = 'close'
    TRAIN_SPLIT_RATIO: float = 0.8
    MIN_DATASET_SIZE: int = 100  # Min records required to train a stock.

    # Model Paths
    MODEL_SAVE_DIR: Path = Path("data/models")

    # LSTM Hyperparameters
    LOOKBACK_DAYS: int = 60
    LSTM_EPOCHS: int = 50
    LSTM_BATCH_SIZE: int = 32
    LSTM_LR: float = 0.001

# =============================================================================
# 2. PYTORCH MODEL & DATASET DEFINITIONS
# (Would live in src/models/ and src/data/)
# =============================================================================

class StockSequenceDataset(Dataset):
    """PyTorch Dataset for creating sequences from stock data."""
    def __init__(self, X: np.ndarray, y: np.ndarray, lookback_days: int):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.lookback_days = lookback_days

    def __len__(self) -> int:
        return len(self.X) - self.lookback_days

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx:idx + self.lookback_days], self.y[idx + self.lookback_days]

class LSTMRegressor(nn.Module):
    """Enhanced LSTM model for stock price prediction with attention mechanism."""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.3):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM for better pattern recognition
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced output layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Self-attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        residual_out = lstm_out + attn_out
        
        # Get the last time step
        last_output = residual_out[:, -1, :]
        
        # Enhanced feedforward layers
        out = self.dropout(self.activation(self.fc1(last_output)))
        out = self.dropout(self.activation(self.fc2(out)))
        out = self.fc3(out)
        
        return out

# =============================================================================
# 3. SINGLE STOCK PREDICTOR
# (The "Worker" class, would live in src/models/predictor.py)
# =============================================================================

class StockModelTrainer:
    """
    Handles training, evaluation, and saving of models for a SINGLE stock.
    This class is instantiated once per stock to ensure no state is carried over.
    """
    def __init__(self, symbol: str, config: Config, device: torch.device):
        self.symbol = symbol
        self.config = config
        self.device = device
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_evaluation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates and returns a dictionary of regression metrics."""
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }

    def _split_and_prepare_data(self, df: pd.DataFrame) -> Tuple:
        """Splits data into train/test sets and extracts features/target."""
        train_size = int(len(df) * self.config.TRAIN_SPLIT_RATIO)
        train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]
        
        # Get actual feature columns from the dataframe (exclude target and metadata columns)
        feature_columns = [col for col in df.columns if col not in ['close', 'date', 'symbol', 'sector']]
        
        X_train = train_df[feature_columns].values
        y_train = train_df[self.config.TARGET_COLUMN].values
        X_test = test_df[feature_columns].values
        y_test = test_df[self.config.TARGET_COLUMN].values
        
        return X_train, y_train, X_test, y_test

    def train_lstm(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trains and evaluates the LSTM model."""
        try:
            X_train, y_train, X_test, y_test = self._split_and_prepare_data(df)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            train_loader = DataLoader(StockSequenceDataset(X_train_scaled, y_train, self.config.LOOKBACK_DAYS), batch_size=self.config.LSTM_BATCH_SIZE, shuffle=True)
            
            model = LSTMRegressor(X_train.shape[1]).to(self.device)
            criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=self.config.LSTM_LR)

            logger.info(f"Starting LSTM training for {self.config.LSTM_EPOCHS} epochs...")
            model.train()
            for epoch in range(self.config.LSTM_EPOCHS):
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(batch_X).squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
            
            logger.info("LSTM training finished. Evaluating on test set...")
            model.eval()
            predictions, actuals = [], []
            
            with torch.no_grad():
                # Create sequences for evaluation
                for i in range(len(X_test_scaled) - self.config.LOOKBACK_DAYS):
                    # Create sequence for prediction
                    sequence = X_test_scaled[i:i + self.config.LOOKBACK_DAYS].reshape(1, self.config.LOOKBACK_DAYS, -1)
                    sequence_tensor = torch.FloatTensor(sequence).to(self.device)
                    
                    pred = model(sequence_tensor).squeeze().cpu().numpy()
                    predictions.append(pred)
                    actuals.append(y_test[i + self.config.LOOKBACK_DAYS])
                
                preds = np.array(predictions)
                acts = np.array(actuals)

            self.models['lstm'], self.scalers['lstm'] = model, scaler
            return {"status": "success", **self._get_evaluation_metrics(acts, preds)}

        except Exception as e:
            logger.error(f"LSTM training failed for {self.symbol}: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def _train_tree_model(self, df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Generic trainer for tree-based models (RandomForest, XGBoost)."""
        try:
            X_train, y_train, X_test, y_test = self._split_and_prepare_data(df)

            if model_name == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_name == 'xgboost':
                model = xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method='gpu_hist' if self.device.type == 'cuda' else 'hist')
            else:
                raise ValueError(f"Unknown tree model: {model_name}")

            logger.info(f"Fitting {model_name} model...")
            model.fit(X_train, y_train)
            
            logger.info("Evaluating on test set...")
            y_pred = model.predict(X_test)
            
            self.models[model_name] = model
            return {"status": "success", **self._get_evaluation_metrics(y_test, y_pred)}

        except Exception as e:
            logger.error(f"{model_name} training failed for {self.symbol}: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def train_random_forest(self, df: pd.DataFrame) -> Dict[str, Any]:
        return self._train_tree_model(df, 'random_forest')

    def train_xgboost(self, df: pd.DataFrame) -> Dict[str, Any]:
        return self._train_tree_model(df, 'xgboost')
    
    def save_models(self):
        """Saves all trained models and scalers for the stock."""
        logger.info(f"Saving models for {self.symbol}...")
        for name, model in self.models.items():
            path = self.config.MODEL_SAVE_DIR / f"{self.symbol.upper()}_{name}.pkl"
            joblib.dump(model, path)
        for name, scaler in self.scalers.items():
            path = self.config.MODEL_SAVE_DIR / f"{self.symbol.upper()}_{name}_scaler.pkl"
            joblib.dump(scaler, path)
        logger.info(f"Successfully saved all artifacts for {self.symbol}.")


# =============================================================================
# 4. BULK TRAINING ORCHESTRATOR
# (The "Orchestrator" class, would live in src/training/pipeline.py)
# =============================================================================

class BulkModelTrainer:
    """Orchestrates training models for a list of stocks."""
    def __init__(self, nse_manager, feature_engineer, config: Config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Bulk trainer initialized on device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU Detected: {torch.cuda.get_device_name(0)}")

        self.nse_manager = nse_manager
        self.feature_engineer = feature_engineer
        self.config = config

    def train_models_for_stocks(self, stocks: List[str] = None, models: List[str] = None, years: List[str] = None) -> Dict[str, Any]:
        """Trains a dedicated set of models for each stock provided."""
        stocks = stocks or self.nse_manager.get_available_stocks()
        models = models or ['lstm', 'random_forest', 'xgboost']
        years = years or ['2022', '2023', '2024']

        logger.info(f"Starting bulk training for {len(stocks)} stocks using models: {models}.")
        
        overall_results = {
            "start_time": datetime.now().isoformat(),
            "training_years": years,
            "config": self.config.__dict__,
            "results_by_stock": {}
        }

        for i, stock_symbol in enumerate(stocks, 1):
            logger.info(f"\n{'='*70}\nProcessing Stock {i}/{len(stocks)}: {stock_symbol.upper()}\n{'='*70}")
            
            try:
                stock_data = self.nse_manager.load_multi_year_data(stock_symbol, years)
                if stock_data.empty or len(stock_data) < self.config.MIN_DATASET_SIZE:
                    logger.warning(f"Skipping {stock_symbol} due to insufficient data ({len(stock_data)} records).")
                    overall_results["results_by_stock"][stock_symbol] = {"status": "skipped", "reason": "Insufficient data"}
                    continue
                
                logger.info(f"Loaded {len(stock_data)} records for {stock_symbol}. Engineering features...")
                processed_data = self.feature_engineer.prepare_features(stock_data)
                logger.info(f"Feature engineering complete. Final dataset size: {len(processed_data)}.")

                # Use our enhanced StockPredictor instead of the old StockModelTrainer
                predictor = StockPredictor()
                
                stock_results = {}
                for model_name in models:
                    logger.info(f"--- Training {model_name.upper()} for {stock_symbol} ---")
                    
                    try:
                        if model_name == 'lstm':
                            result = predictor.train_lstm(processed_data)
                        elif model_name == 'random_forest':
                            result = predictor.train_random_forest(processed_data)
                        elif model_name == 'xgboost':
                            result = predictor.train_xgboost(processed_data)
                        elif model_name == 'arima':
                            result = predictor.train_arima(processed_data)
                        elif model_name == 'prophet':
                            result = predictor.train_prophet(processed_data)
                        else:
                            logger.warning(f"Model type '{model_name}' is not supported.")
                            continue
                        
                        if 'error' in result:
                            logger.error(f"Failed to train {model_name} for {stock_symbol}: {result['error']}")
                            stock_results[model_name] = {"status": "failed", "error": result['error']}
                        else:
                            r2_score = result.get('r2', 0)
                            logger.info(f"Successfully trained {model_name} for {stock_symbol}. R2 Score: {r2_score:.4f}")
                            stock_results[model_name] = {"status": "success", "r2": r2_score}
                    
                    except Exception as e:
                        logger.error(f"Error training {model_name} for {stock_symbol}: {e}")
                        stock_results[model_name] = {"status": "failed", "error": str(e)}

                predictor.save_models(stock_symbol)
                overall_results["results_by_stock"][stock_symbol] = stock_results

            except Exception as e:
                logger.error(f"A critical error occurred while processing {stock_symbol}: {e}", exc_info=True)
                overall_results["results_by_stock"][stock_symbol] = {"status": "critical_failure", "error": str(e)}

        overall_results["end_time"] = datetime.now().isoformat()
        logger.info(f"\n{'='*70}\nBULK TRAINING COMPLETE.\n{'='*70}")
        return overall_results

# =============================================================================
# 5. MAIN EXECUTION SCRIPT
# (Would live in scripts/train.py)
# =============================================================================

# Mock classes removed - now using real dataset and feature engineering

def run_training_pipeline(args=None):
    """Main function to configure and run the training pipeline."""
    print("\n" + "="*60)
    print("      Corrected ML Model Training Pipeline")
    print("="*60)
    print("APPROACH: Train a dedicated model for EACH stock.")
    print("This is the methodologically sound approach.")
    print("="*60 + "\n")
    
    # Use real dataset manager and feature engineer
    from src.data.dataset_manager import nse_dataset_manager
    from src.features.feature_engineering import FeatureEngineer
    
    real_nse_manager = nse_dataset_manager
    real_feature_engineer = FeatureEngineer()
    
    # Initialize the configuration and the main trainer
    config = Config()
    pipeline = BulkModelTrainer(
        nse_manager=real_nse_manager,
        feature_engineer=real_feature_engineer,
        config=config
    )

    # Define the scope of the training run - use ALL available stocks
    stocks_to_train = None  # None means all available stocks
    models_to_train = ['lstm', 'random_forest', 'xgboost']

    results = pipeline.train_models_for_stocks(
        stocks=stocks_to_train,
        models=models_to_train
    )

    # Save results to a JSON file
    results_path = 'bulk_training_results_refactored.json'
    try:
        # Convert Path object in config to string for JSON serialization
        results['config']['MODEL_SAVE_DIR'] = str(results['config']['MODEL_SAVE_DIR'])
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"\nTraining finished. Full results saved to '{results_path}'")
    except Exception as e:
        print(f"\nCould not save results to file: {e}")

if __name__ == "__main__":
    run_training_pipeline()