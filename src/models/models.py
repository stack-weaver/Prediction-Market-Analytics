"""
Machine Learning Models for Stock Price Prediction (Refactored)
Improved version with corrected methodology and robust evaluation.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import joblib
import logging
from .transformer_model import TransformerPredictor
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (replaced config file for self-contained script) ---
LOOKBACK_DAYS = 60
EPOCHS = 50


class StockDataset(Dataset):
    """PyTorch Dataset for creating sequences from stock data."""
    def __init__(self, X: np.ndarray, y: np.ndarray, lookback_days: int):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.lookback_days = lookback_days

    def __len__(self) -> int:
        return len(self.X) - self.lookback_days

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.X[idx:idx + self.lookback_days],
            self.y[idx + self.lookback_days]
        )


class LSTM(nn.Module):
    """Enhanced LSTM model for stock price prediction with attention mechanism."""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.3):
        super(LSTM, self).__init__()
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


class StockPredictor:
    """
    Main class for stock price prediction using multiple ML models.
    This version includes proper train-test splitting and evaluation.
    """
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs('data/models', exist_ok=True)
        logger.info(f"StockPredictor initialized on device: {self.device}")

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get available feature columns dynamically"""
        # Base columns that should always be present
        base_columns = ['open', 'high', 'low', 'volume']
        
        # Technical indicators
        technical_indicators = [
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'Upper_Bollinger', 'Middle_Bollinger', 'Lower_Bollinger',
            'SMA_5', 'EMA_5', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50',
            'OBV', 'Williams_R', 'CCI', 'ATR', 'MFI', 'Stoch_K', 'Stoch_D', 'ROC', 'UO'
        ]
        
        # Lag features
        lag_features = [col for col in df.columns if 'lag_' in col]
        
        # Combine all available features
        available_features = []
        for col in base_columns + technical_indicators + lag_features:
            if col in df.columns and col != 'date' and col != 'close':
                available_features.append(col)
        
        logger.info(f"Using {len(available_features)} features: {available_features[:10]}...")
        return available_features

    def train_models(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all available models"""
        logger.info(f"Training all models for {symbol}")
        results = {}
        
        # Train each model
        try:
            results['lstm'] = self.train_lstm(df)
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            results['lstm'] = {"error": str(e)}
        
        try:
            results['random_forest'] = self.train_random_forest(df)
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            results['random_forest'] = {"error": str(e)}
        
        try:
            results['xgboost'] = self.train_xgboost(df)
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            results['xgboost'] = {"error": str(e)}
        
        try:
            results['arima'] = self.train_arima(df)
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            results['arima'] = {"error": str(e)}
        
        try:
            results['prophet'] = self.train_prophet(df)
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            results['prophet'] = {"error": str(e)}
        
        try:
            results['transformer'] = self.train_transformer(df)
        except Exception as e:
            logger.error(f"Transformer training failed: {e}")
            results['transformer'] = {"error": str(e)}
        
        logger.info(f"Model training completed for {symbol}")
        return results

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Performs a chronological 80/20 split on the dataframe."""
        if len(df) < 10:
            raise ValueError("Dataframe too small to be split.")
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        return train_df, test_df

    def _prepare_sequential_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares data for LSTM by creating sequences."""
        feature_columns = self._get_feature_columns(df)
        X = df[feature_columns].values.astype(np.float64)
        y = df['close'].values.astype(np.float64)
        return X, y

    def _get_evaluation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates and returns a dictionary of regression metrics."""
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }

    def train_lstm(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Trains and evaluates the LSTM model using a chronological train-test split.
        The scaler is fit only on the training data to prevent data leakage.
        """
        try:
            train_df, test_df = self._split_data(df)
            X_train, y_train = self._prepare_sequential_data(train_df)
            X_test, y_test = self._prepare_sequential_data(test_df)

            if len(X_train) < LOOKBACK_DAYS or len(X_test) < LOOKBACK_DAYS:
                return {"error": f"Insufficient data after split for lookback of {LOOKBACK_DAYS}"}

            # Fit scaler ONLY on training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Transform test data with the SAME scaler
            X_test_scaled = scaler.transform(X_test)

            train_dataset = StockDataset(X_train_scaled, y_train, LOOKBACK_DAYS)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            test_dataset = StockDataset(X_test_scaled, y_test, LOOKBACK_DAYS)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            model = LSTM(input_size=X_train.shape[1]).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # --- Training Loop ---
            model.train()
            for epoch in range(EPOCHS):
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                if (epoch + 1) % 10 == 0:
                    logger.info(f"LSTM Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

            # --- Evaluation Loop ---
            model.eval()
            predictions, actuals = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    predictions.extend(outputs.squeeze().cpu().numpy())
                    actuals.extend(batch_y.cpu().numpy())
            
            self.models['lstm'] = model
            self.scalers['lstm'] = scaler
            
            logger.info("LSTM model trained and evaluated successfully.")
            return self._get_evaluation_metrics(np.array(actuals), np.array(predictions))

        except Exception as e:
            logger.error(f"LSTM training failed: {e}", exc_info=True)
            return {"error": str(e)}

    def train_random_forest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trains and evaluates the Random Forest model."""
        try:
            train_df, test_df = self._split_data(df)
            feature_columns = self._get_feature_columns(df)
            
            X_train = train_df[feature_columns].values
            y_train = train_df['close'].values
            X_test = test_df[feature_columns].values
            y_test = test_df['close'].values

            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            self.models['random_forest'] = model
            
            logger.info("Random Forest model trained and evaluated successfully.")
            return self._get_evaluation_metrics(y_test, y_pred)

        except Exception as e:
            logger.error(f"Random Forest training failed: {e}", exc_info=True)
            return {"error": str(e)}

    def train_xgboost(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trains and evaluates the XGBoost model."""
        try:
            train_df, test_df = self._split_data(df)
            feature_columns = self._get_feature_columns(df)
            X_train = train_df[feature_columns].values
            y_train = train_df['close'].values
            X_test = test_df[feature_columns].values
            y_test = test_df['close'].values

            # Use GPU if available
            tree_method = 'gpu_hist' if self.device.type == 'cuda' else 'hist'
            model = xgb.XGBRegressor(
                n_estimators=100,
                random_state=42,
                tree_method=tree_method,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            self.models['xgboost'] = model
            
            logger.info("XGBoost model trained and evaluated successfully.")
            return self._get_evaluation_metrics(y_test, y_pred)

        except Exception as e:
            logger.error(f"XGBoost training failed: {e}", exc_info=True)
            return {"error": str(e)}

    def train_transformer(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trains and evaluates the Transformer model."""
        try:
            train_df, test_df = self._split_data(df)
            feature_columns = self._get_feature_columns(df)
            
            X_train = train_df[feature_columns].values
            y_train = train_df['close'].values
            X_test = test_df[feature_columns].values
            y_test = test_df['close'].values
            
            # Initialize transformer predictor
            transformer = TransformerPredictor(
                input_dim=len(feature_columns),
                d_model=128,  # Smaller for faster training
                num_heads=4,
                num_layers=3,
                sequence_length=min(60, len(X_train) // 2),  # Adaptive sequence length
                device=str(self.device)
            )
            
            # Train the model
            history = transformer.train(
                X_train, y_train,
                epochs=50,  # Reduced for faster training
                batch_size=16,
                validation_split=0.2
            )
            
            # Make predictions on test set
            y_pred = transformer.predict(X_test, steps=1)
            
            # Handle case where prediction returns multiple steps
            if len(y_pred) > len(y_test):
                y_pred = y_pred[:len(y_test)]
            elif len(y_pred) < len(y_test):
                # If we have fewer predictions, only evaluate on available predictions
                y_test = y_test[:len(y_pred)]
            
            self.models['transformer'] = transformer
            
            logger.info("Transformer model trained and evaluated successfully.")
            return self._get_evaluation_metrics(y_test, y_pred)
            
        except Exception as e:
            logger.error(f"Transformer training failed: {e}", exc_info=True)
            return {"error": str(e)}

    def train_arima(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trains and evaluates the ARIMA model."""
        try:
            train_df, test_df = self._split_data(df)
            
            # ARIMA works on univariate time series
            train_series = train_df['close']
            test_series = test_df['close']

            model = ARIMA(train_series, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Make predictions
            predictions = fitted_model.forecast(steps=len(test_series))
            
            self.models['arima'] = fitted_model
            
            logger.info("ARIMA model trained and evaluated successfully.")
            return self._get_evaluation_metrics(test_series.values, predictions)

        except Exception as e:
            logger.error(f"ARIMA training failed: {e}", exc_info=True)
            return {"error": str(e)}

    def train_prophet(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trains and evaluates the Prophet model."""
        try:
            train_df, test_df = self._split_data(df)
            
            # Prepare data for Prophet - check if date column exists
            if 'date' in train_df.columns:
                df_prophet = train_df[['date', 'close']].copy()
            else:
                # Create date index if date column doesn't exist
                df_prophet = train_df[['close']].copy()
                df_prophet['date'] = pd.date_range(start='2024-01-01', periods=len(df_prophet), freq='D')
                df_prophet = df_prophet[['date', 'close']]
            
            df_prophet.columns = ['ds', 'y']
            
            # Remove timezone information
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)
            
            model = Prophet()
            model.fit(df_prophet)
            
            # Make predictions
            future = model.make_future_dataframe(periods=len(test_df))
            forecast = model.predict(future)
            
            # Get predictions for test period
            predictions = forecast['yhat'].tail(len(test_df)).values
            actuals = test_df['close'].values
            
            self.models['prophet'] = model
            
            logger.info("Prophet model trained and evaluated successfully.")
            return self._get_evaluation_metrics(actuals, predictions)

        except Exception as e:
            logger.error(f"Prophet training failed: {e}", exc_info=True)
            return {"error": str(e)}

    def save_models(self, symbol: str):
        """Saves all trained models and scalers for a specific stock."""
        logger.info(f"Saving models for {symbol}...")
        for name, model in self.models.items():
            path = f'data/models/{symbol.upper()}_{name}.pkl'
            joblib.dump(model, path)
        
        for name, scaler in self.scalers.items():
            path = f'data/models/{symbol.upper()}_{name}_scaler.pkl'
            joblib.dump(scaler, path)
        
        logger.info(f"Successfully saved all artifacts for {symbol}.")

    def load_models(self, symbol: str):
        """Loads all trained models and scalers for a specific stock."""
        logger.info(f"Loading models for {symbol}...")
        self.models = {}
        self.scalers = {}
        
        model_files = [
            'lstm', 'random_forest', 'xgboost', 'arima', 'prophet'
        ]
        
        for model_name in model_files:
            model_path = f'data/models/{symbol.upper()}_{model_name}.pkl'
            scaler_path = f'data/models/{symbol.upper()}_{model_name}_scaler.pkl'
            
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                logger.debug(f"Loaded {model_name} model for {symbol}")
            
            if os.path.exists(scaler_path):
                self.scalers[model_name] = joblib.load(scaler_path)
                logger.debug(f"Loaded {model_name} scaler for {symbol}")
        
        logger.info(f"Loaded {len(self.models)} models for {symbol}")

    def forecast(self, model_name: str, data: pd.DataFrame, steps: int = 1) -> List[float]:
        """Makes predictions using a specific model."""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return []
        
        try:
            # Apply feature engineering to match training data
            from src.features.feature_engineering import FeatureEngineer
            feature_engineer = FeatureEngineer()
            processed_data = feature_engineer.prepare_features(data)
            
            # Get feature columns using the same logic as training
            feature_cols = self._get_feature_columns(processed_data)
            
            if model_name == 'lstm':
                return self._forecast_lstm(processed_data, feature_cols, steps)
            elif model_name == 'random_forest':
                return self._forecast_random_forest(processed_data, feature_cols, steps)
            elif model_name == 'xgboost':
                return self._forecast_xgboost(processed_data, feature_cols, steps)
            elif model_name == 'arima':
                return self._forecast_arima(processed_data, steps)
            elif model_name == 'prophet':
                return self._forecast_prophet(processed_data, steps)
            elif model_name == 'transformer':
                return self._forecast_transformer(processed_data, feature_cols, steps)
            else:
                logger.error(f"Unknown model: {model_name}")
                return []
        except Exception as e:
            logger.error(f"Forecast failed for {model_name}: {e}")
            return []

    def _forecast_lstm(self, data: pd.DataFrame, feature_cols: List[str], steps: int) -> List[float]:
        """Makes LSTM predictions."""
        if 'lstm' not in self.models or 'lstm' not in self.scalers:
            return []
        
        model = self.models['lstm']
        scaler = self.scalers['lstm']
        
        # Start with the last known data
        current_data = data.copy()
        predictions = []
        
        model.eval()
        with torch.no_grad():
            for step in range(steps):
                # Prepare the last sequence
                X = current_data[feature_cols].values.astype(np.float64)
                X_scaled = scaler.transform(X)
                
                # Check if we have enough data for the lookback window
                # Use adaptive lookback based on available data
                adaptive_lookback = min(LOOKBACK_DAYS, len(X_scaled) - 1, 30)  # At least 30, max 60
                if len(X_scaled) < adaptive_lookback or adaptive_lookback < 10:
                    logger.warning(f"Insufficient data for LSTM prediction. Need at least 10 records, got {len(X_scaled)}")
                    break
                
                # Get the last lookback sequence
                last_sequence = X_scaled[-adaptive_lookback:].reshape(1, adaptive_lookback, -1)
                
                # Make prediction
                input_tensor = torch.FloatTensor(last_sequence).to(self.device)
                pred = model(input_tensor).cpu().numpy()[0][0]
                predictions.append(pred)
                
                # Update the data for next prediction by adding the predicted close price
                new_row = current_data.iloc[-1].copy()
                new_row['close'] = pred
                
                # Update lag features if they exist
                for col in feature_cols:
                    if 'lag_' in col:
                        try:
                            # Extract lag period from column name (e.g., 'close_lag_1' -> 1)
                            parts = col.split('_')
                            if len(parts) >= 3 and parts[-2] == 'lag':
                                lag_period = int(parts[-1])
                                if len(current_data) >= lag_period:
                                    new_row[col] = current_data['close'].iloc[-lag_period]
                        except (ValueError, IndexError):
                            # Skip if we can't parse the lag period
                            continue
                
                # Add the new row to current_data
                current_data = pd.concat([current_data, new_row.to_frame().T], ignore_index=True)
        
        return predictions

    def _forecast_random_forest(self, data: pd.DataFrame, feature_cols: List[str], steps: int) -> List[float]:
        """Makes Random Forest predictions."""
        if 'random_forest' not in self.models:
            return []
        
        model = self.models['random_forest']
        
        # Start with the last known data
        current_data = data.copy()
        predictions = []
        
        for step in range(steps):
            # Use the last row for prediction
            X = current_data[feature_cols].values
            last_features = X[-1:].reshape(1, -1)
            
            # Make prediction
            pred = model.predict(last_features)[0]
            predictions.append(pred)
            
            # Update the data for next prediction by adding the predicted close price
            # Create a new row with the predicted close price
            new_row = current_data.iloc[-1].copy()
            new_row['close'] = pred
            
            # Update lag features if they exist
            for col in feature_cols:
                if 'lag_' in col:
                    try:
                        # Extract lag period from column name (e.g., 'close_lag_1' -> 1)
                        parts = col.split('_')
                        if len(parts) >= 3 and parts[-2] == 'lag':
                            lag_period = int(parts[-1])
                        else:
                            continue
                    except (ValueError, IndexError):
                        # Skip if we can't parse the lag period
                        continue
                    if len(current_data) >= lag_period:
                        new_row[col] = current_data['close'].iloc[-lag_period]
            
            # Add the new row to current_data
            current_data = pd.concat([current_data, new_row.to_frame().T], ignore_index=True)
        
        return predictions

    def _forecast_xgboost(self, data: pd.DataFrame, feature_cols: List[str], steps: int) -> List[float]:
        """Makes XGBoost predictions."""
        if 'xgboost' not in self.models:
            return []
        
        model = self.models['xgboost']
        
        # Start with the last known data
        current_data = data.copy()
        predictions = []
        
        for step in range(steps):
            # Use the last row for prediction
            X = current_data[feature_cols].values
            last_features = X[-1:].reshape(1, -1)
            
            # Make prediction
            pred = model.predict(last_features)[0]
            predictions.append(pred)
            
            # Update the data for next prediction by adding the predicted close price
            # Create a new row with the predicted close price
            new_row = current_data.iloc[-1].copy()
            new_row['close'] = pred
            
            # Update lag features if they exist
            for col in feature_cols:
                if 'lag_' in col:
                    try:
                        # Extract lag period from column name (e.g., 'close_lag_1' -> 1)
                        parts = col.split('_')
                        if len(parts) >= 3 and parts[-2] == 'lag':
                            lag_period = int(parts[-1])
                        else:
                            continue
                    except (ValueError, IndexError):
                        # Skip if we can't parse the lag period
                        continue
                    if len(current_data) >= lag_period:
                        new_row[col] = current_data['close'].iloc[-lag_period]
            
            # Add the new row to current_data
            current_data = pd.concat([current_data, new_row.to_frame().T], ignore_index=True)
        
        return predictions

    def _forecast_arima(self, data: pd.DataFrame, steps: int) -> List[float]:
        """Makes ARIMA predictions."""
        if 'arima' not in self.models:
            return []
        
        model = self.models['arima']
        predictions = model.forecast(steps=steps)
        return predictions.tolist()

    def _forecast_prophet(self, data: pd.DataFrame, steps: int) -> List[float]:
        """Makes Prophet predictions."""
        if 'prophet' not in self.models:
            return []
        
        model = self.models['prophet']
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        
        # Get the last 'steps' predictions
        predictions = forecast['yhat'].tail(steps).values.tolist()
        return predictions

    def _forecast_transformer(self, data: pd.DataFrame, feature_cols: List[str], steps: int) -> List[float]:
        """Makes Transformer predictions."""
        if 'transformer' not in self.models:
            return []
        
        transformer = self.models['transformer']
        
        try:
            # Prepare data for transformer
            X = data[feature_cols].values.astype(np.float64)
            
            # Make predictions
            predictions = transformer.predict(X, steps=steps)
            
            return predictions.tolist()
            
        except Exception as e:
            logger.error(f"Transformer forecast error: {e}")
            return []