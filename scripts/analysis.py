#!/usr/bin/env python3
"""
Direct Stock Analysis Script (Refactored)

Use trained models directly for prediction and analysis without an API.
Perfect for personal use, research, and interactive demos.
"""

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

# --- Project Structure Setup ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import your updated modules
from data.dataset_manager import nse_dataset_manager
from features.feature_engineering import FeatureEngineer
from models.models import StockPredictor

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class AnalysisConfig:
    """Centralized configuration for the analysis script."""
    MODEL_DIR: Path = Path("data/models")
    MODEL_TYPES: List[str] = field(default_factory=lambda: ['lstm', 'random_forest', 'xgboost', 'arima', 'prophet'])
    
    # Days of historical data to load for making a next-day prediction.
    FORECAST_HISTORY_DAYS: int = 100
    
    # Years of data for calculating long-term stats (e.g., 52-week high/low).
    ANALYSIS_YEARS: List[str] = field(default_factory=lambda: ['2023', '2024'])


# =============================================================================
# 2. STOCK ANALYSIS ENGINE
# =============================================================================

class StockAnalyzer:
    """
    An efficient, direct-use engine for stock prediction and analysis.
    """
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.nse_manager = nse_dataset_manager
        self.feature_engineer = FeatureEngineer()
        # Caches to avoid redundant file I/O and processing
        self._predictors: Dict[str, StockPredictor] = {}
        self._processed_data: Dict[str, pd.DataFrame] = {}

    def _load_predictor(self, symbol: str) -> Optional[StockPredictor]:
        """Loads and caches the StockPredictor object for a given stock."""
        if symbol not in self._predictors:
            logger.info(f"Loading models for {symbol}...")
            predictor = StockPredictor()
            predictor.load_models(symbol)
            if not predictor.models:
                logger.error(f"No models found on disk for {symbol}.")
                return None
            self._predictors[symbol] = predictor
        return self._predictors[symbol]

    def _get_prepared_data(self, symbol: str, years: Optional[List[str]] = None, days: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Loads, processes, and caches historical data to avoid redundant work."""
        cache_key = f"{symbol}_{'|'.join(years) if years else days}"
        if cache_key in self._processed_data:
            return self._processed_data[cache_key]
            
        logger.info(f"Fetching and preparing data for {symbol}...")
        data = self.nse_manager.load_recent_data(symbol, days=days) if days else self.nse_manager.load_multi_year_data(symbol, years)
        
        if data.empty:
            logger.warning(f"No data available for {symbol}.")
            return None
            
        processed_data = self.feature_engineer.prepare_features(data)
        self._processed_data[cache_key] = processed_data
        return processed_data

    def predict_stock_price(self, symbol: str, model_name: str, days_ahead: int = 1) -> Dict[str, Any]:
        """Predicts future stock prices for a single model."""
        try:
            predictor = self._load_predictor(symbol)
            if not predictor or model_name not in predictor.models:
                available = list(predictor.models.keys()) if predictor else []
                return {"status": "failed", "error": f"Model '{model_name}' not available for {symbol}.", "available_models": available}

            data = self._get_prepared_data(symbol, days=self.config.FORECAST_HISTORY_DAYS)
            if data is None or data.empty:
                return {"status": "failed", "error": "Could not retrieve data for prediction."}

            predictions = predictor.forecast(model_name, data, days_ahead)
            if not predictions:
                return {"status": "failed", "error": "Model failed to generate a prediction."}

            current_price = data['close'].iloc[-1]
            price_changes = [((p - current_price) / current_price) * 100 for p in predictions]
            
            return {
                "status": "success", "symbol": symbol, "model": model_name, "current_price": round(current_price, 2),
                "predictions": [round(p, 2) for p in predictions], "price_changes_percent": [round(c, 2) for c in price_changes]
            }
        except Exception as e:
            logger.error(f"Prediction failed for {symbol} with {model_name}: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def compare_models(self, symbol: str, days_ahead: int = 1) -> Dict[str, Any]:
        """Efficiently compares predictions from all available models for a stock."""
        try:
            predictor = self._load_predictor(symbol)
            if not predictor:
                return {"status": "failed", "error": "No models available for this stock."}
            
            # EFFICIENCY: Load and process data only ONCE.
            data = self._get_prepared_data(symbol, days=self.config.FORECAST_HISTORY_DAYS)
            if data is None or data.empty:
                return {"status": "failed", "error": "Could not retrieve data for comparison."}

            results = {}
            all_forecasts = []
            for model_name in predictor.models.keys():
                forecast = predictor.forecast(model_name, data, days_ahead)
                if forecast:
                    results[model_name] = forecast
                    all_forecasts.append(forecast)

            if not all_forecasts:
                return {"status": "failed", "error": "None of the models could generate a forecast."}
            
            # Calculate an ensemble (average) prediction for each day ahead
            ensemble_predictions = np.mean(np.array(all_forecasts), axis=0).tolist()

            return {
                "status": "success", "symbol": symbol, "current_price": round(data['close'].iloc[-1], 2),
                "model_predictions": results, "ensemble_predictions": [round(p, 2) for p in ensemble_predictions]
            }
        except Exception as e:
            logger.error(f"Model comparison failed for {symbol}: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
            
    def get_available_stocks_with_models(self) -> Dict[str, List[str]]:
        """Efficiently finds all stocks that have at least one trained model file."""
        logger.info("Scanning for stocks with trained models...")
        stocks_with_models = {}
        # OPTIMIZATION: Check for file existence directly, which is much faster.
        for stock_file in self.config.MODEL_DIR.glob('*.pkl'):
            symbol = stock_file.stem.split('_')[0]
            model_name = stock_file.stem.split('_')[1]
            if symbol not in stocks_with_models:
                stocks_with_models[symbol] = []
            if model_name in self.config.MODEL_TYPES:
                 stocks_with_models[symbol].append(model_name)
        logger.info(f"Found {len(stocks_with_models)} stocks with trained models.")
        return stocks_with_models


def interactive_analysis_tool(args=None):
    """An interactive command-line tool for stock analysis."""
    print("\n" + "="*60 + "\n          INTERACTIVE STOCK ANALYSIS TOOL\n" + "="*60)
    
    config = AnalysisConfig()
    analyzer = StockAnalyzer(config)
    
    stocks_info = analyzer.get_available_stocks_with_models()
    if not stocks_info:
        print("No stocks with trained models found. Please run the training script first.")
        return
        
    available_stocks = sorted(stocks_info.keys())
    print(f"Analysis available for {len(available_stocks)} stocks.")

    while True:
        print("\n" + "-"*60)
        print("Available stocks:", ", ".join(available_stocks[:10]) + ", ..." if len(available_stocks) > 10 else ", ".join(available_stocks))
        symbol = input("Enter a stock symbol to analyze (or 'quit' to exit): ").strip().upper()
        
        if symbol == 'QUIT':
            break
        if symbol not in available_stocks:
            print(f"Error: No models found for '{symbol}'. Please choose an available stock.")
            continue
        
        print(f"\nAnalyzing {symbol} (Models available: {', '.join(stocks_info[symbol])})...\n")
        
        comparison = analyzer.compare_models(symbol, days_ahead=5) # 5-day forecast
        
        if comparison.get("status") == "success":
            current_price = comparison['current_price']
            print(f"📊 5-Day Forecast (Current Price: ₹{current_price:.2f})")
            
            for model_name, preds in comparison['model_predictions'].items():
                pred_day1 = preds[0]
                change = ((pred_day1 - current_price) / current_price) * 100
                arrow = "▲" if change > 0 else "▼"
                print(f"  - {model_name:<15} | Day 1: ₹{pred_day1:.2f} ({change:+.2f}%) {arrow}")
            
            ensemble = comparison['ensemble_predictions']
            ensemble_change = ((ensemble[0] - current_price) / current_price) * 100
            print("-" * 50)
            print(f"  - {'ENSEMBLE (AVG)':<15} | Day 1: ₹{ensemble[0]:.2f} ({ensemble_change:+.2f}%)")
            print(f"  - {'Full 5-Day Ensemble':<15} | {', '.join([f'₹{p:.2f}' for p in ensemble])}")
        else:
            print(f"Could not perform analysis: {comparison.get('error')}")

    print("\n" + "="*60 + "\nExiting Analysis Tool. Thank you!\n" + "="*60)


if __name__ == "__main__":
    interactive_analysis_tool()