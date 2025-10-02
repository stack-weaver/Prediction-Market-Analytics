#!/usr/bin/env python3
"""
Simple Stock Prediction Interface (Refactored)

A direct, no-API-needed interface for generating and comparing stock predictions.
"""

import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# --- Project Structure Setup ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# In a real project, these would be direct imports.
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
class PredictionConfig:
    """Centralized configuration for the prediction script."""
    # Days of historical data to load for making a next-day prediction.
    # More days provide more context for features like moving averages.
    FORECAST_HISTORY_DAYS: int = 100
    
    # Years of data to load for calculating long-term stats like 52-week high/low.
    SUMMARY_YEARS: List[str] = field(default_factory=lambda: ['2023', '2024'])


# =============================================================================
# 2. PREDICTION INTERFACE
# =============================================================================

class SimpleStockPredictor:
    """
    An efficient, simple interface for generating stock predictions directly.
    """
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.nse_manager = nse_dataset_manager
        self.feature_engineer = FeatureEngineer()
        # Caches to avoid redundant work
        self._predictors: Dict[str, StockPredictor] = {}
        self._processed_data: Dict[str, pd.DataFrame] = {}

    def _load_predictor(self, symbol: str) -> Optional[StockPredictor]:
        """Loads and caches the predictor object for a given stock."""
        if symbol not in self._predictors:
            logger.info(f"Loading models for {symbol}...")
            predictor = StockPredictor()
            predictor.load_models(symbol)
            if not predictor.models:
                logger.error(f"No models found for {symbol}.")
                return None
            self._predictors[symbol] = predictor
        return self._predictors[symbol]

    def _get_prepared_data(self, symbol: str, years: Optional[List[str]] = None, days: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Loads, processes, and caches historical data."""
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

    def quick_predict(self, symbol: str, model_name: str) -> Dict[str, Any]:
        """Generates a quick, single-model prediction for a stock."""
        try:
            predictor = self._load_predictor(symbol)
            if not predictor or model_name not in predictor.models:
                available = list(predictor.models.keys()) if predictor else []
                return {"status": "failed", "error": f"Model '{model_name}' not found for {symbol}.", "available_models": available}

            data = self._get_prepared_data(symbol, days=self.config.FORECAST_HISTORY_DAYS)
            if data is None or data.empty:
                return {"status": "failed", "error": f"Could not retrieve data for {symbol}."}

            prediction = predictor.forecast(model_name, data, steps=1)
            if not prediction:
                return {"status": "failed", "error": "Model failed to generate a prediction."}

            current_price = data['close'].iloc[-1]
            predicted_price = prediction[0]
            change = ((predicted_price - current_price) / current_price) * 100
            
            return {
                "status": "success", "symbol": symbol, "model": model_name,
                "current_price": round(current_price, 2), "predicted_price": round(predicted_price, 2),
                "change_percent": round(change, 2), "direction": "UP" if change > 0 else "DOWN",
            }
        except Exception as e:
            logger.error(f"Quick prediction failed for {symbol}: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def compare_all_models(self, symbol: str) -> Dict[str, Any]:
        """Efficiently compares predictions from all available models."""
        try:
            predictor = self._load_predictor(symbol)
            if not predictor:
                return {"status": "failed", "error": f"No models trained for {symbol}."}

            # EFFICIENCY: Load and process data only ONCE.
            data = self._get_prepared_data(symbol, days=self.config.FORECAST_HISTORY_DAYS)
            if data is None or data.empty:
                return {"status": "failed", "error": f"Could not retrieve data for {symbol}."}

            results, predictions = {}, []
            current_price = data['close'].iloc[-1]
            
            for model_name in predictor.models.keys():
                prediction = predictor.forecast(model_name, data, steps=1)
                if prediction:
                    predicted_price = prediction[0]
                    change = ((predicted_price - current_price) / current_price) * 100
                    results[model_name] = {"predicted_price": round(predicted_price, 2), "change_percent": round(change, 2)}
                    predictions.append(predicted_price)
            
            if not predictions:
                return {"status": "failed", "error": "None of the models could make a prediction."}
            
            avg_prediction = sum(predictions) / len(predictions)
            
            return {
                "status": "success", "symbol": symbol, "current_price": round(current_price, 2),
                "model_results": results, "average_prediction": round(avg_prediction, 2),
                "consensus": "BULLISH" if avg_prediction > current_price else "BEARISH"
            }
        except Exception as e:
            logger.error(f"Model comparison failed for {symbol}: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

def interactive_demo(args=None):
    """An interactive command-line demo of the predictor."""
    print("\n" + "="*60 + "\n           SIMPLE STOCK PREDICTION DEMO\n" + "="*60)
    
    config = PredictionConfig()
    predictor = SimpleStockPredictor(config)
    
    available_stocks = predictor.nse_manager.get_available_stocks()
    print(f"✅ Models available for {len(available_stocks)} stocks.")
    print(f"   (e.g., {', '.join(available_stocks[:5])}, ...)")
    
    while True:
        print("\n" + "-"*60)
        symbol = input("Enter a stock symbol to analyze (or 'quit' to exit): ").strip().upper()
        if symbol == 'QUIT':
            break
        if symbol not in available_stocks:
            print(f"❌ Error: No models found for '{symbol}'. Please choose from the available stocks.")
            continue

        print(f"\n📈 Analyzing {symbol}...\n")
        
        # --- Run Comparison ---
        comparison = predictor.compare_all_models(symbol)
        if comparison.get("status") == "success":
            print(f"📊 Model Comparison (Current Price: ₹{comparison['current_price']:.2f})")
            for model, result in comparison['model_results'].items():
                change = result['change_percent']
                arrow = "▲" if change > 0 else "▼"
                print(f"   - {model:<15} | Predicted: ₹{result['predicted_price']:.2f} ({change:+.2f}%) {arrow}")
            
            avg_change = ((comparison['average_prediction'] - comparison['current_price']) / comparison['current_price']) * 100
            print("-" * 40)
            print(f"   - {'CONSENSUS':<15} | Average:   ₹{comparison['average_prediction']:.2f} ({avg_change:+.2f}%)")
            print(f"   - {'OVERALL VIEW':<15} | {comparison['consensus']}")
        else:
            print(f"❌ Could not perform comparison: {comparison.get('error')}")

    print("\n" + "="*60 + "\n👋 Exiting Demo. Thank you!\n" + "="*60)


if __name__ == "__main__":
    interactive_demo()