#!/usr/bin/env python3
"""
Model Testing and Evaluation Script (Refactored)

Implements a professional, walk-forward backtesting framework to rigorously
evaluate and compare time-series models.
"""

import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
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
class TestConfig:
    """Centralized configuration for the testing pipeline."""
    TEST_YEARS: List[str] = ('2024',)
    TEST_SPLIT_RATIO: float = 0.2  # Use the last 20% of the period for testing
    PLOTS_DIR: Path = Path("data/plots/test_results")

# =============================================================================
# 2. MODEL TESTING AND EVALUATION FRAMEWORK
# =============================================================================

class ModelTester:
    """A professional framework for model testing and evaluation."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.nse_manager = nse_dataset_manager
        self.feature_engineer = FeatureEngineer()
        self.predictors: Dict[str, StockPredictor] = {}
        self.config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    def load_stock_predictor(self, symbol: str) -> Optional[StockPredictor]:
        """Loads or retrieves a cached predictor for a specific stock."""
        if symbol not in self.predictors:
            logger.info(f"Loading models for {symbol} from disk...")
            predictor = StockPredictor()
            predictor.load_models(symbol)
            if not predictor.models:
                logger.warning(f"No models found for {symbol}. Cannot create predictor.")
                return None
            self.predictors[symbol] = predictor
        return self.predictors[symbol]

    def _perform_walk_forward_validation(self, predictor: StockPredictor, model_name: str, test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Simulates live trading by predicting one step at a time."""
        predictions, actuals = [], []
        
        # Ensure the test_data index is a DatetimeIndex for plotting
        if not isinstance(test_data.index, pd.DatetimeIndex):
             test_data['date'] = pd.to_datetime(test_data['date'])
             test_data = test_data.set_index('date')

        logger.info(f"Starting walk-forward validation for {len(test_data)-1} steps...")
        for i in range(len(test_data) - 1):
            # Use all data up to the current point to predict the next day
            history = test_data.iloc[:i + 1]
            try:
                pred = predictor.forecast(model_name, history, steps=1)
                if pred:
                    predictions.append(pred[0])
                    actuals.append(test_data.iloc[i + 1]['close'])
            except Exception as e:
                logger.debug(f"Prediction failed at step {i} for {model_name}: {e}")
                continue
        
        return np.array(predictions), np.array(actuals)

    def test_single_model(self, symbol: str, model_name: str, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Tests a single model's performance using walk-forward validation."""
        try:
            predictor = self.load_stock_predictor(symbol)
            if not predictor or model_name not in predictor.models:
                return {"status": "failed", "error": f"Model '{model_name}' not available for {symbol}."}

            predictions, actuals = self._perform_walk_forward_validation(predictor, model_name, test_df)

            if len(predictions) == 0:
                return {"status": "failed", "error": "No valid predictions were generated."}
            
            # Calculate metrics
            mse = np.mean((predictions - actuals) ** 2)
            pred_direction = np.diff(predictions) > 0
            actual_direction = np.diff(actuals) > 0
            
            return {
                "status": "success",
                "test_samples": len(predictions),
                "metrics": {
                    "rmse": np.sqrt(mse),
                    "mae": np.mean(np.abs(predictions - actuals)),
                    "mape": np.mean(np.abs((actuals - predictions) / actuals)) * 100,
                    "directional_accuracy": np.mean(pred_direction == actual_direction) * 100,
                },
                "predictions": predictions.tolist(),
                "actuals": actuals.tolist()
            }
        except Exception as e:
            logger.error(f"Failed to test model {model_name} for {symbol}: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def compare_models(self, symbol: str) -> Dict[str, Any]:
        """Loads data once and compares all available models for a stock."""
        logger.info(f"Starting model comparison for {symbol} using data from {self.config.TEST_YEARS}...")
        try:
            data = self.nse_manager.load_multi_year_data(symbol, self.config.TEST_YEARS)
            if data.empty:
                return {"status": "failed", "error": f"No test data available for {symbol}."}

            processed_data = self.feature_engineer.prepare_features(data)
            test_size = int(len(processed_data) * self.config.TEST_SPLIT_RATIO)
            test_data = processed_data.tail(test_size)

            predictor = self.load_stock_predictor(symbol)
            if not predictor:
                return {"status": "failed", "error": f"Could not load predictor for {symbol}."}

            available_models = list(predictor.models.keys())
            results = {model: self.test_single_model(symbol, model, test_data.copy()) for model in available_models}
            
            successful_results = {k: v for k, v in results.items() if v.get("status") == "success"}
            best_model = min(successful_results, key=lambda m: successful_results[m]["metrics"]["rmse"], default=None)

            return {
                "status": "success", "symbol": symbol, "available_models": available_models,
                "model_results": results, "best_model": {"name": best_model, "rmse": successful_results[best_model]["metrics"]["rmse"] if best_model else None},
                "test_period": f"{test_data.index[0].date()} to {test_data.index[-1].date()}"
            }
        except Exception as e:
            logger.error(f"Model comparison failed for {symbol}: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
            
    def plot_predictions(self, symbol: str, comparison_results: Dict[str, Any]):
        """Plots the best model's predictions against actual values."""
        best_model_name = comparison_results.get("best_model", {}).get("name")
        if not best_model_name:
            logger.warning(f"Cannot plot for {symbol}, no best model found.")
            return

        model_res = comparison_results["model_results"][best_model_name]
        if model_res.get("status") != "success":
            logger.warning(f"Cannot plot for {symbol}, best model '{best_model_name}' failed testing.")
            return
            
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # We need the dates for the actuals. The predictions will align with them.
        test_period_dates = pd.to_datetime(comparison_results['test_period'].split(' to '))
        date_index = pd.date_range(start=test_period_dates[0], end=test_period_dates[1], freq='B') # Business days
        
        # Align actuals and predictions to the date index
        actuals = model_res['actuals']
        predictions = model_res['predictions']
        
        # Plotting requires same length arrays
        plot_len = min(len(date_index) -1, len(actuals), len(predictions))
        
        ax.plot(date_index[1:plot_len+1], actuals[:plot_len], label='Actual Price', color='dodgerblue', linewidth=2)
        ax.plot(date_index[1:plot_len+1], predictions[:plot_len], label=f'Predicted Price ({best_model_name})', color='orangered', linestyle='--')

        ax.set_title(f'Prediction vs. Actual Price for {symbol.upper()}', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Stock Price (INR)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True)
        
        fig.tight_layout()
        plot_path = self.config.PLOTS_DIR / f"{symbol.upper()}_prediction_vs_actual.png"
        fig.savefig(plot_path)
        logger.info(f"Prediction plot saved to '{plot_path}'")
        plt.close(fig)

def main(args=None):
    """Main function to demonstrate the ModelTester."""
    print("\n" + "="*60 + "\nMODEL TESTING AND EVALUATION (REFACTORED)\n" + "="*60)
    
    config = TestConfig()
    tester = ModelTester(config)
    
    demo_symbol = "RELIANCE"
    print(f"\nRunning comparison for a single stock: {demo_symbol.upper()}")
    
    comparison = tester.compare_models(demo_symbol)
    
    if comparison.get("status") == "success":
        best = comparison["best_model"]
        print(f"  Best Model: {best['name']} (RMSE: {best['rmse']:.4f})")
        for name, res in comparison['model_results'].items():
            if res.get('status') == 'success':
                metrics = res['metrics']
                print(f"    - {name:<15} | RMSE: {metrics['rmse']:.4f}, MAPE: {metrics['mape']:.2f}%, Dir. Acc: {metrics['directional_accuracy']:.2f}%")
            else:
                print(f"    - {name:<15} | FAILED: {res.get('error')}")
        
        # Generate and save a plot for the best model
        tester.plot_predictions(demo_symbol, comparison)
    else:
        print(f"  Error during comparison for {demo_symbol}: {comparison.get('error')}")

    print("\n" + "="*60 + "\nMODEL TESTING DEMO COMPLETE!\n" + "="*60)


if __name__ == "__main__":
    main()