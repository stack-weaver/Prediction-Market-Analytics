# Configuration settings for Indian Stock Price Prediction ML Project

import os
from typing import List, Dict, Any
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database Configuration
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/stock_prediction"
    REDIS_URL: str = "redis://localhost:6379"
    
    # API Keys (Not used - dataset-based project)
    # ALPHA_VANTAGE_API_KEY: str = ""
    # NEWS_API_KEY: str = ""
    # GROWW_API_TOKEN: str = ""
    
    # Indian Stock Configuration (NSE Dataset Format)
    NIFTY_50_STOCKS: List[str] = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
        "ICICIBANK", "KOTAKBANK", "ITC", "SBIN", "BHARTIARTL",
        "ASIANPAINT", "MARUTI", "AXISBANK", "LT", "WIPRO",
        "ULTRACEMCO", "NESTLEIND", "SUNPHARMA", "TITAN", "POWERGRID",
        "NTPC", "ONGC", "TECHM", "TATAMOTORS", "BAJFINANCE",
        "HCLTECH", "DRREDDY", "JSWSTEEL", "BAJAJFINSV", "COALINDIA",
        "GRASIM", "BRITANNIA", "CIPLA", "EICHERMOT", "HEROMOTOCO",
        "INDUSINDBK", "UPL", "DIVISLAB", "APOLLOHOSP", "TATASTEEL",
        "ADANIPORTS", "ADANIENT", "ADANIGREEN", "ADANITRANS", "ADANIPOWER",
        "ADANITOTAL", "ADANIGAS", "MM", "LTIM", "SBILIFE"
    ]
    
    SENSEX_STOCKS: List[str] = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
        "ICICIBANK", "KOTAKBANK", "ITC", "SBIN", "BHARTIARTL",
        "ASIANPAINT", "MARUTI", "AXISBANK", "LT", "WIPRO",
        "ULTRACEMCO", "NESTLEIND", "SUNPHARMA", "TITAN", "POWERGRID",
        "NTPC", "ONGC", "TECHM", "TATAMOTORS", "BAJFINANCE",
        "HCLTECH", "DRREDDY", "JSWSTEEL", "BAJAJFINSV", "COALINDIA"
    ]
    
    # Model Configuration
    LOOKBACK_DAYS: int = 60
    PREDICTION_DAYS: int = 5
    TRAIN_TEST_SPLIT: float = 0.8
    
    # Technical Indicators
    TECHNICAL_INDICATORS: Dict[str, Any] = {
        "RSI_PERIOD": 14,
        "MACD_FAST": 12,
        "MACD_SLOW": 26,
        "MACD_SIGNAL": 9,
        "BOLLINGER_PERIOD": 20,
        "BOLLINGER_STD": 2,
        "SMA_PERIODS": [5, 10, 20, 50, 200],
        "EMA_PERIODS": [5, 10, 20, 50, 200]
    }
    
    # Risk Metrics
    RISK_CONFIG: Dict[str, Any] = {
        "VAR_CONFIDENCE": 0.05,
        "SHARPE_RISK_FREE_RATE": 0.06,  # 6% risk-free rate for India
        "MAX_PORTFOLIO_RISK": 0.15,  # 15% maximum portfolio risk
        "REBALANCE_FREQUENCY": "monthly"
    }
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # Data Collection (Not used - dataset-based project)
    # DATA_COLLECTION_INTERVAL: int = 300  # 5 minutes
    # MAX_RETRIES: int = 3
    # REQUEST_TIMEOUT: int = 30
    
    # Model Training
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    EARLY_STOPPING_PATIENCE: int = 10
    
    # Sentiment Analysis (Not used - dataset-based project)
    # SENTIMENT_SOURCES: List[str] = [
    #     "economic_times",
    #     "business_standard", 
    #     "money_control",
    #     "livemint",
    #     "financial_express"
    # ]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/stock_prediction.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Model paths
MODEL_PATHS = {
    "LSTM": "data/models/lstm_model.pkl",
    "ARIMA": "data/models/arima_model.pkl", 
    "RANDOM_FOREST": "data/models/rf_model.pkl",
    "XGBOOST": "data/models/xgb_model.pkl",
    "PROPHET": "data/models/prophet_model.pkl"
}

# Feature columns for ML models
FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "rsi", "macd", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
    "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
    "price_change", "volume_change", "volatility",
    "sentiment_score", "news_count"
]

# Target columns
TARGET_COLUMNS = [
    "close_next_1d", "close_next_3d", "close_next_5d",
    "price_direction", "volatility_next"
]
