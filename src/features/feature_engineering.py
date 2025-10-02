"""
Data Preprocessing and Feature Engineering Module (Refactored)
Handles technical indicators, feature engineering, and data preparation for ML models
"""

import pandas as pd
import numpy as np
import talib
from typing import List, Dict, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Calculate various technical indicators for stock data
    Refactored for better performance and maintainability
    """
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            return pd.Series(talib.RSI(prices.values.astype(np.float64), timeperiod=period))
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            return pd.Series([np.nan] * len(prices))
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            macd, macd_signal, macd_hist = talib.MACD(
                prices.values.astype(np.float64), 
                fastperiod=fast, 
                slowperiod=slow, 
                signalperiod=signal
            )
            return pd.Series(macd), pd.Series(macd_signal), pd.Series(macd_hist)
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}")
            nan_series = pd.Series([np.nan] * len(prices))
            return nan_series, nan_series, nan_series
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            upper, middle, lower = talib.BBANDS(
                prices.values.astype(np.float64), 
                timeperiod=period, 
                nbdevup=std_dev, 
                nbdevdn=std_dev
            )
            return pd.Series(upper), pd.Series(middle), pd.Series(lower)
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed: {e}")
            nan_series = pd.Series([np.nan] * len(prices))
            return nan_series, nan_series, nan_series
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        try:
            return pd.Series(talib.SMA(prices.values.astype(np.float64), timeperiod=period))
        except Exception as e:
            logger.warning(f"SMA calculation failed: {e}")
            return pd.Series([np.nan] * len(prices))
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        try:
            return pd.Series(talib.EMA(prices.values.astype(np.float64), timeperiod=period))
        except Exception as e:
            logger.warning(f"EMA calculation failed: {e}")
            return pd.Series([np.nan] * len(prices))
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        try:
            return pd.Series(talib.OBV(close.values.astype(np.float64), volume.values.astype(np.float64)))
        except Exception as e:
            logger.warning(f"OBV calculation failed: {e}")
            return pd.Series([np.nan] * len(close))
    
    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        try:
            return pd.Series(talib.WILLR(
                high.values.astype(np.float64), 
                low.values.astype(np.float64), 
                close.values.astype(np.float64), 
                timeperiod=period
            ))
        except Exception as e:
            logger.warning(f"Williams %R calculation failed: {e}")
            return pd.Series([np.nan] * len(close))
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Commodity Channel Index"""
        try:
            return pd.Series(talib.CCI(
                high.values.astype(np.float64), 
                low.values.astype(np.float64), 
                close.values.astype(np.float64), 
                timeperiod=period
            ))
        except Exception as e:
            logger.warning(f"CCI calculation failed: {e}")
            return pd.Series([np.nan] * len(close))
    
    @staticmethod
    def calculate_ichimoku_a(high: pd.Series, low: pd.Series) -> pd.Series:
        """Calculate Ichimoku A line (Tenkan-sen + Kijun-sen) / 2"""
        try:
            # Calculate Tenkan-sen (9-period high + low) / 2
            tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
            # Calculate Kijun-sen (26-period high + low) / 2  
            kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
            # Ichimoku A = (Tenkan-sen + Kijun-sen) / 2
            return (tenkan_sen + kijun_sen) / 2
        except Exception as e:
            logger.warning(f"Ichimoku A calculation failed: {e}")
            return pd.Series([np.nan] * len(high))
    
    @staticmethod
    def calculate_ichimoku_b(high: pd.Series, low: pd.Series) -> pd.Series:
        """Calculate Ichimoku B line (52-period high + low) / 2"""
        try:
            return (high.rolling(52).max() + low.rolling(52).min()) / 2
        except Exception as e:
            logger.warning(f"Ichimoku B calculation failed: {e}")
            return pd.Series([np.nan] * len(high))
    
    @staticmethod
    def calculate_ichimoku_base(high: pd.Series, low: pd.Series) -> pd.Series:
        """Calculate Ichimoku Base line (Kijun-sen)"""
        try:
            return (high.rolling(26).max() + low.rolling(26).min()) / 2
        except Exception as e:
            logger.warning(f"Ichimoku Base calculation failed: {e}")
            return pd.Series([np.nan] * len(high))
    
    @staticmethod
    def calculate_ichimoku_conversion(high: pd.Series, low: pd.Series) -> pd.Series:
        """Calculate Ichimoku Conversion line (Tenkan-sen)"""
        try:
            return (high.rolling(9).max() + low.rolling(9).min()) / 2
        except Exception as e:
            logger.warning(f"Ichimoku Conversion calculation failed: {e}")
            return pd.Series([np.nan] * len(high))
    
    @staticmethod
    def calculate_psar(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Parabolic SAR"""
        try:
            return pd.Series(talib.SAR(high.values.astype(np.float64), low.values.astype(np.float64)))
        except Exception as e:
            logger.warning(f"PSAR calculation failed: {e}")
            return pd.Series([np.nan] * len(close))
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            return pd.Series(talib.ATR(high.values.astype(np.float64), low.values.astype(np.float64), close.values.astype(np.float64), timeperiod=period))
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}")
            return pd.Series([np.nan] * len(close))
    
    @staticmethod
    def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        try:
            return pd.Series(talib.MFI(high.values.astype(np.float64), low.values.astype(np.float64), close.values.astype(np.float64), volume.values.astype(np.float64), timeperiod=period))
        except Exception as e:
            logger.warning(f"MFI calculation failed: {e}")
            return pd.Series([np.nan] * len(close))
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        try:
            slowk, slowd = talib.STOCH(high.values.astype(np.float64), low.values.astype(np.float64), close.values.astype(np.float64), 
                                     fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
            return pd.Series(slowk), pd.Series(slowd)
        except Exception as e:
            logger.warning(f"Stochastic calculation failed: {e}")
            nan_series = pd.Series([np.nan] * len(close))
            return nan_series, nan_series
    
    @staticmethod
    def calculate_roc(close: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Rate of Change"""
        try:
            return pd.Series(talib.ROC(close.values.astype(np.float64), timeperiod=period))
        except Exception as e:
            logger.warning(f"ROC calculation failed: {e}")
            return pd.Series([np.nan] * len(close))
    
    @staticmethod
    def calculate_proc(close: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Price Rate of Change"""
        try:
            # PROC = (Close - Close[n-periods ago]) / Close[n-periods ago] * 100
            return ((close - close.shift(period)) / close.shift(period)) * 100
        except Exception as e:
            logger.warning(f"PROC calculation failed: {e}")
            return pd.Series([np.nan] * len(close))
    
    @staticmethod
    def calculate_ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Ultimate Oscillator"""
        try:
            return pd.Series(talib.ULTOSC(high.values.astype(np.float64), low.values.astype(np.float64), close.values.astype(np.float64)))
        except Exception as e:
            logger.warning(f"Ultimate Oscillator calculation failed: {e}")
            return pd.Series([np.nan] * len(close))


class FeatureEngineer:
    """
    Main class for feature engineering and data preprocessing
    Refactored for better performance and maintainability
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'Upper_Bollinger', 'Middle_Bollinger', 'Lower_Bollinger',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
            'OBV', 'Williams_R', 'CCI'
        ]
        self.target_column = 'close'
        logger.info("FeatureEngineer initialized")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the raw stock data
        """
        try:
            # Make a copy to avoid modifying original data
            df_clean = df.copy()
            
            # Ensure date column is datetime
            if 'date' in df_clean.columns:
                df_clean['date'] = pd.to_datetime(df_clean['date'])
                df_clean = df_clean.set_index('date')
            
            # Convert numeric columns to float64
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype(np.float64)
            
            # Remove rows with missing values in essential columns
            df_clean = df_clean.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            
            # Sort by date
            df_clean = df_clean.sort_index()
            
            # Remove duplicates
            df_clean = df_clean.drop_duplicates()
            
            logger.info(f"Data cleaned: {len(df_clean)} records remaining")
            return df_clean
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe
        """
        try:
            df_indicators = df.copy()
            
            # Calculate RSI
            df_indicators['RSI'] = TechnicalIndicators.calculate_rsi(df_indicators['close'])
            
            # Calculate MACD
            macd, macd_signal, macd_hist = TechnicalIndicators.calculate_macd(df_indicators['close'])
            df_indicators['MACD'] = macd
            df_indicators['MACD_signal'] = macd_signal
            df_indicators['MACD_hist'] = macd_hist
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(df_indicators['close'])
            df_indicators['Upper_Bollinger'] = bb_upper
            df_indicators['Middle_Bollinger'] = bb_middle
            df_indicators['Lower_Bollinger'] = bb_lower
            
            # Calculate Moving Averages
            for period in [5, 10, 20, 50]:
                df_indicators[f'SMA_{period}'] = TechnicalIndicators.calculate_sma(df_indicators['close'], period)
                df_indicators[f'EMA_{period}'] = TechnicalIndicators.calculate_ema(df_indicators['close'], period)
            
            # Calculate OBV
            df_indicators['OBV'] = TechnicalIndicators.calculate_obv(df_indicators['close'], df_indicators['volume'])
            
            # Calculate Williams %R
            df_indicators['Williams_R'] = TechnicalIndicators.calculate_williams_r(
                df_indicators['high'], df_indicators['low'], df_indicators['close']
            )
            
            # Calculate CCI
            df_indicators['CCI'] = TechnicalIndicators.calculate_cci(
                df_indicators['high'], df_indicators['low'], df_indicators['close']
            )
            
            # Add advanced technical indicators
            df_indicators = self._add_advanced_indicators(df_indicators)
            
            logger.info(f"Technical indicators added: {len(df_indicators.columns)} columns")
            return df_indicators
            
        except Exception as e:
            logger.error(f"Technical indicators calculation failed: {e}")
            return df
    
    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced technical indicators for better model performance
        """
        try:
            df_advanced = df.copy()
            
            # Ichimoku Cloud components
            df_advanced['Ichimoku_A'] = TechnicalIndicators.calculate_ichimoku_a(df_advanced['high'], df_advanced['low'])
            df_advanced['Ichimoku_B'] = TechnicalIndicators.calculate_ichimoku_b(df_advanced['high'], df_advanced['low'])
            df_advanced['Ichimoku_Base'] = TechnicalIndicators.calculate_ichimoku_base(df_advanced['high'], df_advanced['low'])
            df_advanced['Ichimoku_Conversion'] = TechnicalIndicators.calculate_ichimoku_conversion(df_advanced['high'], df_advanced['low'])
            
            # Parabolic SAR
            df_advanced['PSAR'] = TechnicalIndicators.calculate_psar(df_advanced['high'], df_advanced['low'], df_advanced['close'])
            
            # Average True Range
            df_advanced['ATR'] = TechnicalIndicators.calculate_atr(df_advanced['high'], df_advanced['low'], df_advanced['close'])
            
            # Commodity Channel Index
            df_advanced['CCI_20'] = TechnicalIndicators.calculate_cci(df_advanced['high'], df_advanced['low'], df_advanced['close'], 20)
            
            # Money Flow Index
            df_advanced['MFI'] = TechnicalIndicators.calculate_mfi(df_advanced['high'], df_advanced['low'], df_advanced['close'], df_advanced['volume'])
            
            # Stochastic Oscillator
            df_advanced['Stoch_K'], df_advanced['Stoch_D'] = TechnicalIndicators.calculate_stochastic(df_advanced['high'], df_advanced['low'], df_advanced['close'])
            
            # Rate of Change
            df_advanced['ROC'] = TechnicalIndicators.calculate_roc(df_advanced['close'])
            
            # Price Rate of Change
            df_advanced['PROC'] = TechnicalIndicators.calculate_proc(df_advanced['close'])
            
            # Ultimate Oscillator
            df_advanced['UO'] = TechnicalIndicators.calculate_ultimate_oscillator(df_advanced['high'], df_advanced['low'], df_advanced['close'])
            
            logger.info("Advanced technical indicators added")
            return df_advanced
            
        except Exception as e:
            logger.error(f"Advanced indicators calculation failed: {e}")
            return df
    
    def add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Add lag features for time series prediction
        """
        try:
            df_lags = df.copy()
            
            for lag in lags:
                df_lags[f'close_lag_{lag}'] = df_lags['close'].shift(lag)
                df_lags[f'volume_lag_{lag}'] = df_lags['volume'].shift(lag)
            
            logger.info(f"Lag features added: {len(lags)} lags")
            return df_lags
            
        except Exception as e:
            logger.error(f"Lag features calculation failed: {e}")
            return df
    
    def add_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add target variables for prediction
        """
        try:
            df_targets = df.copy()
            
            # Next day close price
            df_targets['next_close'] = df_targets['close'].shift(-1)
            
            # Price change percentage
            df_targets['price_change'] = df_targets['close'].pct_change()
            
            # Volatility (rolling standard deviation)
            df_targets['volatility'] = df_targets['price_change'].rolling(window=20).std()
            
            logger.info("Target variables added")
            return df_targets
            
        except Exception as e:
            logger.error(f"Target variables calculation failed: {e}")
            return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        """
        try:
            logger.info("Starting feature engineering pipeline...")
            
            # Step 1: Clean data
            df_processed = self.clean_data(df)
            
            # Step 2: Add technical indicators
            df_processed = self.add_technical_indicators(df_processed)
            
            # Step 3: Add lag features
            df_processed = self.add_lag_features(df_processed)
            
            # Step 4: Add target variables
            df_processed = self.add_target_variables(df_processed)
            
            # Step 5: Remove rows with NaN values (only essential columns)
            # Keep rows where basic price data exists, allow NaN in technical indicators
            essential_columns = ['open', 'high', 'low', 'close', 'volume']
            df_processed = df_processed.dropna(subset=essential_columns)
            
            # Fill remaining NaN values with more robust methods
            # Forward fill first, then backward fill for any remaining NaNs
            df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
            
            # If any NaN values still remain, fill with 0 (last resort)
            df_processed = df_processed.fillna(0)
            
            logger.info(f"Feature engineering complete: {len(df_processed)} records, {len(df_processed.columns)} features")
            return df_processed
            
        except Exception as e:
            logger.error(f"Feature engineering pipeline failed: {e}")
            return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns"""
        return self.feature_columns.copy()
    
    def get_target_column(self) -> str:
        """Get target column name"""
        return self.target_column
    
    def scale_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Scale features using StandardScaler
        """
        try:
            df_scaled = df.copy()
            
            # Get numeric columns only
            numeric_columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
            
            if fit_scaler:
                df_scaled[numeric_columns] = self.scaler.fit_transform(df_scaled[numeric_columns])
            else:
                df_scaled[numeric_columns] = self.scaler.transform(df_scaled[numeric_columns])
            
            logger.info(f"Features scaled: {len(numeric_columns)} columns")
            return df_scaled
            
        except Exception as e:
            logger.error(f"Feature scaling failed: {e}")
            return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> pd.DataFrame:
        """
        Select top k features using statistical tests
        """
        try:
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            
            logger.info(f"Feature selection complete: {len(selected_features)} features selected")
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return X