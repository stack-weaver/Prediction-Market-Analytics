"""
NSE Dataset Manager (Refactored)
Manages the comprehensive NSE dataset (2022-2024) for stock prediction
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class NSEDatasetManager:
    """
    Manages the comprehensive NSE dataset (2022-2024)
    Refactored for better performance and maintainability
    """
    
    def __init__(self):
        self.data_dir = Path("data/raw/dataset")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Available years
        self.years = ["2022", "2023", "2024"]
        
        # NSE Stock Universe
        self.nse_stocks = [
            "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
            "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BHARTIARTL", "BPCL",
            "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
            "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
            "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK",
            "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT",
            "LTIM", "MARUTI", "MM", "NESTLEIND", "NTPC",
            "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
            "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS",
            "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO"
        ]
        
        # Stock metadata
        self.stock_metadata = {
            "RELIANCE": {"sector": "Energy", "market_cap": "Large Cap"},
            "TCS": {"sector": "Technology", "market_cap": "Large Cap"},
            "HDFCBANK": {"sector": "Banking", "market_cap": "Large Cap"},
            "INFY": {"sector": "Technology", "market_cap": "Large Cap"},
            "HINDUNILVR": {"sector": "FMCG", "market_cap": "Large Cap"},
            "ICICIBANK": {"sector": "Banking", "market_cap": "Large Cap"},
            "KOTAKBANK": {"sector": "Banking", "market_cap": "Large Cap"},
            "SBIN": {"sector": "Banking", "market_cap": "Large Cap"},
            "BHARTIARTL": {"sector": "Telecom", "market_cap": "Large Cap"},
            "ITC": {"sector": "FMCG", "market_cap": "Large Cap"},
            "ASIANPAINT": {"sector": "Paints", "market_cap": "Large Cap"},
            "MARUTI": {"sector": "Automobile", "market_cap": "Large Cap"},
            "TITAN": {"sector": "Jewelry", "market_cap": "Large Cap"},
            "NESTLEIND": {"sector": "FMCG", "market_cap": "Large Cap"},
            "ULTRACEMCO": {"sector": "Cement", "market_cap": "Large Cap"}
        }
        
        # Cache for loaded data
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info(f"NSEDatasetManager initialized with {len(self.nse_stocks)} stocks")
    
    def get_available_stocks(self) -> List[str]:
        """Returns list of available stock symbols"""
        return self.nse_stocks.copy()
    
    def get_available_years(self) -> List[str]:
        """Returns list of available years"""
        return self.years.copy()
    
    def load_stock_data(self, symbol: str, year: str) -> pd.DataFrame:
        """Load historical data for a single stock for a specific year"""
        try:
            filepath_day = self.data_dir / year / 'day' / f"{symbol}_day.csv"
            filepath_minute = self.data_dir / year / 'minute' / f"{symbol}_minute.csv"
            
            if filepath_day.exists():
                df = pd.read_csv(filepath_day)
                df['date'] = pd.to_datetime(df['date'])
                df['symbol'] = symbol
                df['sector'] = self.stock_metadata.get(symbol, {}).get('sector', 'N/A')
                logger.debug(f"Loaded {len(df)} records for {symbol} ({year})")
                return df
            elif filepath_minute.exists():
                df = pd.read_csv(filepath_minute)
                df['date'] = pd.to_datetime(df['date_time'], format='%Y%m%d_%H%M')
                df['symbol'] = symbol
                df['sector'] = self.stock_metadata.get(symbol, {}).get('sector', 'N/A')
                logger.debug(f"Loaded {len(df)} records for {symbol} ({year})")
                return df
            else:
                logger.warning(f"No data file found for {symbol} in {year}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading data for {symbol} ({year}): {e}")
            return pd.DataFrame()
    
    def load_multi_year_data(self, symbol: str, years: List[str]) -> pd.DataFrame:
        """Load historical data for a stock across multiple years"""
        cache_key = f"{symbol}_{'_'.join(years)}"
        
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        all_data = []
        for year in years:
            data = self.load_stock_data(symbol, year)
            if not data.empty:
                all_data.append(data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values('date').reset_index(drop=True)
            self._data_cache[cache_key] = combined_data
            logger.info(f"Loaded {len(combined_data)} total records for {symbol} across {len(years)} years")
            return combined_data
        else:
            logger.warning(f"No data found for {symbol} in any of the specified years")
            return pd.DataFrame()
    
    def load_recent_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Load recent data for a stock (last N days)"""
        try:
            # Load data from the most recent year
            recent_year = self.years[-1]
            data = self.load_stock_data(symbol, recent_year)
            
            if not data.empty:
                # Get the last N days
                recent_data = data.tail(days)
                logger.debug(f"Loaded {len(recent_data)} recent records for {symbol}")
                return recent_data
            else:
                logger.warning(f"No recent data found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading recent data for {symbol}: {e}")
            return pd.DataFrame()
    
    def load_portfolio_data(self, symbols: List[str], years: List[str]) -> Dict[str, pd.DataFrame]:
        """Load data for multiple stocks (portfolio)"""
        portfolio_data = {}
        
        for symbol in symbols:
            data = self.load_multi_year_data(symbol, years)
            if not data.empty:
                portfolio_data[symbol] = data
            else:
                logger.warning(f"No data available for {symbol}")
        
        logger.info(f"Loaded portfolio data for {len(portfolio_data)} stocks")
        return portfolio_data
    
    def get_data_summary(self) -> Dict[str, any]:
        """Get summary statistics of the dataset"""
        try:
            summary = {
                "total_stocks": len(self.nse_stocks),
                "available_years": self.years,
                "data_directory": str(self.data_dir),
                "stocks_by_sector": {},
                "total_records": 0
            }   
            
            # Count stocks by sector
            for symbol in self.nse_stocks:
                sector = self.stock_metadata.get(symbol, {}).get('sector', 'Unknown')
                summary["stocks_by_sector"][sector] = summary["stocks_by_sector"].get(sector, 0) + 1
            
            # Estimate total records (approximate)
            for year in self.years:
                year_dir = self.data_dir / year / 'day'
                if year_dir.exists():
                    files = list(year_dir.glob('*.csv'))
                    summary["total_records"] += len(files) * 250  # Approximate records per file
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {e}")
            return {"error": str(e)}
    
    def clear_cache(self):
        """Clear the data cache to free memory"""
        self._data_cache.clear()
        logger.info("Data cache cleared")

# Global instance for easy import
nse_dataset_manager = NSEDatasetManager()