"""
Risk Assessment Module
Implements VaR, Sharpe ratio, volatility modeling with GARCH, and other risk metrics
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from arch import arch_model
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings

logger = logging.getLogger(__name__)

class RiskMetrics:
    """
    Calculate various risk metrics for stocks and portfolios
    """
    
    @staticmethod
    def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
        """
        Calculate returns from price series
        
        Args:
            prices: Price series
            method: 'simple' or 'log'
        
        Returns:
            Returns series
        """
        if method == 'simple':
            returns = prices.pct_change().dropna()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")
        
        return returns
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05, method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Returns series
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            method: 'historical', 'parametric', or 'monte_carlo'
        
        Returns:
            VaR value
        """
        if method == 'historical':
            # Historical VaR
            var = np.percentile(returns, confidence_level * 100)
        
        elif method == 'parametric':
            # Parametric VaR (assuming normal distribution)
            mean_return = returns.mean()
            std_return = returns.std()
            var = mean_return + stats.norm.ppf(confidence_level) * std_return
        
        elif method == 'monte_carlo':
            # Monte Carlo VaR
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random returns
            np.random.seed(42)
            random_returns = np.random.normal(mean_return, std_return, 10000)
            var = np.percentile(random_returns, confidence_level * 100)
        
        else:
            raise ValueError("Method must be 'historical', 'parametric', or 'monte_carlo'")
        
        return var
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        
        Args:
            returns: Returns series
            confidence_level: Confidence level
        
        Returns:
            CVaR value
        """
        var = RiskMetrics.calculate_var(returns, confidence_level, 'historical')
        cvar = returns[returns <= var].mean()
        return cvar
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06, 
                              annualized: bool = True) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate (annual)
            annualized: Whether to annualize the ratio
        
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if annualized:
            sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        else:
            sharpe = excess_returns.mean() / returns.std()
        
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.06,
                               annualized: bool = True) -> float:
        """
        Calculate Sortino ratio (downside deviation)
        
        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate
            annualized: Whether to annualize
        
        Returns:
            Sortino ratio
        """
        excess_returns = returns - risk_free_rate / 252
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_deviation == 0:
            return 0
        
        if annualized:
            sortino = np.sqrt(252) * excess_returns.mean() / downside_deviation
        else:
            sortino = excess_returns.mean() / downside_deviation
        
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown
        
        Args:
            returns: Returns series
        
        Returns:
            Dictionary with max drawdown metrics
        """
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        max_dd = drawdown.min()
        
        # Find the period of maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_start_idx = running_max.loc[:max_dd_idx].idxmax()
        
        return {
            'max_drawdown': float(max_dd),
            'max_drawdown_start': str(max_dd_start_idx),
            'max_drawdown_end': str(max_dd_idx),
            'max_drawdown_duration': int((max_dd_idx - max_dd_start_idx).days) if hasattr(max_dd_idx - max_dd_start_idx, 'days') else 0
        }
    
    @staticmethod
    def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta coefficient
        
        Args:
            stock_returns: Stock returns
            market_returns: Market returns (e.g., Nifty 50)
        
        Returns:
            Beta coefficient
        """
        # Align the series
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        stock_aligned = aligned_data.iloc[:, 0]
        market_aligned = aligned_data.iloc[:, 1]
        
        # Calculate covariance and variance
        covariance = np.cov(stock_aligned, market_aligned)[0, 1]
        market_variance = np.var(market_aligned)
        
        beta = covariance / market_variance
        return beta
    
    @staticmethod
    def calculate_treynor_ratio(returns: pd.Series, market_returns: pd.Series, 
                               risk_free_rate: float = 0.06) -> float:
        """
        Calculate Treynor ratio
        
        Args:
            returns: Stock returns
            market_returns: Market returns
            risk_free_rate: Risk-free rate
        
        Returns:
            Treynor ratio
        """
        beta = RiskMetrics.calculate_beta(returns, market_returns)
        excess_returns = returns - risk_free_rate / 252
        
        treynor = excess_returns.mean() / beta
        return treynor

class GARCHModel:
    """
    GARCH volatility modeling
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
    
    def fit(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Fit GARCH model to returns
        
        Args:
            returns: Returns series
        
        Returns:
            Model fit results
        """
        try:
            # Create GARCH model
            self.model = arch_model(returns, vol='GARCH', p=self.p, q=self.q)
            
            # Fit the model
            self.fitted_model = self.model.fit(disp='off')
            
            # Get volatility forecasts
            forecasts = self.fitted_model.forecast(horizon=1)
            volatility = np.sqrt(forecasts.variance.values[-1, 0])
            
            return {
                'success': True,
                'volatility': volatility,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'loglikelihood': self.fitted_model.loglikelihood,
                'params': self.fitted_model.params.to_dict()
            }
            
        except Exception as e:
            logger.error(f"GARCH model fitting failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'volatility': returns.std()
            }
    
    def forecast_volatility(self, horizon: int = 1) -> np.ndarray:
        """
        Forecast volatility
        
        Args:
            horizon: Forecast horizon
        
        Returns:
            Volatility forecasts
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        forecasts = self.fitted_model.forecast(horizon=horizon)
        return np.sqrt(forecasts.variance.values[-1, :])

class PortfolioRiskAnalyzer:
    """
    Portfolio-level risk analysis
    """
    
    def __init__(self):
        self.risk_free_rate = settings.RISK_CONFIG['SHARPE_RISK_FREE_RATE']
    
    def calculate_portfolio_var(self, weights: np.ndarray, returns: pd.DataFrame, 
                              confidence_level: float = 0.05) -> float:
        """
        Calculate portfolio VaR
        
        Args:
            weights: Portfolio weights
            returns: Returns DataFrame
            confidence_level: Confidence level
        
        Returns:
            Portfolio VaR
        """
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calculate VaR
        var = RiskMetrics.calculate_var(portfolio_returns, confidence_level)
        return var
    
    def calculate_portfolio_volatility(self, weights: np.ndarray, 
                                     covariance_matrix: np.ndarray) -> float:
        """
        Calculate portfolio volatility
        
        Args:
            weights: Portfolio weights
            covariance_matrix: Covariance matrix
        
        Returns:
            Portfolio volatility
        """
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        return portfolio_volatility
    
    def calculate_portfolio_sharpe(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """
        Calculate portfolio Sharpe ratio
        
        Args:
            weights: Portfolio weights
            returns: Returns DataFrame
        
        Returns:
            Portfolio Sharpe ratio
        """
        portfolio_returns = (returns * weights).sum(axis=1)
        sharpe = RiskMetrics.calculate_sharpe_ratio(portfolio_returns, self.risk_free_rate)
        return sharpe
    
    def calculate_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix
        
        Args:
            returns: Returns DataFrame
        
        Returns:
            Correlation matrix
        """
        return returns.corr()
    
    def calculate_covariance_matrix(self, returns: pd.DataFrame, 
                                   method: str = 'empirical') -> np.ndarray:
        """
        Calculate covariance matrix
        
        Args:
            returns: Returns DataFrame
            method: 'empirical' or 'ledoit_wolf'
        
        Returns:
            Covariance matrix
        """
        if method == 'empirical':
            cov_matrix = returns.cov().values
        elif method == 'ledoit_wolf':
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns).covariance_
        else:
            raise ValueError("Method must be 'empirical' or 'ledoit_wolf'")
        
        return cov_matrix

class RiskAssessment:
    """
    Main class for comprehensive risk assessment
    """
    
    def __init__(self):
        self.portfolio_analyzer = PortfolioRiskAnalyzer()
    
    def analyze_stock_risk(self, prices: pd.Series, market_prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Comprehensive risk analysis for a single stock
        
        Args:
            prices: Stock price series
            market_prices: Market price series (optional)
        
        Returns:
            Dictionary with risk metrics
        """
        logger.info("Analyzing stock risk metrics...")
        
        # Calculate returns
        returns = RiskMetrics.calculate_returns(prices)
        
        if len(returns) < 30:
            logger.warning("Insufficient data for risk analysis")
            return {}
        
        # Basic risk metrics
        risk_metrics = {
            'volatility': float(returns.std() * np.sqrt(252)),  # Annualized
            'var_95': float(RiskMetrics.calculate_var(returns, 0.05)),
            'var_99': float(RiskMetrics.calculate_var(returns, 0.01)),
            'cvar_95': float(RiskMetrics.calculate_cvar(returns, 0.05)),
            'sharpe_ratio': float(RiskMetrics.calculate_sharpe_ratio(returns)),
            'sortino_ratio': float(RiskMetrics.calculate_sortino_ratio(returns)),
            'max_drawdown': RiskMetrics.calculate_max_drawdown(returns)
        }
        
        # Market-related metrics
        if market_prices is not None:
            market_returns = RiskMetrics.calculate_returns(market_prices)
            risk_metrics.update({
                'beta': float(RiskMetrics.calculate_beta(returns, market_returns)),
                'treynor_ratio': float(RiskMetrics.calculate_treynor_ratio(returns, market_returns))
            })
        
        # GARCH volatility modeling
        garch_model = GARCHModel()
        garch_results = garch_model.fit(returns)
        risk_metrics['garch_volatility'] = float(garch_results.get('volatility', returns.std()))
        risk_metrics['garch_success'] = bool(garch_results.get('success', False))
        
        return risk_metrics
    
    def analyze_portfolio_risk(self, prices_df: pd.DataFrame, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze portfolio risk
        
        Args:
            prices_df: DataFrame with stock prices
            weights: Portfolio weights (if None, equal weights assumed)
        
        Returns:
            Dictionary with portfolio risk metrics
        """
        logger.info("Analyzing portfolio risk...")
        
        # Calculate returns
        returns_df = prices_df.pct_change().dropna()
        
        if weights is None:
            # Equal weights
            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
        
        # Portfolio metrics
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Risk metrics
        risk_metrics = {
            'portfolio_volatility': RiskMetrics.calculate_returns(portfolio_returns).std() * np.sqrt(252),
            'portfolio_var_95': RiskMetrics.calculate_var(portfolio_returns, 0.05),
            'portfolio_sharpe': RiskMetrics.calculate_sharpe_ratio(portfolio_returns),
            'portfolio_max_drawdown': RiskMetrics.calculate_max_drawdown(portfolio_returns),
            'weights': weights.tolist()
        }
        
        # Covariance and correlation
        cov_matrix = self.portfolio_analyzer.calculate_covariance_matrix(returns_df)
        corr_matrix = self.portfolio_analyzer.calculate_correlation_matrix(returns_df)
        
        risk_metrics.update({
            'covariance_matrix': cov_matrix.tolist(),
            'correlation_matrix': corr_matrix.values.tolist(),
            'portfolio_volatility_cov': self.portfolio_analyzer.calculate_portfolio_volatility(weights, cov_matrix)
        })
        
        return risk_metrics
    
    def calculate_risk_metrics_batch(self, prices_dict: Dict[str, pd.Series]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate risk metrics for multiple stocks
        
        Args:
            prices_dict: Dictionary with stock symbols and price series
        
        Returns:
            Dictionary with risk metrics for each stock
        """
        logger.info(f"Calculating risk metrics for {len(prices_dict)} stocks...")
        
        results = {}
        
        for symbol, prices in prices_dict.items():
            try:
                risk_metrics = self.analyze_stock_risk(prices)
                results[symbol] = risk_metrics
            except Exception as e:
                logger.error(f"Error calculating risk metrics for {symbol}: {str(e)}")
                results[symbol] = {}
        
        return results
    
    def get_risk_summary(self, risk_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create risk summary DataFrame
        
        Args:
            risk_metrics: Dictionary with risk metrics
        
        Returns:
            DataFrame with risk summary
        """
        summary_data = []
        
        for symbol, metrics in risk_metrics.items():
            if metrics:
                summary_data.append({
                    'symbol': symbol,
                    'volatility': metrics.get('volatility', 0),
                    'var_95': metrics.get('var_95', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', {}).get('max_drawdown', 0),
                    'beta': metrics.get('beta', 0),
                    'garch_volatility': metrics.get('garch_volatility', 0)
                })
        
        return pd.DataFrame(summary_data)

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    # Sample stock prices
    stock_prices = pd.Series(
        100 * np.cumprod(1 + np.random.randn(252) * 0.02),
        index=dates
    )
    
    # Sample market prices
    market_prices = pd.Series(
        1000 * np.cumprod(1 + np.random.randn(252) * 0.015),
        index=dates
    )
    
    # Initialize risk assessor
    risk_assessor = RiskAssessment()
    
    # Analyze single stock risk
    print("Analyzing single stock risk...")
    stock_risk = risk_assessor.analyze_stock_risk(stock_prices, market_prices)
    print(f"Stock Risk Metrics: {stock_risk}")
    
    # Create portfolio data
    portfolio_prices = pd.DataFrame({
        'Stock1': stock_prices,
        'Stock2': stock_prices * 1.1 + np.random.randn(252) * 5,
        'Stock3': stock_prices * 0.9 + np.random.randn(252) * 3
    })
    
    # Analyze portfolio risk
    print("\nAnalyzing portfolio risk...")
    portfolio_risk = risk_assessor.analyze_portfolio_risk(portfolio_prices)
    print(f"Portfolio Risk Metrics: {portfolio_risk}")
    
    # Batch analysis
    print("\nBatch risk analysis...")
    prices_dict = {
        'STOCK1': stock_prices,
        'STOCK2': portfolio_prices['Stock2'],
        'STOCK3': portfolio_prices['Stock3']
    }
    
    batch_results = risk_assessor.calculate_risk_metrics_batch(prices_dict)
    risk_summary = risk_assessor.get_risk_summary(batch_results)
    print(f"Risk Summary:\n{risk_summary}")
    
    print("Risk assessment completed!")
