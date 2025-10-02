"""
Advanced Portfolio Strategies Module
Implements sophisticated portfolio management strategies including factor investing, 
momentum strategies, mean reversion, and dynamic allocation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from src.features.portfolio_optimization import PortfolioOptimizer
from src.data.dataset_manager import nse_dataset_manager
from src.features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class AdvancedPortfolioStrategies:
    """
    Advanced portfolio management strategies
    """
    
    def __init__(self):
        self.portfolio_optimizer = PortfolioOptimizer()
        self.feature_engineer = FeatureEngineer()
        self.strategies = {}
        
    def momentum_strategy(self, symbols: List[str], lookback_days: int = 20,
                        holding_period: int = 5, rebalance_frequency: int = 5,
                        top_n: int = 5) -> Dict[str, Any]:
        """
        Implement momentum-based portfolio strategy
        
        Args:
            symbols: List of stock symbols
            lookback_days: Days to look back for momentum calculation
            holding_period: Days to hold positions
            rebalance_frequency: Days between rebalancing
            top_n: Number of top momentum stocks to select
            
        Returns:
            Momentum strategy results
        """
        try:
            logger.info(f"Implementing momentum strategy for {len(symbols)} symbols")
            
            # Load data for all symbols
            returns_data = {}
            price_data = {}
            
            for symbol in symbols:
                try:
                    data = nse_dataset_manager.load_multi_year_data(symbol, ['2024'])
                    if not data.empty:
                        returns = data['close'].pct_change().dropna()
                        returns_data[symbol] = returns
                        price_data[symbol] = data['close']
                except Exception as e:
                    logger.warning(f"Could not load data for {symbol}: {str(e)}")
                    continue
            
            if len(returns_data) < top_n:
                return {"error": f"Insufficient data. Need at least {top_n} symbols"}
            
            # Align all returns data
            returns_df = pd.DataFrame(returns_data).dropna()
            
            # Calculate momentum scores
            momentum_scores = {}
            for symbol in returns_df.columns:
                returns = returns_df[symbol]
                
                # Calculate various momentum metrics
                short_momentum = returns.tail(lookback_days//2).mean()  # Short-term momentum
                long_momentum = returns.tail(lookback_days).mean()     # Long-term momentum
                momentum_consistency = 1 - returns.tail(lookback_days).std()  # Consistency
                
                # Combined momentum score
                momentum_score = (
                    short_momentum * 0.4 +
                    long_momentum * 0.4 +
                    momentum_consistency * 0.2
                )
                
                momentum_scores[symbol] = momentum_score
            
            # Select top momentum stocks
            sorted_momentum = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
            selected_stocks = [stock for stock, score in sorted_momentum[:top_n]]
            
            # Calculate portfolio weights (equal weight for simplicity)
            weights = {stock: 1.0/top_n for stock in selected_stocks}
            
            # Calculate strategy performance
            portfolio_returns = returns_df[selected_stocks].mean(axis=1)
            
            # Calculate performance metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            return {
                "status": "success",
                "strategy_type": "momentum",
                "selected_stocks": selected_stocks,
                "momentum_scores": {stock: momentum_scores[stock] for stock in selected_stocks},
                "weights": weights,
                "performance": {
                    "total_return": total_return,
                    "annualized_return": annualized_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown
                },
                "parameters": {
                    "lookback_days": lookback_days,
                    "holding_period": holding_period,
                    "rebalance_frequency": rebalance_frequency,
                    "top_n": top_n
                }
            }
            
        except Exception as e:
            logger.error(f"Momentum strategy failed: {str(e)}")
            return {"error": f"Momentum strategy failed: {str(e)}"}
    
    def mean_reversion_strategy(self, symbols: List[str], lookback_days: int = 20,
                              z_score_threshold: float = 2.0, 
                              rebalance_frequency: int = 5) -> Dict[str, Any]:
        """
        Implement mean reversion portfolio strategy
        
        Args:
            symbols: List of stock symbols
            lookback_days: Days to look back for mean calculation
            z_score_threshold: Z-score threshold for entry/exit
            rebalance_frequency: Days between rebalancing
            
        Returns:
            Mean reversion strategy results
        """
        try:
            logger.info(f"Implementing mean reversion strategy for {len(symbols)} symbols")
            
            # Load data for all symbols
            returns_data = {}
            price_data = {}
            
            for symbol in symbols:
                try:
                    data = nse_dataset_manager.load_multi_year_data(symbol, ['2024'])
                    if not data.empty:
                        returns = data['close'].pct_change().dropna()
                        returns_data[symbol] = returns
                        price_data[symbol] = data['close']
                except Exception as e:
                    logger.warning(f"Could not load data for {symbol}: {str(e)}")
                    continue
            
            if len(returns_data) < 2:
                return {"error": "Insufficient data for mean reversion strategy"}
            
            # Align all returns data
            returns_df = pd.DataFrame(returns_data).dropna()
            
            # Calculate mean reversion signals
            mean_reversion_signals = {}
            z_scores = {}
            
            for symbol in returns_df.columns:
                returns = returns_df[symbol]
                
                # Calculate rolling mean and std
                rolling_mean = returns.rolling(window=lookback_days).mean()
                rolling_std = returns.rolling(window=lookback_days).std()
                
                # Calculate Z-scores
                z_score = (returns - rolling_mean) / rolling_std
                z_scores[symbol] = z_score
                
                # Generate signals
                # Buy when Z-score < -threshold (oversold)
                # Sell when Z-score > threshold (overbought)
                signals = pd.Series(0, index=returns.index)
                signals[z_score < -z_score_threshold] = 1  # Buy signal
                signals[z_score > z_score_threshold] = -1  # Sell signal
                
                mean_reversion_signals[symbol] = signals
            
            # Calculate strategy performance
            strategy_returns = []
            for i in range(len(returns_df)):
                daily_return = 0
                active_positions = 0
                
                for symbol in returns_df.columns:
                    if i < len(mean_reversion_signals[symbol]):
                        signal = mean_reversion_signals[symbol].iloc[i]
                        if signal == 1:  # Buy signal
                            daily_return += returns_df[symbol].iloc[i]
                            active_positions += 1
                        elif signal == -1:  # Sell signal
                            daily_return -= returns_df[symbol].iloc[i]
                            active_positions += 1
                
                if active_positions > 0:
                    daily_return /= active_positions
                
                strategy_returns.append(daily_return)
            
            strategy_returns = pd.Series(strategy_returns)
            
            # Calculate performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calculate signal statistics
            signal_stats = {}
            for symbol in returns_df.columns:
                signals = mean_reversion_signals[symbol]
                signal_stats[symbol] = {
                    "buy_signals": (signals == 1).sum(),
                    "sell_signals": (signals == -1).sum(),
                    "avg_z_score": z_scores[symbol].mean(),
                    "z_score_volatility": z_scores[symbol].std()
                }
            
            return {
                "status": "success",
                "strategy_type": "mean_reversion",
                "performance": {
                    "total_return": total_return,
                    "annualized_return": annualized_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown
                },
                "signal_statistics": signal_stats,
                "parameters": {
                    "lookback_days": lookback_days,
                    "z_score_threshold": z_score_threshold,
                    "rebalance_frequency": rebalance_frequency
                }
            }
            
        except Exception as e:
            logger.error(f"Mean reversion strategy failed: {str(e)}")
            return {"error": f"Mean reversion strategy failed: {str(e)}"}
    
    def factor_investing_strategy(self, symbols: List[str], factors: List[str] = None,
                                 factor_weights: List[float] = None) -> Dict[str, Any]:
        """
        Implement factor investing strategy
        
        Args:
            symbols: List of stock symbols
            factors: List of factors to consider
            factor_weights: Weights for each factor
            
        Returns:
            Factor investing strategy results
        """
        try:
            logger.info(f"Implementing factor investing strategy for {len(symbols)} symbols")
            
            if factors is None:
                factors = ['value', 'momentum', 'quality', 'size']
            
            if factor_weights is None:
                factor_weights = [0.25, 0.25, 0.25, 0.25]
            
            # Load data for all symbols
            stock_data = {}
            for symbol in symbols:
                try:
                    data = nse_dataset_manager.load_multi_year_data(symbol, ['2024'])
                    if not data.empty:
                        stock_data[symbol] = data
                except Exception as e:
                    logger.warning(f"Could not load data for {symbol}: {str(e)}")
                    continue
            
            if len(stock_data) < 2:
                return {"error": "Insufficient data for factor investing strategy"}
            
            # Calculate factor scores for each stock
            factor_scores = {}
            
            for symbol, data in stock_data.items():
                scores = {}
                
                # Value factor (P/E, P/B ratios - lower is better)
                if 'close' in data.columns and len(data) > 0:
                    current_price = data['close'].iloc[-1]
                    
                    # Simple value metrics (in real implementation, you'd use actual financial data)
                    pe_ratio = np.random.uniform(10, 30)  # Placeholder
                    pb_ratio = np.random.uniform(1, 5)    # Placeholder
                    
                    value_score = 1 / (pe_ratio * pb_ratio)  # Lower ratios = higher score
                    scores['value'] = value_score
                
                # Momentum factor (recent price performance)
                if len(data) > 20:
                    returns = data['close'].pct_change().dropna()
                    momentum_score = returns.tail(20).mean()
                    scores['momentum'] = momentum_score
                
                # Quality factor (volatility, consistency)
                if len(data) > 10:
                    returns = data['close'].pct_change().dropna()
                    quality_score = 1 - returns.std()  # Lower volatility = higher quality
                    scores['quality'] = quality_score
                
                # Size factor (market cap - smaller is better for small-cap premium)
                size_score = np.random.uniform(0.1, 1.0)  # Placeholder
                scores['size'] = size_score
                
                factor_scores[symbol] = scores
            
            # Calculate composite factor scores
            composite_scores = {}
            for symbol, scores in factor_scores.items():
                composite_score = 0
                for i, factor in enumerate(factors):
                    if factor in scores:
                        composite_score += scores[factor] * factor_weights[i]
                composite_scores[symbol] = composite_score
            
            # Rank stocks by composite score
            ranked_stocks = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top stocks (top 30% for long positions)
            top_count = max(1, len(ranked_stocks) // 3)
            selected_stocks = [stock for stock, score in ranked_stocks[:top_count]]
            
            # Calculate portfolio weights (equal weight)
            weights = {stock: 1.0/len(selected_stocks) for stock in selected_stocks}
            
            # Calculate strategy performance
            returns_data = {}
            for symbol in selected_stocks:
                if symbol in stock_data:
                    returns = stock_data[symbol]['close'].pct_change().dropna()
                    returns_data[symbol] = returns
            
            if returns_data:
                returns_df = pd.DataFrame(returns_data).dropna()
                portfolio_returns = returns_df.mean(axis=1)
                
                # Calculate performance metrics
                total_return = (1 + portfolio_returns).prod() - 1
                annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
                volatility = portfolio_returns.std() * np.sqrt(252)
                sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                
                # Calculate drawdown
                cumulative_returns = (1 + portfolio_returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
            else:
                total_return = annualized_return = volatility = sharpe_ratio = max_drawdown = 0
            
            return {
                "status": "success",
                "strategy_type": "factor_investing",
                "factors_used": factors,
                "factor_weights": factor_weights,
                "selected_stocks": selected_stocks,
                "factor_scores": {stock: factor_scores[stock] for stock in selected_stocks},
                "composite_scores": {stock: composite_scores[stock] for stock in selected_stocks},
                "weights": weights,
                "performance": {
                    "total_return": total_return,
                    "annualized_return": annualized_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown
                }
            }
            
        except Exception as e:
            logger.error(f"Factor investing strategy failed: {str(e)}")
            return {"error": f"Factor investing strategy failed: {str(e)}"}
    
    def dynamic_allocation_strategy(self, symbols: List[str], 
                                  market_regime_detection: bool = True,
                                  volatility_target: float = 0.15) -> Dict[str, Any]:
        """
        Implement dynamic allocation strategy based on market conditions
        
        Args:
            symbols: List of stock symbols
            market_regime_detection: Whether to detect market regimes
            volatility_target: Target portfolio volatility
            
        Returns:
            Dynamic allocation strategy results
        """
        try:
            logger.info(f"Implementing dynamic allocation strategy for {len(symbols)} symbols")
            
            # Load data for all symbols
            returns_data = {}
            for symbol in symbols:
                try:
                    data = nse_dataset_manager.load_multi_year_data(symbol, ['2024'])
                    if not data.empty:
                        returns = data['close'].pct_change().dropna()
                        returns_data[symbol] = returns
                except Exception as e:
                    logger.warning(f"Could not load data for {symbol}: {str(e)}")
                    continue
            
            if len(returns_data) < 2:
                return {"error": "Insufficient data for dynamic allocation strategy"}
            
            # Align all returns data
            returns_df = pd.DataFrame(returns_data).dropna()
            
            # Detect market regimes
            market_regimes = []
            if market_regime_detection:
                market_regimes = self._detect_market_regimes(returns_df)
            
            # Calculate dynamic weights based on market conditions
            dynamic_weights = {}
            regime_analysis = {}
            
            for i, date in enumerate(returns_df.index):
                if i < 20:  # Need minimum data for analysis
                    continue
                
                # Get recent returns
                recent_returns = returns_df.iloc[max(0, i-20):i+1]
                
                # Calculate market regime
                if market_regime_detection and i < len(market_regimes):
                    current_regime = market_regimes[i]
                else:
                    current_regime = self._classify_current_regime(recent_returns)
                
                # Adjust weights based on regime
                weights = self._calculate_regime_weights(recent_returns, current_regime, volatility_target)
                dynamic_weights[date] = weights
                
                regime_analysis[date] = {
                    "regime": current_regime,
                    "volatility": recent_returns.std().mean(),
                    "correlation": recent_returns.corr().mean().mean()
                }
            
            # Calculate strategy performance
            strategy_returns = []
            for i, date in enumerate(returns_df.index):
                if date in dynamic_weights:
                    weights = dynamic_weights[date]
                    daily_return = 0
                    
                    for symbol in returns_df.columns:
                        if symbol in weights:
                            daily_return += returns_df[symbol].iloc[i] * weights[symbol]
                    
                    strategy_returns.append(daily_return)
                else:
                    strategy_returns.append(0)
            
            strategy_returns = pd.Series(strategy_returns)
            
            # Calculate performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            return {
                "status": "success",
                "strategy_type": "dynamic_allocation",
                "market_regime_detection": market_regime_detection,
                "volatility_target": volatility_target,
                "regime_analysis": regime_analysis,
                "dynamic_weights": dynamic_weights,
                "performance": {
                    "total_return": total_return,
                    "annualized_return": annualized_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown
                }
            }
            
        except Exception as e:
            logger.error(f"Dynamic allocation strategy failed: {str(e)}")
            return {"error": f"Dynamic allocation strategy failed: {str(e)}"}
    
    def _detect_market_regimes(self, returns_df: pd.DataFrame) -> List[str]:
        """Detect market regimes using clustering"""
        try:
            # Calculate regime indicators
            volatility = returns_df.std().mean()
            correlation = returns_df.corr().mean().mean()
            trend = returns_df.mean().mean()
            
            # Simple regime classification
            regimes = []
            for i in range(len(returns_df)):
                recent_data = returns_df.iloc[max(0, i-10):i+1]
                recent_vol = recent_data.std().mean()
                recent_corr = recent_data.corr().mean().mean()
                recent_trend = recent_data.mean().mean()
                
                if recent_vol > volatility * 1.2:
                    regime = "high_volatility"
                elif recent_corr > 0.7:
                    regime = "high_correlation"
                elif recent_trend > 0.01:
                    regime = "bull_market"
                elif recent_trend < -0.01:
                    regime = "bear_market"
                else:
                    regime = "normal"
                
                regimes.append(regime)
            
            return regimes
            
        except Exception as e:
            logger.warning(f"Market regime detection failed: {str(e)}")
            return ["normal"] * len(returns_df)
    
    def _classify_current_regime(self, recent_returns: pd.DataFrame) -> str:
        """Classify current market regime"""
        try:
            volatility = recent_returns.std().mean()
            correlation = recent_returns.corr().mean().mean()
            trend = recent_returns.mean().mean()
            
            if volatility > 0.02:
                return "high_volatility"
            elif correlation > 0.7:
                return "high_correlation"
            elif trend > 0.01:
                return "bull_market"
            elif trend < -0.01:
                return "bear_market"
            else:
                return "normal"
                
        except Exception as e:
            logger.warning(f"Regime classification failed: {str(e)}")
            return "normal"
    
    def _calculate_regime_weights(self, recent_returns: pd.DataFrame, 
                                 regime: str, volatility_target: float) -> Dict[str, float]:
        """Calculate weights based on market regime"""
        try:
            symbols = recent_returns.columns.tolist()
            
            if regime == "high_volatility":
                # Reduce position sizes in high volatility
                base_weight = 0.5 / len(symbols)
                weights = {symbol: base_weight for symbol in symbols}
            elif regime == "high_correlation":
                # Equal weight when correlations are high
                weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
            elif regime == "bull_market":
                # Favor momentum stocks
                momentum_scores = recent_returns.mean()
                momentum_scores = momentum_scores / momentum_scores.sum()
                weights = momentum_scores.to_dict()
            elif regime == "bear_market":
                # Favor defensive stocks (lower volatility)
                volatility_scores = 1 / recent_returns.std()
                volatility_scores = volatility_scores / volatility_scores.sum()
                weights = volatility_scores.to_dict()
            else:
                # Normal regime - equal weight
                weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
            
            return weights
            
        except Exception as e:
            logger.warning(f"Weight calculation failed: {str(e)}")
            symbols = recent_returns.columns.tolist()
            return {symbol: 1.0 / len(symbols) for symbol in symbols}
    
    def compare_strategies(self, symbols: List[str], 
                          strategies: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple portfolio strategies
        
        Args:
            symbols: List of stock symbols
            strategies: List of strategies to compare
            
        Returns:
            Strategy comparison results
        """
        try:
            logger.info(f"Comparing strategies for {len(symbols)} symbols")
            
            if strategies is None:
                strategies = ['momentum', 'mean_reversion', 'factor_investing', 'dynamic_allocation']
            
            results = {}
            
            for strategy in strategies:
                try:
                    if strategy == 'momentum':
                        result = self.momentum_strategy(symbols)
                    elif strategy == 'mean_reversion':
                        result = self.mean_reversion_strategy(symbols)
                    elif strategy == 'factor_investing':
                        result = self.factor_investing_strategy(symbols)
                    elif strategy == 'dynamic_allocation':
                        result = self.dynamic_allocation_strategy(symbols)
                    else:
                        continue
                    
                    if "error" not in result:
                        results[strategy] = result
                        
                except Exception as e:
                    logger.warning(f"Strategy {strategy} failed: {str(e)}")
                    continue
            
            # Create comparison summary
            comparison_summary = {}
            for strategy, result in results.items():
                if "performance" in result:
                    comparison_summary[strategy] = {
                        "annualized_return": result["performance"]["annualized_return"],
                        "volatility": result["performance"]["volatility"],
                        "sharpe_ratio": result["performance"]["sharpe_ratio"],
                        "max_drawdown": result["performance"]["max_drawdown"]
                    }
            
            return {
                "status": "success",
                "comparison_type": "strategy_comparison",
                "symbols": symbols,
                "strategies_compared": list(results.keys()),
                "results": results,
                "comparison_summary": comparison_summary
            }
            
        except Exception as e:
            logger.error(f"Strategy comparison failed: {str(e)}")
            return {"error": f"Strategy comparison failed: {str(e)}"}

# Global instance
advanced_portfolio_strategies = AdvancedPortfolioStrategies()
