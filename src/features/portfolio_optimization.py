"""
Portfolio Optimization Module
Implements Modern Portfolio Theory, Black-Litterman model, and other optimization techniques
"""

import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize, differential_evolution
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings

logger = logging.getLogger(__name__)

class ModernPortfolioTheory:
    """
    Modern Portfolio Theory implementation
    """
    
    def __init__(self, risk_free_rate: float = 0.06):
        self.risk_free_rate = risk_free_rate
    
    def calculate_expected_returns(self, returns: pd.DataFrame, method: str = 'mean') -> np.ndarray:
        """
        Calculate expected returns
        
        Args:
            returns: Returns DataFrame
            method: 'mean', 'capm', or 'black_litterman'
        
        Returns:
            Expected returns array
        """
        if method == 'mean':
            return returns.mean().values * 252  # Annualized
        
        elif method == 'capm':
            # CAPM-based expected returns
            market_return = returns.mean().mean() * 252
            betas = self._calculate_betas(returns)
            expected_returns = self.risk_free_rate + betas * (market_return - self.risk_free_rate)
            return expected_returns
        
        else:
            raise ValueError("Method must be 'mean' or 'capm'")
    
    def _calculate_betas(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate beta coefficients"""
        market_returns = returns.mean(axis=1)  # Equal-weighted market
        betas = []
        
        for col in returns.columns:
            stock_returns = returns[col]
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance
            betas.append(beta)
        
        return np.array(betas)
    
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
            return returns.cov().values * 252  # Annualized
        elif method == 'ledoit_wolf':
            lw = LedoitWolf()
            return lw.fit(returns).covariance_ * 252
        else:
            raise ValueError("Method must be 'empirical' or 'ledoit_wolf'")
    
    def optimize_portfolio(self, expected_returns: np.ndarray, 
                          cov_matrix: np.ndarray, 
                          target_return: Optional[float] = None,
                          risk_tolerance: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize portfolio using quadratic programming
        
        Args:
            expected_returns: Expected returns array
            cov_matrix: Covariance matrix
            target_return: Target return (for efficient frontier)
            risk_tolerance: Risk tolerance parameter
        
        Returns:
            Optimization results
        """
        n_assets = len(expected_returns)
        
        # Portfolio weights (decision variables)
        weights = cp.Variable(n_assets)
        
        # Expected portfolio return
        portfolio_return = expected_returns @ weights
        
        # Portfolio variance
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0  # Long-only portfolio
        ]
        
        if target_return is not None:
            # Efficient frontier: minimize variance for given return
            constraints.append(portfolio_return >= target_return)
            objective = cp.Minimize(portfolio_variance)
        elif risk_tolerance is not None:
            # Risk-return optimization: maximize return - risk_tolerance * variance
            objective = cp.Maximize(portfolio_return - risk_tolerance * portfolio_variance)
        else:
            # Maximum Sharpe ratio
            excess_return = portfolio_return - self.risk_free_rate
            objective = cp.Maximize(excess_return / cp.sqrt(portfolio_variance))
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            logger.warning(f"Optimization failed with status: {problem.status}")
            return {}
        
        optimal_weights = weights.value
        
        # Calculate portfolio metrics
        portfolio_return_val = portfolio_return.value
        portfolio_variance_val = portfolio_variance.value
        portfolio_volatility = np.sqrt(portfolio_variance_val)
        sharpe_ratio = (portfolio_return_val - self.risk_free_rate) / portfolio_volatility
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return_val,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'status': problem.status
        }
    
    def efficient_frontier(self, expected_returns: np.ndarray, 
                          cov_matrix: np.ndarray, 
                          num_portfolios: int = 100) -> pd.DataFrame:
        """
        Calculate efficient frontier
        
        Args:
            expected_returns: Expected returns array
            cov_matrix: Covariance matrix
            num_portfolios: Number of portfolios to generate
        
        Returns:
            DataFrame with efficient frontier data
        """
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            result = self.optimize_portfolio(expected_returns, cov_matrix, target_return)
            if result:
                efficient_portfolios.append({
                    'expected_return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'weights': result['weights']
                })
        
        return pd.DataFrame(efficient_portfolios)

class BlackLittermanModel:
    """
    Black-Litterman model implementation
    """
    
    def __init__(self, risk_free_rate: float = 0.06, tau: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.tau = tau  # Confidence parameter
    
    def calculate_implied_returns(self, market_caps: np.ndarray, 
                                 cov_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate implied equilibrium returns
        
        Args:
            market_caps: Market capitalizations
            cov_matrix: Covariance matrix
        
        Returns:
            Implied returns
        """
        # Market portfolio weights
        market_weights = market_caps / market_caps.sum()
        
        # Risk aversion parameter (lambda)
        market_return = 0.08  # Assumed market return
        market_variance = market_weights.T @ cov_matrix @ market_weights
        risk_aversion = (market_return - self.risk_free_rate) / market_variance
        
        # Implied returns
        implied_returns = risk_aversion * cov_matrix @ market_weights
        
        return implied_returns
    
    def black_litterman_returns(self, implied_returns: np.ndarray,
                               cov_matrix: np.ndarray,
                               views: np.ndarray,
                               pick_matrix: np.ndarray,
                               uncertainty: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Black-Litterman expected returns
        
        Args:
            implied_returns: Implied equilibrium returns
            cov_matrix: Covariance matrix
            views: View returns
            pick_matrix: Pick matrix (which assets are in views)
            uncertainty: Uncertainty matrix
        
        Returns:
            Tuple of (expected returns, posterior covariance)
        """
        # Prior covariance
        prior_cov = self.tau * cov_matrix
        
        # Posterior covariance
        temp = np.linalg.inv(uncertainty) + pick_matrix.T @ np.linalg.inv(prior_cov) @ pick_matrix
        posterior_cov = np.linalg.inv(temp)
        
        # Posterior expected returns
        temp2 = np.linalg.inv(prior_cov) @ implied_returns + pick_matrix.T @ np.linalg.inv(uncertainty) @ views
        posterior_returns = posterior_cov @ temp2
        
        return posterior_returns, posterior_cov

class PortfolioOptimizer:
    """
    Main portfolio optimization class
    """
    
    def __init__(self):
        self.mpt = ModernPortfolioTheory()
        self.bl_model = BlackLittermanModel()
    
    def optimize_maximum_sharpe(self, returns: pd.DataFrame, 
                               method: str = 'empirical') -> Dict[str, Any]:
        """
        Optimize portfolio for maximum Sharpe ratio
        
        Args:
            returns: Returns DataFrame
            method: Covariance estimation method
        
        Returns:
            Optimization results
        """
        logger.info("Optimizing portfolio for maximum Sharpe ratio...")
        
        # Calculate expected returns and covariance
        expected_returns = self.mpt.calculate_expected_returns(returns)
        cov_matrix = self.mpt.calculate_covariance_matrix(returns, method)
        
        # Optimize
        result = self.mpt.optimize_portfolio(expected_returns, cov_matrix)
        
        if result:
            result['method'] = 'maximum_sharpe'
            result['assets'] = returns.columns.tolist()
        
        return result
    
    def optimize_minimum_variance(self, returns: pd.DataFrame,
                                 method: str = 'empirical') -> Dict[str, Any]:
        """
        Optimize portfolio for minimum variance
        
        Args:
            returns: Returns DataFrame
            method: Covariance estimation method
        
        Returns:
            Optimization results
        """
        logger.info("Optimizing portfolio for minimum variance...")
        
        # Calculate expected returns and covariance
        expected_returns = self.mpt.calculate_expected_returns(returns)
        cov_matrix = self.mpt.calculate_covariance_matrix(returns, method)
        
        # Find minimum variance portfolio
        n_assets = len(expected_returns)
        weights = cp.Variable(n_assets)
        
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            logger.warning(f"Minimum variance optimization failed: {problem.status}")
            return {}
        
        optimal_weights = weights.value
        portfolio_return = expected_returns @ optimal_weights
        portfolio_variance_val = portfolio_variance.value
        portfolio_volatility = np.sqrt(portfolio_variance_val)
        sharpe_ratio = (portfolio_return - self.mpt.risk_free_rate) / portfolio_volatility
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'method': 'minimum_variance',
            'assets': returns.columns.tolist()
        }
    
    def optimize_equal_risk_contribution(self, returns: pd.DataFrame,
                                        method: str = 'empirical') -> Dict[str, Any]:
        """
        Optimize portfolio for equal risk contribution
        
        Args:
            returns: Returns DataFrame
            method: Covariance estimation method
        
        Returns:
            Optimization results
        """
        logger.info("Optimizing portfolio for equal risk contribution...")
        
        cov_matrix = self.mpt.calculate_covariance_matrix(returns, method)
        n_assets = len(returns.columns)
        
        def risk_contrib(weights):
            weights = np.array(weights)
            portfolio_variance = weights.T @ cov_matrix @ weights
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_variance
            return risk_contrib
        
        def objective(weights):
            risk_contrib_val = risk_contrib(weights)
            target = np.ones(n_assets) / n_assets
            return np.sum((risk_contrib_val - target) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result.success:
            logger.warning(f"Equal risk contribution optimization failed: {result.message}")
            return {}
        
        optimal_weights = result.x
        expected_returns = self.mpt.calculate_expected_returns(returns)
        portfolio_return = expected_returns @ optimal_weights
        portfolio_variance = optimal_weights.T @ cov_matrix @ optimal_weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.mpt.risk_free_rate) / portfolio_volatility
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'method': 'equal_risk_contribution',
            'assets': returns.columns.tolist()
        }
    
    def optimize_with_constraints(self, returns: pd.DataFrame,
                                max_weight: float = 0.4,
                                sector_constraints: Optional[Dict[str, float]] = None,
                                method: str = 'empirical') -> Dict[str, Any]:
        """
        Optimize portfolio with additional constraints
        
        Args:
            returns: Returns DataFrame
            max_weight: Maximum weight per asset
            sector_constraints: Sector weight constraints
            method: Covariance estimation method
        
        Returns:
            Optimization results
        """
        logger.info("Optimizing portfolio with constraints...")
        
        expected_returns = self.mpt.calculate_expected_returns(returns)
        cov_matrix = self.mpt.calculate_covariance_matrix(returns, method)
        n_assets = len(expected_returns)
        
        weights = cp.Variable(n_assets)
        portfolio_return = expected_returns @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        # Basic constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            weights <= max_weight
        ]
        
        # Add sector constraints if provided
        if sector_constraints:
            for sector, max_sector_weight in sector_constraints.items():
                # This is a simplified example - in practice, you'd need sector mapping
                pass
        
        # Maximize Sharpe ratio
        excess_return = portfolio_return - self.mpt.risk_free_rate
        objective = cp.Maximize(excess_return / cp.sqrt(portfolio_variance))
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            logger.warning(f"Constrained optimization failed: {problem.status}")
            return {}
        
        optimal_weights = weights.value
        portfolio_return_val = portfolio_return.value
        portfolio_variance_val = portfolio_variance.value
        portfolio_volatility = np.sqrt(portfolio_variance_val)
        sharpe_ratio = (portfolio_return_val - self.mpt.risk_free_rate) / portfolio_volatility
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return_val,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'method': 'constrained_optimization',
            'assets': returns.columns.tolist(),
            'max_weight': max_weight
        }
    
    def calculate_efficient_frontier(self, returns: pd.DataFrame,
                                   num_portfolios: int = 100,
                                   method: str = 'empirical') -> pd.DataFrame:
        """
        Calculate efficient frontier
        
        Args:
            returns: Returns DataFrame
            num_portfolios: Number of portfolios
            method: Covariance estimation method
        
        Returns:
            Efficient frontier DataFrame
        """
        logger.info("Calculating efficient frontier...")
        
        expected_returns = self.mpt.calculate_expected_returns(returns)
        cov_matrix = self.mpt.calculate_covariance_matrix(returns, method)
        
        efficient_frontier = self.mpt.efficient_frontier(expected_returns, cov_matrix, num_portfolios)
        
        return efficient_frontier
    
    def compare_optimization_methods(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compare different optimization methods
        
        Args:
            returns: Returns DataFrame
        
        Returns:
            Comparison DataFrame
        """
        logger.info("Comparing optimization methods...")
        
        methods = [
            ('maximum_sharpe', self.optimize_maximum_sharpe),
            ('minimum_variance', self.optimize_minimum_variance),
            ('equal_risk_contribution', self.optimize_equal_risk_contribution)
        ]
        
        results = []
        
        for method_name, method_func in methods:
            try:
                result = method_func(returns)
                if result:
                    results.append({
                        'method': method_name,
                        'expected_return': result['expected_return'],
                        'volatility': result['volatility'],
                        'sharpe_ratio': result['sharpe_ratio']
                    })
            except Exception as e:
                logger.error(f"Error in {method_name}: {str(e)}")
        
        return pd.DataFrame(results)
    
    def rebalance_portfolio(self, current_weights: np.ndarray,
                          target_weights: np.ndarray,
                          transaction_cost: float = 0.001) -> Dict[str, Any]:
        """
        Calculate rebalancing trades
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            transaction_cost: Transaction cost per trade
        
        Returns:
            Rebalancing information
        """
        weight_changes = target_weights - current_weights
        
        # Calculate transaction costs
        total_cost = np.sum(np.abs(weight_changes)) * transaction_cost
        
        # Identify trades
        trades = []
        for i, change in enumerate(weight_changes):
            if abs(change) > 0.001:  # Minimum trade threshold
                trades.append({
                    'asset_index': i,
                    'weight_change': change,
                    'trade_type': 'buy' if change > 0 else 'sell'
                })
        
        return {
            'weight_changes': weight_changes.tolist(),
            'total_transaction_cost': total_cost,
            'number_of_trades': len(trades),
            'trades': trades
        }
    
    def optimize_multi_objective(self, returns: pd.DataFrame,
                                objectives: List[str] = ['return', 'risk', 'diversification'],
                                weights: List[float] = [0.4, 0.4, 0.2],
                                method: str = 'empirical') -> Dict[str, Any]:
        """
        Multi-objective portfolio optimization
        
        Args:
            returns: Returns DataFrame
            objectives: List of objectives ['return', 'risk', 'diversification', 'momentum']
            weights: Weights for each objective
            method: Covariance estimation method
        
        Returns:
            Multi-objective optimization results
        """
        logger.info("Performing multi-objective optimization...")
        
        expected_returns = self.mpt.calculate_expected_returns(returns)
        cov_matrix = self.mpt.calculate_covariance_matrix(returns, method)
        n_assets = len(expected_returns)
        
        def multi_objective_function(w):
            w = np.array(w)
            
            # Normalize weights
            w = w / np.sum(w)
            
            total_score = 0
            
            for i, objective in enumerate(objectives):
                if objective == 'return':
                    # Maximize expected return
                    score = -(expected_returns @ w)  # Negative for minimization
                elif objective == 'risk':
                    # Minimize portfolio variance
                    score = w.T @ cov_matrix @ w
                elif objective == 'diversification':
                    # Maximize diversification (minimize concentration)
                    score = -np.sum(w * np.log(w + 1e-10))  # Negative entropy
                elif objective == 'momentum':
                    # Maximize momentum (recent performance)
                    recent_returns = returns.tail(20).mean().values
                    score = -(recent_returns @ w)
                else:
                    score = 0
                
                total_score += weights[i] * score
            
            return total_score
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize using differential evolution for global optimization
        result = differential_evolution(
            multi_objective_function,
            bounds,
            constraints=constraints,
            seed=42,
            maxiter=1000
        )
        
        if not result.success:
            logger.warning(f"Multi-objective optimization failed: {result.message}")
            return {}
        
        optimal_weights = result.x / np.sum(result.x)  # Normalize
        portfolio_return = expected_returns @ optimal_weights
        portfolio_variance = optimal_weights.T @ cov_matrix @ optimal_weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.mpt.risk_free_rate) / portfolio_volatility
        
        # Calculate diversification metrics
        diversification_ratio = np.sum(optimal_weights * np.log(optimal_weights + 1e-10))
        concentration_ratio = np.sum(optimal_weights ** 2)
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': diversification_ratio,
            'concentration_ratio': concentration_ratio,
            'method': 'multi_objective',
            'objectives': objectives,
            'objective_weights': weights,
            'assets': returns.columns.tolist()
        }
    
    def dynamic_rebalancing(self, returns: pd.DataFrame,
                           rebalance_frequency: int = 30,
                           lookback_window: int = 252,
                           method: str = 'empirical') -> Dict[str, Any]:
        """
        Dynamic portfolio rebalancing strategy
        
        Args:
            returns: Returns DataFrame
            rebalance_frequency: Days between rebalancing
            lookback_window: Days to look back for optimization
            method: Covariance estimation method
        
        Returns:
            Dynamic rebalancing results
        """
        logger.info("Implementing dynamic rebalancing strategy...")
        
        portfolio_values = []
        weights_history = []
        rebalance_dates = []
        
        # Initial portfolio (equal weight)
        n_assets = len(returns.columns)
        current_weights = np.ones(n_assets) / n_assets
        portfolio_value = 1.0
        
        for i in range(lookback_window, len(returns)):
            # Check if it's time to rebalance
            if (i - lookback_window) % rebalance_frequency == 0:
                # Use recent data for optimization
                recent_returns = returns.iloc[i-lookback_window:i]
                
                # Optimize portfolio
                result = self.optimize_maximum_sharpe(recent_returns, method)
                
                if result:
                    current_weights = result['weights']
                    rebalance_dates.append(returns.index[i])
            
            # Calculate portfolio return for this period
            period_return = returns.iloc[i] @ current_weights
            portfolio_value *= (1 + period_return)
            
            portfolio_values.append(portfolio_value)
            weights_history.append(current_weights.copy())
        
        # Calculate performance metrics
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        total_return = portfolio_values[-1] - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (portfolio_returns.mean() * 252 - self.mpt.risk_free_rate) / volatility
        
        # Calculate turnover
        turnover = []
        for i in range(1, len(weights_history)):
            turnover.append(np.sum(np.abs(weights_history[i] - weights_history[i-1])))
        avg_turnover = np.mean(turnover) if turnover else 0
        
        return {
            'portfolio_values': portfolio_values,
            'weights_history': weights_history,
            'rebalance_dates': rebalance_dates,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'avg_turnover': avg_turnover,
            'rebalance_frequency': rebalance_frequency,
            'lookback_window': lookback_window
        }
    
    def risk_parity_optimization(self, returns: pd.DataFrame,
                                method: str = 'empirical') -> Dict[str, Any]:
        """
        Risk parity portfolio optimization
        
        Args:
            returns: Returns DataFrame
            method: Covariance estimation method
        
        Returns:
            Risk parity optimization results
        """
        logger.info("Optimizing risk parity portfolio...")
        
        cov_matrix = self.mpt.calculate_covariance_matrix(returns, method)
        n_assets = len(returns.columns)
        
        def risk_parity_objective(weights):
            weights = np.array(weights)
            portfolio_variance = weights.T @ cov_matrix @ weights
            
            # Risk contributions
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_variance
            
            # Target risk contribution (equal for all assets)
            target_contrib = np.ones(n_assets) / n_assets
            
            # Minimize sum of squared differences
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(risk_parity_objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if not result.success:
            logger.warning(f"Risk parity optimization failed: {result.message}")
            return {}
        
        optimal_weights = result.x
        expected_returns = self.mpt.calculate_expected_returns(returns)
        portfolio_return = expected_returns @ optimal_weights
        portfolio_variance = optimal_weights.T @ cov_matrix @ optimal_weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.mpt.risk_free_rate) / portfolio_volatility
        
        # Calculate actual risk contributions
        marginal_contrib = cov_matrix @ optimal_weights
        risk_contrib = optimal_weights * marginal_contrib / portfolio_variance
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'risk_contributions': risk_contrib,
            'method': 'risk_parity',
            'assets': returns.columns.tolist()
        }
    
    def factor_based_optimization(self, returns: pd.DataFrame,
                                 factors: Optional[pd.DataFrame] = None,
                                 method: str = 'empirical') -> Dict[str, Any]:
        """
        Factor-based portfolio optimization
        
        Args:
            returns: Returns DataFrame
            factors: Factor returns DataFrame
            method: Covariance estimation method
        
        Returns:
            Factor-based optimization results
        """
        logger.info("Performing factor-based optimization...")
        
        if factors is None:
            # Create simple factors from returns
            factors = pd.DataFrame({
                'market': returns.mean(axis=1),
                'momentum': returns.rolling(20).mean().mean(axis=1),
                'volatility': returns.rolling(20).std().mean(axis=1)
            }).dropna()
        
        # Align returns and factors
        common_index = returns.index.intersection(factors.index)
        returns_aligned = returns.loc[common_index]
        factors_aligned = factors.loc[common_index]
        
        # Calculate factor loadings
        factor_loadings = {}
        for asset in returns_aligned.columns:
            asset_returns = returns_aligned[asset]
            loadings = []
            for factor in factors_aligned.columns:
                correlation = np.corrcoef(asset_returns, factors_aligned[factor])[0, 1]
                loadings.append(correlation)
            factor_loadings[asset] = loadings
        
        # Convert to DataFrame
        factor_loadings_df = pd.DataFrame(factor_loadings).T
        factor_loadings_df.columns = factors_aligned.columns
        
        # Optimize based on factor exposure
        expected_returns = self.mpt.calculate_expected_returns(returns_aligned)
        cov_matrix = self.mpt.calculate_covariance_matrix(returns_aligned, method)
        n_assets = len(expected_returns)
        
        def factor_objective(weights):
            weights = np.array(weights)
            
            # Portfolio factor exposure
            portfolio_exposure = factor_loadings_df.values.T @ weights
            
            # Target factor exposure (balanced)
            target_exposure = np.ones(len(factors_aligned.columns)) / len(factors_aligned.columns)
            
            # Minimize deviation from target exposure
            exposure_penalty = np.sum((portfolio_exposure - target_exposure) ** 2)
            
            # Add return maximization
            return_penalty = -(expected_returns @ weights)
            
            return exposure_penalty + 0.1 * return_penalty
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(factor_objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if not result.success:
            logger.warning(f"Factor-based optimization failed: {result.message}")
            return {}
        
        optimal_weights = result.x
        portfolio_return = expected_returns @ optimal_weights
        portfolio_variance = optimal_weights.T @ cov_matrix @ optimal_weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.mpt.risk_free_rate) / portfolio_volatility
        
        # Calculate factor exposures
        factor_exposures = factor_loadings_df.values.T @ optimal_weights
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'factor_exposures': factor_exposures,
            'factor_loadings': factor_loadings_df,
            'method': 'factor_based',
            'assets': returns_aligned.columns.tolist()
        }

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    # Sample returns for 5 stocks
    returns_data = {}
    for i in range(5):
        returns_data[f'Stock_{i+1}'] = np.random.randn(252) * 0.02
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    
    # Optimize for maximum Sharpe ratio
    print("Optimizing for maximum Sharpe ratio...")
    max_sharpe_result = optimizer.optimize_maximum_sharpe(returns_df)
    print(f"Maximum Sharpe Result: {max_sharpe_result}")
    
    # Optimize for minimum variance
    print("\nOptimizing for minimum variance...")
    min_var_result = optimizer.optimize_minimum_variance(returns_df)
    print(f"Minimum Variance Result: {min_var_result}")
    
    # Calculate efficient frontier
    print("\nCalculating efficient frontier...")
    efficient_frontier = optimizer.calculate_efficient_frontier(returns_df)
    print(f"Efficient Frontier Shape: {efficient_frontier.shape}")
    
    # Compare methods
    print("\nComparing optimization methods...")
    comparison = optimizer.compare_optimization_methods(returns_df)
    print(f"Method Comparison:\n{comparison}")
    
    print("Portfolio optimization completed!")
