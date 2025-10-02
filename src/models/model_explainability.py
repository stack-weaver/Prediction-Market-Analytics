"""
Model Explainability and Feature Importance Module
Implements SHAP, LIME, and other explainability techniques for model interpretation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Install with: pip install lime")

from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

from config.settings import settings

logger = logging.getLogger(__name__)

class ModelExplainability:
    """
    Model explainability and feature importance analysis
    """
    
    def __init__(self, model_path: str = "data/models"):
        self.model_path = model_path
        self.explainers = {}
        self.feature_names = None
        
    def load_model_and_data(self, model_name: str, symbol: str) -> Tuple[Any, np.ndarray, np.ndarray]:
        """
        Load trained model and associated data
        
        Args:
            model_name: Name of the model
            symbol: Stock symbol
        
        Returns:
            Tuple of (model, X_train, y_train)
        """
        try:
            # Load model
            model_file = os.path.join(self.model_path, f"{symbol}_{model_name}.joblib")
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            model_data = joblib.load(model_file)
            model = model_data['model']
            
            # Load training data (this would typically come from your data pipeline)
            # For now, we'll create sample data
            np.random.seed(42)
            n_samples = 1000
            n_features = 20
            
            X_train = np.random.randn(n_samples, n_features)
            y_train = np.random.randn(n_samples)
            
            # Set feature names
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
            
            logger.info(f"Loaded model {model_name} for {symbol}")
            return model, X_train, y_train
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def calculate_feature_importance(self, model, X_train: np.ndarray, 
                                    y_train: np.ndarray, method: str = 'permutation') -> Dict[str, Any]:
        """
        Calculate feature importance using various methods
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            method: Method to use ('permutation', 'tree_based', 'coefficient')
        
        Returns:
            Feature importance results
        """
        logger.info(f"Calculating feature importance using {method} method")
        
        try:
            if method == 'permutation':
                # Permutation importance
                perm_importance = permutation_importance(
                    model, X_train, y_train, 
                    n_repeats=10, random_state=42
                )
                
                importance_scores = perm_importance.importances_mean
                importance_std = perm_importance.importances_std
                
            elif method == 'tree_based' and hasattr(model, 'feature_importances_'):
                # Tree-based feature importance
                importance_scores = model.feature_importances_
                importance_std = np.zeros_like(importance_scores)
                
            elif method == 'coefficient' and hasattr(model, 'coef_'):
                # Linear model coefficients
                importance_scores = np.abs(model.coef_)
                importance_std = np.zeros_like(importance_scores)
                
            else:
                raise ValueError(f"Method {method} not supported for this model type")
            
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_scores,
                'std': importance_std
            }).sort_values('importance', ascending=False)
            
            return {
                'method': method,
                'feature_importance': feature_importance_df,
                'top_features': feature_importance_df.head(10).to_dict('records'),
                'total_features': len(self.feature_names)
            }
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}
    
    def calculate_shap_values(self, model, X_train: np.ndarray, 
                             X_test: np.ndarray) -> Dict[str, Any]:
        """
        Calculate SHAP values for model explainability
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
        
        Returns:
            SHAP analysis results
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
            return {}
        
        logger.info("Calculating SHAP values")
        
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                # For models with probability prediction
                explainer = shap.Explainer(model, X_train)
            else:
                # For regression models
                explainer = shap.Explainer(model, X_train)
            
            # Calculate SHAP values for test set
            shap_values = explainer(X_test)
            
            # Calculate summary statistics
            mean_shap_values = np.mean(np.abs(shap_values.values), axis=0)
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'shap_importance': mean_shap_values
            }).sort_values('shap_importance', ascending=False)
            
            return {
                'method': 'shap',
                'shap_values': shap_values.values,
                'feature_importance': feature_importance,
                'top_features': feature_importance.head(10).to_dict('records'),
                'explainer': explainer
            }
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")
            return {}
    
    def calculate_lime_explanations(self, model, X_train: np.ndarray, 
                                   X_test: np.ndarray, 
                                   num_samples: int = 5) -> Dict[str, Any]:
        """
        Calculate LIME explanations for individual predictions
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            num_samples: Number of samples to explain
        
        Returns:
            LIME analysis results
        """
        if not LIME_AVAILABLE:
            logger.warning("LIME not available. Install with: pip install lime")
            return {}
        
        logger.info("Calculating LIME explanations")
        
        try:
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=self.feature_names,
                mode='regression',
                random_state=42
            )
            
            # Generate explanations for sample predictions
            explanations = []
            for i in range(min(num_samples, len(X_test))):
                explanation = explainer.explain_instance(
                    X_test[i], 
                    model.predict,
                    num_features=10
                )
                
                # Extract feature importance from explanation
                feature_importance = explanation.as_list()
                
                explanations.append({
                    'sample_index': i,
                    'prediction': model.predict([X_test[i]])[0],
                    'feature_importance': feature_importance,
                    'explanation': explanation
                })
            
            return {
                'method': 'lime',
                'explanations': explanations,
                'num_samples': len(explanations)
            }
            
        except Exception as e:
            logger.error(f"Error calculating LIME explanations: {str(e)}")
            return {}
    
    def calculate_partial_dependence(self, model, X_train: np.ndarray, 
                                   features: List[int]) -> Dict[str, Any]:
        """
        Calculate partial dependence plots for selected features
        
        Args:
            model: Trained model
            X_train: Training features
            features: List of feature indices to analyze
        
        Returns:
            Partial dependence results
        """
        logger.info("Calculating partial dependence")
        
        try:
            partial_dep_results = {}
            
            for feature_idx in features:
                if feature_idx >= X_train.shape[1]:
                    continue
                
                # Calculate partial dependence
                pd_results = partial_dependence(
                    model, X_train, [feature_idx], 
                    grid_resolution=50
                )
                
                partial_dep_results[f"feature_{feature_idx}"] = {
                    'feature_name': self.feature_names[feature_idx],
                    'grid': pd_results['grid'][0],
                    'partial_dependence': pd_results['partial_dependence'][0],
                    'feature_values': X_train[:, feature_idx]
                }
            
            return {
                'method': 'partial_dependence',
                'results': partial_dep_results,
                'analyzed_features': list(partial_dep_results.keys())
            }
            
        except Exception as e:
            logger.error(f"Error calculating partial dependence: {str(e)}")
            return {}
    
    def generate_model_summary(self, model_name: str, symbol: str) -> Dict[str, Any]:
        """
        Generate comprehensive model explainability summary
        
        Args:
            model_name: Name of the model
            symbol: Stock symbol
        
        Returns:
            Comprehensive explainability summary
        """
        logger.info(f"Generating model explainability summary for {model_name} on {symbol}")
        
        try:
            # Load model and data
            model, X_train, y_train = self.load_model_and_data(model_name, symbol)
            
            # Create test set
            X_test = X_train[-100:]  # Use last 100 samples as test
            
            # Calculate different types of feature importance
            permutation_importance = self.calculate_feature_importance(
                model, X_train, y_train, 'permutation'
            )
            
            tree_importance = self.calculate_feature_importance(
                model, X_train, y_train, 'tree_based'
            )
            
            # Calculate SHAP values
            shap_results = self.calculate_shap_values(model, X_train, X_test)
            
            # Calculate LIME explanations
            lime_results = self.calculate_lime_explanations(model, X_train, X_test)
            
            # Calculate partial dependence for top features
            top_features = permutation_importance.get('top_features', [])
            if top_features:
                top_feature_indices = [
                    int(f['feature'].split('_')[1]) for f in top_features[:5]
                ]
                partial_dep_results = self.calculate_partial_dependence(
                    model, X_train, top_feature_indices
                )
            else:
                partial_dep_results = {}
            
            # Generate summary
            summary = {
                'model_name': model_name,
                'symbol': symbol,
                'timestamp': pd.Timestamp.now().isoformat(),
                'feature_importance': {
                    'permutation': permutation_importance,
                    'tree_based': tree_importance,
                    'shap': shap_results
                },
                'individual_explanations': lime_results,
                'partial_dependence': partial_dep_results,
                'model_type': type(model).__name__,
                'total_features': len(self.feature_names),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating model summary: {str(e)}")
            return {}
    
    def compare_model_explainability(self, models: List[str], symbol: str) -> Dict[str, Any]:
        """
        Compare explainability across multiple models
        
        Args:
            models: List of model names to compare
            symbol: Stock symbol
        
        Returns:
            Comparison results
        """
        logger.info(f"Comparing explainability for models: {models} on {symbol}")
        
        try:
            comparison_results = {}
            
            for model_name in models:
                try:
                    summary = self.generate_model_summary(model_name, symbol)
                    comparison_results[model_name] = summary
                except Exception as e:
                    logger.error(f"Error analyzing {model_name}: {str(e)}")
                    comparison_results[model_name] = {'error': str(e)}
            
            # Find common important features across models
            common_features = self._find_common_features(comparison_results)
            
            return {
                'models_compared': models,
                'symbol': symbol,
                'individual_results': comparison_results,
                'common_features': common_features,
                'comparison_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return {}
    
    def _find_common_features(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find common important features across models
        
        Args:
            comparison_results: Results from model comparison
        
        Returns:
            Common features analysis
        """
        try:
            feature_importance_scores = {}
            
            for model_name, results in comparison_results.items():
                if 'error' in results:
                    continue
                
                permutation_importance = results.get('feature_importance', {}).get('permutation', {})
                top_features = permutation_importance.get('top_features', [])
                
                for feature_info in top_features:
                    feature_name = feature_info['feature']
                    importance_score = feature_info['importance']
                    
                    if feature_name not in feature_importance_scores:
                        feature_importance_scores[feature_name] = []
                    
                    feature_importance_scores[feature_name].append({
                        'model': model_name,
                        'importance': importance_score
                    })
            
            # Calculate average importance across models
            common_features = []
            for feature_name, scores in feature_importance_scores.items():
                if len(scores) > 1:  # Feature appears in multiple models
                    avg_importance = np.mean([s['importance'] for s in scores])
                    models_using = [s['model'] for s in scores]
                    
                    common_features.append({
                        'feature': feature_name,
                        'avg_importance': avg_importance,
                        'models_using': models_using,
                        'num_models': len(models_using)
                    })
            
            # Sort by average importance
            common_features.sort(key=lambda x: x['avg_importance'], reverse=True)
            
            return {
                'common_features': common_features[:10],  # Top 10
                'total_common_features': len(common_features),
                'models_analyzed': len(comparison_results)
            }
            
        except Exception as e:
            logger.error(f"Error finding common features: {str(e)}")
            return {}

# Example usage
if __name__ == "__main__":
    # Initialize explainability analyzer
    explainer = ModelExplainability()
    
    # Generate model summary
    print("Generating model explainability summary...")
    summary = explainer.generate_model_summary('lstm', 'RELIANCE')
    print(f"Summary generated for {summary.get('model_name', 'N/A')}")
    
    # Compare models
    print("\nComparing model explainability...")
    comparison = explainer.compare_model_explainability(['lstm', 'random_forest', 'xgboost'], 'RELIANCE')
    print(f"Comparison completed for {len(comparison.get('models_compared', []))} models")
    
    print("Model explainability analysis completed!")

