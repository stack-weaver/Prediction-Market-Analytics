"""
Advanced Charting Module for Stock Price Prediction System
Implements candlestick charts, volume profiles, technical indicators, and advanced visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.features.feature_engineering import FeatureEngineer
from src.data.dataset_manager import nse_dataset_manager

logger = logging.getLogger(__name__)

class AdvancedCharting:
    """
    Advanced charting and visualization for stock data
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff4444',
            'neutral': '#888888',
            'volume_up': '#4CAF50',
            'volume_down': '#F44336',
            'background': '#1e1e1e',
            'grid': '#333333'
        }
    
    def create_candlestick_chart(self, df: pd.DataFrame, symbol: str, 
                               days: int = 90, show_volume: bool = True,
                               show_indicators: bool = True) -> Dict[str, Any]:
        """
        Create advanced candlestick chart with volume and technical indicators
        
        Args:
            df: Stock data DataFrame
            symbol: Stock symbol
            days: Number of days to display
            show_volume: Whether to show volume subplot
            show_indicators: Whether to show technical indicators
            
        Returns:
            Chart configuration and data
        """
        try:
            logger.info(f"Creating candlestick chart for {symbol}")
            
            # Prepare data
            chart_data = df.tail(days).copy()
            if chart_data.empty:
                return {"error": "No data available for charting"}
            
            # Create subplots
            if show_volume and show_indicators:
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(f'{symbol} Price Chart', 'Volume', 'Technical Indicators'),
                    row_heights=[0.6, 0.2, 0.2]
                )
            elif show_volume:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=(f'{symbol} Price Chart', 'Volume'),
                    row_heights=[0.7, 0.3]
                )
            else:
                fig = make_subplots(
                    rows=1, cols=1,
                    subplot_titles=(f'{symbol} Price Chart',)
                )
            
            # Add candlestick chart
            candlestick = go.Candlestick(
                x=chart_data['date'],
                open=chart_data['open'],
                high=chart_data['high'],
                low=chart_data['low'],
                close=chart_data['close'],
                name='Price',
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish'],
                increasing_fillcolor=self.colors['bullish'],
                decreasing_fillcolor=self.colors['bearish']
            )
            
            fig.add_trace(candlestick, row=1, col=1)
            
            # Add moving averages
            if 'sma_20' in chart_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=chart_data['date'],
                        y=chart_data['sma_20'],
                        name='SMA 20',
                        line=dict(color='orange', width=2)
                    ),
                    row=1, col=1
                )
            
            if 'ema_12' in chart_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=chart_data['date'],
                        y=chart_data['ema_12'],
                        name='EMA 12',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
            
            # Add Bollinger Bands
            if all(col in chart_data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                fig.add_trace(
                    go.Scatter(
                        x=chart_data['date'],
                        y=chart_data['bb_upper'],
                        name='BB Upper',
                        line=dict(color='purple', width=1, dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=chart_data['date'],
                        y=chart_data['bb_lower'],
                        name='BB Lower',
                        line=dict(color='purple', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,0,128,0.1)',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Add volume chart
            if show_volume and 'volume' in chart_data.columns:
                volume_colors = []
                for i in range(len(chart_data)):
                    if i == 0:
                        volume_colors.append(self.colors['neutral'])
                    else:
                        if chart_data['close'].iloc[i] > chart_data['close'].iloc[i-1]:
                            volume_colors.append(self.colors['volume_up'])
                        else:
                            volume_colors.append(self.colors['volume_down'])
                
                volume_chart = go.Bar(
                    x=chart_data['date'],
                    y=chart_data['volume'],
                    name='Volume',
                    marker_color=volume_colors,
                    opacity=0.7
                )
                
                row_num = 2 if show_indicators else 2
                fig.add_trace(volume_chart, row=row_num, col=1)
            
            # Add technical indicators
            if show_indicators:
                indicator_row = 3 if show_volume else 2
                
                # RSI
                if 'rsi' in chart_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data['date'],
                            y=chart_data['rsi'],
                            name='RSI',
                            line=dict(color='red', width=2)
                        ),
                        row=indicator_row, col=1
                    )
                    
                    # Add RSI levels
                    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                                annotation_text="Overbought", row=indicator_row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                                annotation_text="Oversold", row=indicator_row, col=1)
                
                # MACD
                if 'macd' in chart_data.columns and 'macd_signal' in chart_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data['date'],
                            y=chart_data['macd'],
                            name='MACD',
                            line=dict(color='blue', width=2)
                        ),
                        row=indicator_row, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data['date'],
                            y=chart_data['macd_signal'],
                            name='MACD Signal',
                            line=dict(color='orange', width=2)
                        ),
                        row=indicator_row, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Advanced Candlestick Chart',
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                height=800 if show_volume and show_indicators else 600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update x-axis
            fig.update_xaxes(
                title_text="Date",
                showgrid=True,
                gridcolor=self.colors['grid']
            )
            
            # Update y-axes
            fig.update_yaxes(
                title_text="Price",
                showgrid=True,
                gridcolor=self.colors['grid'],
                row=1, col=1
            )
            
            if show_volume:
                fig.update_yaxes(
                    title_text="Volume",
                    showgrid=True,
                    gridcolor=self.colors['grid'],
                    row=2, col=1
                )
            
            if show_indicators:
                fig.update_yaxes(
                    title_text="Indicator Value",
                    showgrid=True,
                    gridcolor=self.colors['grid'],
                    row=indicator_row, col=1
                )
            
            return {
                "status": "success",
                "chart_type": "candlestick",
                "symbol": symbol,
                "data_points": len(chart_data),
                "indicators_included": show_indicators,
                "volume_included": show_volume,
                "figure": fig.to_dict(),
                "chart_config": {
                    "days": days,
                    "show_volume": show_volume,
                    "show_indicators": show_indicators
                }
            }
            
        except Exception as e:
            logger.error(f"Candlestick chart creation failed: {str(e)}")
            return {"error": f"Candlestick chart creation failed: {str(e)}"}
    
    def create_volume_profile(self, df: pd.DataFrame, symbol: str, 
                            days: int = 30, bins: int = 20) -> Dict[str, Any]:
        """
        Create volume profile chart showing price-volume distribution
        
        Args:
            df: Stock data DataFrame
            symbol: Stock symbol
            days: Number of days to analyze
            bins: Number of price bins for volume profile
            
        Returns:
            Volume profile data and chart
        """
        try:
            logger.info(f"Creating volume profile for {symbol}")
            
            # Prepare data
            profile_data = df.tail(days).copy()
            if profile_data.empty or 'volume' not in profile_data.columns:
                return {"error": "No volume data available"}
            
            # Calculate price range
            min_price = profile_data['low'].min()
            max_price = profile_data['high'].max()
            price_range = max_price - min_price
            
            # Create price bins
            bin_size = price_range / bins
            price_bins = np.arange(min_price, max_price + bin_size, bin_size)
            
            # Calculate volume for each price bin
            volume_profile = []
            for i in range(len(price_bins) - 1):
                bin_low = price_bins[i]
                bin_high = price_bins[i + 1]
                bin_volume = 0
                
                for _, row in profile_data.iterrows():
                    # Calculate overlap between price range and bin
                    overlap_low = max(bin_low, row['low'])
                    overlap_high = min(bin_high, row['high'])
                    
                    if overlap_low < overlap_high:
                        # Calculate volume proportion
                        price_overlap = overlap_high - overlap_low
                        price_range_row = row['high'] - row['low']
                        if price_range_row > 0:
                            volume_proportion = price_overlap / price_range_row
                            bin_volume += row['volume'] * volume_proportion
                
                volume_profile.append({
                    'price_level': (bin_low + bin_high) / 2,
                    'volume': bin_volume,
                    'bin_low': bin_low,
                    'bin_high': bin_high
                })
            
            # Create volume profile chart
            fig = go.Figure()
            
            # Add volume bars
            fig.add_trace(
                go.Bar(
                    x=[p['volume'] for p in volume_profile],
                    y=[p['price_level'] for p in volume_profile],
                    orientation='h',
                    name='Volume Profile',
                    marker_color='rgba(0, 255, 136, 0.7)',
                    text=[f"{p['volume']:,.0f}" for p in volume_profile],
                    textposition='inside'
                )
            )
            
            # Add current price line
            current_price = profile_data['close'].iloc[-1]
            fig.add_vline(
                x=current_price,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Current Price: {current_price:.2f}"
            )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Volume Profile ({days} days)',
                xaxis_title='Volume',
                yaxis_title='Price Level',
                template='plotly_dark',
                height=600,
                showlegend=True
            )
            
            # Find key levels
            sorted_profile = sorted(volume_profile, key=lambda x: x['volume'], reverse=True)
            key_levels = sorted_profile[:5]  # Top 5 volume levels
            
            return {
                "status": "success",
                "chart_type": "volume_profile",
                "symbol": symbol,
                "days_analyzed": days,
                "volume_profile": volume_profile,
                "key_levels": key_levels,
                "current_price": current_price,
                "figure": fig.to_dict(),
                "analysis": {
                    "total_volume": sum(p['volume'] for p in volume_profile),
                    "price_range": {"min": min_price, "max": max_price},
                    "bin_size": bin_size,
                    "bins": bins
                }
            }
            
        except Exception as e:
            logger.error(f"Volume profile creation failed: {str(e)}")
            return {"error": f"Volume profile creation failed: {str(e)}"}
    
    def create_technical_analysis_dashboard(self, df: pd.DataFrame, symbol: str,
                                          days: int = 90) -> Dict[str, Any]:
        """
        Create comprehensive technical analysis dashboard
        
        Args:
            df: Stock data DataFrame
            symbol: Stock symbol
            days: Number of days to analyze
            
        Returns:
            Technical analysis dashboard data
        """
        try:
            logger.info(f"Creating technical analysis dashboard for {symbol}")
            
            # Prepare data with technical indicators
            analysis_data = df.tail(days).copy()
            if analysis_data.empty:
                return {"error": "No data available for analysis"}
            
            # Ensure technical indicators are calculated
            if not all(col in analysis_data.columns for col in ['rsi', 'macd', 'bb_upper', 'bb_lower']):
                analysis_data = self.feature_engineer.prepare_features(analysis_data)
            
            # Create subplots for comprehensive analysis
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    f'{symbol} Price & Moving Averages',
                    'RSI & MACD',
                    'Bollinger Bands',
                    'Volume & OBV'
                ),
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # Price chart with moving averages
            fig.add_trace(
                go.Scatter(
                    x=analysis_data['date'],
                    y=analysis_data['close'],
                    name='Close Price',
                    line=dict(color='white', width=2)
                ),
                row=1, col=1
            )
            
            # Add moving averages
            if 'sma_20' in analysis_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=analysis_data['date'],
                        y=analysis_data['sma_20'],
                        name='SMA 20',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'ema_12' in analysis_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=analysis_data['date'],
                        y=analysis_data['ema_12'],
                        name='EMA 12',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )
            
            # RSI chart
            if 'rsi' in analysis_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=analysis_data['date'],
                        y=analysis_data['rsi'],
                        name='RSI',
                        line=dict(color='red', width=2)
                    ),
                    row=2, col=1
                )
                
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                            annotation_text="Overbought", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", 
                            annotation_text="Oversold", row=2, col=1)
            
            # MACD chart
            if 'macd' in analysis_data.columns and 'macd_signal' in analysis_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=analysis_data['date'],
                        y=analysis_data['macd'],
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=analysis_data['date'],
                        y=analysis_data['macd_signal'],
                        name='MACD Signal',
                        line=dict(color='orange', width=2)
                    ),
                    row=2, col=1
                )
            
            # Bollinger Bands
            if all(col in analysis_data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                fig.add_trace(
                    go.Scatter(
                        x=analysis_data['date'],
                        y=analysis_data['bb_upper'],
                        name='BB Upper',
                        line=dict(color='purple', width=1, dash='dash'),
                        showlegend=False
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=analysis_data['date'],
                        y=analysis_data['bb_lower'],
                        name='BB Lower',
                        line=dict(color='purple', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,0,128,0.1)',
                        showlegend=False
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=analysis_data['date'],
                        y=analysis_data['close'],
                        name='Price',
                        line=dict(color='white', width=1)
                    ),
                    row=3, col=1
                )
            
            # Volume chart
            if 'volume' in analysis_data.columns:
                fig.add_trace(
                    go.Bar(
                        x=analysis_data['date'],
                        y=analysis_data['volume'],
                        name='Volume',
                        marker_color='rgba(0, 255, 136, 0.7)'
                    ),
                    row=4, col=1
                )
            
            # OBV chart
            if 'obv' in analysis_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=analysis_data['date'],
                        y=analysis_data['obv'],
                        name='OBV',
                        line=dict(color='cyan', width=2),
                        yaxis='y2'
                    ),
                    row=4, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Technical Analysis Dashboard',
                template='plotly_dark',
                height=1000,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Calculate technical signals
            signals = self._calculate_technical_signals(analysis_data)
            
            return {
                "status": "success",
                "chart_type": "technical_dashboard",
                "symbol": symbol,
                "days_analyzed": days,
                "figure": fig.to_dict(),
                "technical_signals": signals,
                "current_indicators": {
                    "rsi": float(analysis_data['rsi'].iloc[-1]) if 'rsi' in analysis_data.columns else None,
                    "macd": float(analysis_data['macd'].iloc[-1]) if 'macd' in analysis_data.columns else None,
                    "bb_position": self._calculate_bb_position(analysis_data),
                    "sma_20": float(analysis_data['sma_20'].iloc[-1]) if 'sma_20' in analysis_data.columns else None,
                    "ema_12": float(analysis_data['ema_12'].iloc[-1]) if 'ema_12' in analysis_data.columns else None
                }
            }
            
        except Exception as e:
            logger.error(f"Technical analysis dashboard creation failed: {str(e)}")
            return {"error": f"Technical analysis dashboard creation failed: {str(e)}"}
    
    def _calculate_technical_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical trading signals"""
        try:
            signals = {
                "rsi_signal": "neutral",
                "macd_signal": "neutral",
                "bb_signal": "neutral",
                "trend_signal": "neutral",
                "volume_signal": "neutral",
                "overall_signal": "neutral"
            }
            
            if len(df) < 2:
                return signals
            
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # RSI Signal
            if 'rsi' in df.columns:
                rsi = current['rsi']
                if rsi > 70:
                    signals["rsi_signal"] = "bearish"
                elif rsi < 30:
                    signals["rsi_signal"] = "bullish"
            
            # MACD Signal
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd = current['macd']
                macd_signal = current['macd_signal']
                if macd > macd_signal:
                    signals["macd_signal"] = "bullish"
                else:
                    signals["macd_signal"] = "bearish"
            
            # Bollinger Bands Signal
            if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
                price = current['close']
                bb_upper = current['bb_upper']
                bb_lower = current['bb_lower']
                
                if price > bb_upper:
                    signals["bb_signal"] = "bearish"
                elif price < bb_lower:
                    signals["bb_signal"] = "bullish"
            
            # Trend Signal
            if 'sma_20' in df.columns:
                price = current['close']
                sma = current['sma_20']
                if price > sma:
                    signals["trend_signal"] = "bullish"
                else:
                    signals["trend_signal"] = "bearish"
            
            # Volume Signal
            if 'volume' in df.columns:
                current_volume = current['volume']
                avg_volume = df['volume'].tail(20).mean()
                if current_volume > avg_volume * 1.5:
                    signals["volume_signal"] = "bullish"
                elif current_volume < avg_volume * 0.5:
                    signals["volume_signal"] = "bearish"
            
            # Overall Signal
            bullish_count = sum(1 for signal in signals.values() if signal == "bullish")
            bearish_count = sum(1 for signal in signals.values() if signal == "bearish")
            
            if bullish_count > bearish_count:
                signals["overall_signal"] = "bullish"
            elif bearish_count > bullish_count:
                signals["overall_signal"] = "bearish"
            
            return signals
            
        except Exception as e:
            logger.warning(f"Failed to calculate technical signals: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_bb_position(self, df: pd.DataFrame) -> str:
        """Calculate Bollinger Bands position"""
        try:
            if not all(col in df.columns for col in ['bb_upper', 'bb_lower']):
                return "unknown"
            
            current = df.iloc[-1]
            price = current['close']
            bb_upper = current['bb_upper']
            bb_lower = current['bb_lower']
            
            bb_range = bb_upper - bb_lower
            if bb_range == 0:
                return "unknown"
            
            position = (price - bb_lower) / bb_range
            
            if position > 0.8:
                return "above_upper"
            elif position < 0.2:
                return "below_lower"
            else:
                return "middle"
                
        except Exception as e:
            logger.warning(f"Failed to calculate BB position: {str(e)}")
            return "unknown"
    
    def create_correlation_heatmap(self, symbols: List[str], days: int = 90) -> Dict[str, Any]:
        """
        Create correlation heatmap for multiple stocks
        
        Args:
            symbols: List of stock symbols
            days: Number of days to analyze
            
        Returns:
            Correlation heatmap data
        """
        try:
            logger.info(f"Creating correlation heatmap for {len(symbols)} symbols")
            
            # Load data for all symbols
            price_data = {}
            for symbol in symbols:
                try:
                    data = nse_dataset_manager.load_multi_year_data(symbol, ['2024'])
                    if not data.empty:
                        price_data[symbol] = data['close'].tail(days)
                except Exception as e:
                    logger.warning(f"Could not load data for {symbol}: {str(e)}")
                    continue
            
            if len(price_data) < 2:
                return {"error": "Insufficient data for correlation analysis"}
            
            # Create correlation matrix
            df_corr = pd.DataFrame(price_data)
            correlation_matrix = df_corr.corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Stock Price Correlation Heatmap',
                template='plotly_dark',
                height=600,
                width=800
            )
            
            # Find highest and lowest correlations
            corr_values = correlation_matrix.values
            np.fill_diagonal(corr_values, np.nan)  # Remove diagonal
            
            max_corr_idx = np.unravel_index(np.nanargmax(corr_values), corr_values.shape)
            min_corr_idx = np.unravel_index(np.nanargmin(corr_values), corr_values.shape)
            
            symbols_list = correlation_matrix.columns.tolist()
            
            return {
                "status": "success",
                "chart_type": "correlation_heatmap",
                "symbols": symbols,
                "correlation_matrix": correlation_matrix.to_dict(),
                "figure": fig.to_dict(),
                "analysis": {
                    "highest_correlation": {
                        "symbols": [symbols_list[max_corr_idx[0]], symbols_list[max_corr_idx[1]]],
                        "value": float(corr_values[max_corr_idx])
                    },
                    "lowest_correlation": {
                        "symbols": [symbols_list[min_corr_idx[0]], symbols_list[min_corr_idx[1]]],
                        "value": float(corr_values[min_corr_idx])
                    },
                    "average_correlation": float(np.nanmean(corr_values))
                }
            }
            
        except Exception as e:
            logger.error(f"Correlation heatmap creation failed: {str(e)}")
            return {"error": f"Correlation heatmap creation failed: {str(e)}"}

# Global instance
advanced_charting = AdvancedCharting()
