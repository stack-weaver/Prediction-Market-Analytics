import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
} from '@mui/material';
import { AnalysisResponse } from '../types';

interface AnalysisCardProps {
  analysis: AnalysisResponse | null;
  loading: boolean;
}

const AnalysisCard: React.FC<AnalysisCardProps> = ({ analysis, loading }) => {
  if (loading) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Technical Analysis
          </Typography>
          <LinearProgress />
        </CardContent>
      </Card>
    );
  }

  if (!analysis) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Technical Analysis
          </Typography>
          <Typography color="text.secondary">
            Select a stock to see analysis
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const getRSIColor = (rsi: number) => {
    if (rsi > 70) return 'error';
    if (rsi < 30) return 'success';
    return 'default';
  };

  const getRSILabel = (rsi: number) => {
    if (rsi > 70) return 'Overbought';
    if (rsi < 30) return 'Oversold';
    return 'Neutral';
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Technical Analysis - {analysis.symbol}
        </Typography>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {/* Technical Indicators */}
          <Box>
            <Typography variant="subtitle1" gutterBottom>
              Technical Indicators
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <Box sx={{ flex: '1 1 150px', minWidth: '150px' }}>
                <Box sx={{ p: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    RSI (14)
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="body2">
                      {analysis.technical_indicators.rsi.toFixed(1)}
                    </Typography>
                    <Chip
                      label={getRSILabel(analysis.technical_indicators.rsi)}
                      color={getRSIColor(analysis.technical_indicators.rsi) as any}
                      size="small"
                    />
                  </Box>
                </Box>
              </Box>
              <Box sx={{ flex: '1 1 150px', minWidth: '150px' }}>
                <Box sx={{ p: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    MACD
                  </Typography>
                  <Typography variant="body2">
                    {analysis.technical_indicators.macd.toFixed(3)}
                  </Typography>
                </Box>
              </Box>
              <Box sx={{ flex: '1 1 150px', minWidth: '150px' }}>
                <Box sx={{ p: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    SMA (20)
                  </Typography>
                  <Typography variant="body2">
                    ₹{analysis.technical_indicators.sma_20.toFixed(2)}
                  </Typography>
                </Box>
              </Box>
              <Box sx={{ flex: '1 1 150px', minWidth: '150px' }}>
                <Box sx={{ p: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    EMA (12)
                  </Typography>
                  <Typography variant="body2">
                    ₹{analysis.technical_indicators.ema_12.toFixed(2)}
                  </Typography>
                </Box>
              </Box>
            </Box>
          </Box>

          {/* Risk Metrics */}
          <Box>
            <Typography variant="subtitle1" gutterBottom>
              Risk Metrics
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <Box sx={{ flex: '1 1 150px', minWidth: '150px' }}>
                <Box sx={{ p: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    Volatility (Annual)
                  </Typography>
                  <Typography variant="body2">
                    {(analysis.risk_metrics.volatility * 100).toFixed(1)}%
                  </Typography>
                </Box>
              </Box>
              <Box sx={{ flex: '1 1 150px', minWidth: '150px' }}>
                <Box sx={{ p: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    Sharpe Ratio
                  </Typography>
                  <Typography variant="body2">
                    {analysis.risk_metrics.sharpe_ratio.toFixed(2)}
                  </Typography>
                </Box>
              </Box>
              <Box sx={{ flex: '1 1 150px', minWidth: '150px' }}>
                <Box sx={{ p: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    Max Drawdown
                  </Typography>
                  <Typography variant="body2" color="error.main">
                    {(analysis.risk_metrics.max_drawdown * 100).toFixed(1)}%
                  </Typography>
                </Box>
              </Box>
              <Box sx={{ flex: '1 1 150px', minWidth: '150px' }}>
                <Box sx={{ p: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    VaR (95%)
                  </Typography>
                  <Typography variant="body2" color="error.main">
                    {(analysis.risk_metrics.var_95 * 100).toFixed(1)}%
                  </Typography>
                </Box>
              </Box>
            </Box>
          </Box>

          {/* Model Performance */}
          <Box>
            <Typography variant="subtitle1" gutterBottom>
              Model Performance
            </Typography>
            {Object.entries(analysis.model_performance).map(([model, metrics]) => {
              const modelNames = {
                'lstm': '🧠 LSTM Neural Network',
                'random_forest': '🌳 Random Forest',
                'xgboost': '⚡ XGBoost',
                'arima': '📈 ARIMA',
                'prophet': '🔮 Prophet',
                'transformer': '🎯 Transformer'
              };
              
              return (
                <Box key={model} sx={{ mb: 1, p: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary" fontWeight="bold">
                    {modelNames[model as keyof typeof modelNames] || model.toUpperCase()}
                  </Typography>
                <Box sx={{ display: 'flex', gap: 2, mt: 0.5 }}>
                  <Typography variant="caption">
                    RMSE: {metrics.rmse.toFixed(3)}
                  </Typography>
                  <Typography variant="caption">
                    MAPE: {metrics.mape.toFixed(1)}%
                  </Typography>
                  <Typography variant="caption">
                    R²: {metrics.r2.toFixed(3)}
                  </Typography>
                </Box>
              </Box>
              );
            })}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default AnalysisCard;
