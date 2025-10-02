import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
} from '@mui/material';
import { TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material';
import { PredictionResponse } from '../types';

interface PredictionCardProps {
  prediction: PredictionResponse | null;
  loading: boolean;
}

const PredictionCard: React.FC<PredictionCardProps> = ({ prediction, loading }) => {
  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'BULLISH':
        return <TrendingUp color="success" />;
      case 'BEARISH':
        return <TrendingDown color="error" />;
      default:
        return <TrendingFlat color="warning" />;
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'BULLISH':
        return 'success';
      case 'BEARISH':
        return 'error';
      default:
        return 'warning';
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Predictions
          </Typography>
          <LinearProgress />
        </CardContent>
      </Card>
    );
  }

  if (!prediction) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Predictions
          </Typography>
          <Typography color="text.secondary">
            Select a stock to see predictions
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const priceChange = ((prediction.ensemble_prediction[0] - prediction.current_price) / prediction.current_price) * 100;

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Predictions for {prediction.symbol}
          </Typography>
          <Chip
            icon={getTrendIcon(prediction.trend)}
            label={prediction.trend}
            color={getTrendColor(prediction.trend) as any}
            variant="outlined"
          />
        </Box>

        <Box sx={{ display: 'flex', gap: 2 }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Current Price
            </Typography>
            <Typography variant="h5">
              ₹{prediction.current_price.toFixed(2)}
            </Typography>
          </Box>
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Predicted Price (1 Day)
            </Typography>
            <Typography variant="h5" color={priceChange >= 0 ? 'success.main' : 'error.main'}>
              ₹{prediction.ensemble_prediction[0].toFixed(2)}
            </Typography>
            <Typography variant="body2" color={priceChange >= 0 ? 'success.main' : 'error.main'}>
              {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
            </Typography>
          </Box>
        </Box>

        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Confidence Score
          </Typography>
          <LinearProgress
            variant="determinate"
            value={prediction.confidence_score * 100}
            sx={{ height: 8, borderRadius: 4 }}
          />
          <Typography variant="caption" color="text.secondary">
            {(prediction.confidence_score * 100).toFixed(1)}%
          </Typography>
        </Box>

        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            5-Day Forecast
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {prediction.ensemble_prediction.map((price, index) => (
              <Chip
                key={index}
                label={`Day ${index + 1}: ₹${price.toFixed(2)}`}
                variant="outlined"
                size="small"
              />
            ))}
          </Box>
        </Box>

        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Individual Model Results ({Object.keys(prediction.predictions).length} Models)
          </Typography>
          {Object.entries(prediction.predictions).map(([model, prices]) => {
            const modelNames = {
              'lstm': '🧠 LSTM Neural Network',
              'random_forest': '🌳 Random Forest',
              'xgboost': '⚡ XGBoost',
              'arima': '📈 ARIMA',
              'prophet': '🔮 Prophet',
              'transformer': '🎯 Transformer'
            };
            
            return (
              <Box key={model} sx={{ 
                mb: 2, 
                p: 2, 
                border: '1px solid', 
                borderColor: 'grey.200',
                borderRadius: 2,
                bgcolor: 'grey.50'
              }}>
                <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                  {modelNames[model as keyof typeof modelNames] || model.toUpperCase()}
                </Typography>
                <Typography variant="h6" color="primary.main" gutterBottom>
                  ₹{prices[0].toFixed(2)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  5-day forecast: {prices.map((price, idx) => `₹${price.toFixed(2)}`).join(' → ')}
                </Typography>
              </Box>
            );
          })}
        </Box>
      </CardContent>
    </Card>
  );
};

export default PredictionCard;
