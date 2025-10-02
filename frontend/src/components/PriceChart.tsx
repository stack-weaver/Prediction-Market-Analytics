import React from 'react';
import {
  Box,
  Typography,
  Chip,
  Grid,
  Paper,
  LinearProgress,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material';
import { ChartData, PredictionResponse } from '../types';

interface PriceChartProps {
  chartData: ChartData | null;
  prediction: PredictionResponse | null;
  symbol: string;
  loading: boolean;
}

const PriceChart: React.FC<PriceChartProps> = ({
  chartData,
  prediction,
  symbol,
  loading,
}) => {
  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Price Chart - {symbol}
        </Typography>
        <LinearProgress />
      </Box>
    );
  }

  if (!chartData) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>
          Price Chart
        </Typography>
        <Typography color="textSecondary">
          Select a stock to see historical price chart
        </Typography>
      </Box>
    );
  }

  // Prepare data for chart
  const priceData = chartData.dates.map((date, index) => ({
    date,
    price: chartData.prices[index],
    volume: chartData.volumes[index],
  }));

  // Get current price and calculate change
  const currentPrice = chartData.prices[chartData.prices.length - 1];
  const previousPrice = chartData.prices[chartData.prices.length - 2];
  const priceChange = currentPrice - previousPrice;
  const changePercent = (priceChange / previousPrice) * 100;

  const getTrendIcon = () => {
    if (changePercent > 0.5) return <TrendingUp color="success" />;
    if (changePercent < -0.5) return <TrendingDown color="error" />;
    return <TrendingFlat color="action" />;
  };

  const getTrendColor = () => {
    if (changePercent > 0.5) return 'success.main';
    if (changePercent < -0.5) return 'error.main';
    return 'text.secondary';
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Paper sx={{ p: 1, border: '1px solid #ccc' }}>
          <Typography variant="body2">{`Date: ${new Date(label).toLocaleDateString()}`}</Typography>
          <Typography variant="body2" color="primary">
            {`Price: ₹${payload[0].value.toFixed(2)}`}
          </Typography>
        </Paper>
      );
    }
    return null;
  };

  return (
    <Box>
      {/* Price Header */}
      <Box sx={{ mb: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="subtitle1" fontWeight="bold">
            {symbol} Current Price
          </Typography>
          <Box display="flex" alignItems="center" gap={1}>
            {getTrendIcon()}
            <Typography variant="h5" color={getTrendColor()} fontWeight="bold">
              ₹{currentPrice.toFixed(2)}
            </Typography>
          </Box>
        </Box>
        <Box display="flex" alignItems="center" gap={2}>
          <Chip
            label={`${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%`}
            color={changePercent >= 0 ? 'success' : 'error'}
            size="small"
          />
          <Typography variant="body2" color="textSecondary">
            24h Change: ₹{priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}
          </Typography>
        </Box>
      </Box>

      {/* Key Metrics */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom fontWeight="bold">
          Key Statistics
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6} sm={3}>
            <Box sx={{ p: 1.5, border: '1px solid', borderColor: 'grey.300', borderRadius: 1, textAlign: 'center' }}>
              <Typography variant="caption" color="textSecondary">24h High</Typography>
              <Typography variant="body1" fontWeight="bold">
                ₹{Math.max(...chartData.high).toFixed(2)}
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box sx={{ p: 1.5, border: '1px solid', borderColor: 'grey.300', borderRadius: 1, textAlign: 'center' }}>
              <Typography variant="caption" color="textSecondary">24h Low</Typography>
              <Typography variant="body1" fontWeight="bold">
                ₹{Math.min(...chartData.low).toFixed(2)}
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box sx={{ p: 1.5, border: '1px solid', borderColor: 'grey.300', borderRadius: 1, textAlign: 'center' }}>
              <Typography variant="caption" color="textSecondary">Volume</Typography>
              <Typography variant="body1" fontWeight="bold">
                {(chartData.volumes[chartData.volumes.length - 1] / 1000).toFixed(0)}K
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box sx={{ p: 1.5, border: '1px solid', borderColor: 'grey.300', borderRadius: 1, textAlign: 'center' }}>
              <Typography variant="caption" color="textSecondary">Range</Typography>
              <Typography variant="body1" fontWeight="bold">
                ₹{(Math.max(...chartData.high) - Math.min(...chartData.low)).toFixed(2)}
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Box>

      {/* Price Chart */}
      <Box sx={{ height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={priceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="date" 
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <YAxis 
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => `₹${value}`}
              domain={['dataMin - 10', 'dataMax + 10']}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line 
              type="monotone" 
              dataKey="price" 
              stroke="#1976d2" 
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
            {/* Current price reference line */}
            <ReferenceLine 
              y={currentPrice} 
              stroke="#ff9800" 
              strokeDasharray="5 5"
              label={{ value: `Current: ₹${currentPrice.toFixed(2)}`, position: 'right' }}
            />
            {/* Prediction line if available */}
            {prediction && (
              <ReferenceLine 
                y={prediction.ensemble_prediction[0]} 
                stroke="#4caf50" 
                strokeDasharray="3 3"
                label={{ value: `Predicted: ₹${prediction.ensemble_prediction[0].toFixed(2)}`, position: 'right' }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </Box>

      {/* Prediction Summary */}
      {prediction && (
        <Box sx={{ mt: 3, p: 2, bgcolor: 'primary.light', borderRadius: 1, border: '1px solid', borderColor: 'primary.main' }}>
          <Typography variant="subtitle2" gutterBottom fontWeight="bold" color="primary.contrastText">
            🔮 AI Prediction Summary
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, mt: 2 }}>
            <Box sx={{ minWidth: 120 }}>
              <Typography variant="caption" color="primary.contrastText" sx={{ opacity: 0.8 }}>
                Next Day Target
              </Typography>
              <Typography variant="h6" fontWeight="bold" color="primary.contrastText">
                ₹{prediction.ensemble_prediction[0].toFixed(2)}
              </Typography>
            </Box>
            <Box sx={{ minWidth: 100 }}>
              <Typography variant="caption" color="primary.contrastText" sx={{ opacity: 0.8 }}>
                Confidence
              </Typography>
              <Typography variant="h6" fontWeight="bold" color="primary.contrastText">
                {(prediction.confidence_score * 100).toFixed(1)}%
              </Typography>
            </Box>
            <Box sx={{ minWidth: 80 }}>
              <Typography variant="caption" color="primary.contrastText" sx={{ opacity: 0.8 }}>
                Trend
              </Typography>
              <Typography variant="h6" fontWeight="bold" color="primary.contrastText">
                {prediction.trend}
              </Typography>
            </Box>
            <Box sx={{ minWidth: 80 }}>
              <Typography variant="caption" color="primary.contrastText" sx={{ opacity: 0.8 }}>
                Models
              </Typography>
              <Typography variant="h6" fontWeight="bold" color="primary.contrastText">
                {Object.keys(prediction.predictions).length}
              </Typography>
            </Box>
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default PriceChart;
