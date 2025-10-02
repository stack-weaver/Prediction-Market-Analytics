import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  CircularProgress,
  FormControlLabel,
  Switch,
  Chip,
  Grid,
} from '@mui/material';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  AreaChart,
  BarChart,
} from 'recharts';
import { TrendingUp, TrendingDown, ShowChart } from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

interface CandlestickData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sma20?: number;
  sma50?: number;
  upperBB?: number;
  lowerBB?: number;
}

interface ChartData {
  dates: string[];
  prices: number[];
  volumes: number[];
  high: number[];
  low: number[];
  open: number[];
}

interface PredictionData {
  ensemble_prediction: number[];
  confidence_bands?: {
    upper: number[];
    lower: number[];
  };
  confidence_score: number;
  trend: string;
}

interface EnhancedCandlestickChartProps {
  symbol: string;
}

const EnhancedCandlestickChart: React.FC<EnhancedCandlestickChartProps> = ({ symbol }) => {
  const [chartData, setChartData] = useState<CandlestickData[]>([]);
  const [predictionData, setPredictionData] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [showVolume, setShowVolume] = useState(true);
  const [showMA, setShowMA] = useState(true);
  const [showBB, setShowBB] = useState(false);
  const [showPrediction, setShowPrediction] = useState(true);

  useEffect(() => {
    if (symbol) {
      fetchData();
    }
  }, [symbol]);

  const fetchData = async () => {
    setLoading(true);
    setError('');
    try {
      // Fetch chart data
      console.log(`Fetching chart data for ${symbol}`);
      const chartResponse = await axios.get(`/api/chart/${symbol}?days=60`, {
        timeout: 15000,
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (chartResponse.status !== 200) {
        throw new Error(`HTTP ${chartResponse.status}: ${chartResponse.statusText}`);
      }
      
      const rawData: ChartData = chartResponse.data;
      console.log('Chart data received:', rawData);
      
      if (!rawData || !rawData.dates || rawData.dates.length === 0) {
        throw new Error('No chart data available for this symbol');
      }
      
      // Fetch prediction data
      const predictionResponse = await axios.post(`/api/predict`, {
        symbol: symbol,
        days_ahead: 5
      });
      console.log('Prediction data received:', predictionResponse.data);
      setPredictionData(predictionResponse.data);

      // Process chart data
      const processedData: CandlestickData[] = rawData.dates.map((date, index) => {
        const close = rawData.prices[index];
        const volume = rawData.volumes[index];
        const high = rawData.high[index];
        const low = rawData.low[index];
        const open = rawData.open[index];

        // Calculate simple moving averages
        let sma20: number | undefined, sma50: number | undefined;
        if (index >= 19) {
          sma20 = rawData.prices.slice(index - 19, index + 1).reduce((a, b) => a + b) / 20;
        }
        if (index >= 49) {
          sma50 = rawData.prices.slice(index - 49, index + 1).reduce((a, b) => a + b) / 50;
        }

        // Calculate Bollinger Bands (simplified)
        let upperBB, lowerBB;
        if (sma20) {
          const prices = rawData.prices.slice(Math.max(0, index - 19), index + 1);
          const variance = prices.reduce((acc, price) => acc + Math.pow(price - sma20!, 2), 0) / prices.length;
          const stdDev = Math.sqrt(variance);
          upperBB = sma20! + (2 * stdDev);
          lowerBB = sma20! - (2 * stdDev);
        }

        return {
          date,
          open,
          high,
          low,
          close,
          volume,
          sma20,
          sma50,
          upperBB,
          lowerBB,
        };
      });

      // Add prediction data to the end
      if (predictionResponse.data.ensemble_prediction) {
        const lastDate = new Date(rawData.dates[rawData.dates.length - 1]);
        const predictionData = predictionResponse.data;
        
        predictionData.ensemble_prediction.forEach((pred: number, index: number) => {
          const futureDate = new Date(lastDate);
          futureDate.setDate(futureDate.getDate() + index + 1);
          
          // Safely access confidence bands with fallbacks
          const upperBand = predictionData.confidence_bands?.upper?.[index] || pred * 1.05;
          const lowerBand = predictionData.confidence_bands?.lower?.[index] || pred * 0.95;
          
          processedData.push({
            date: futureDate.toISOString().split('T')[0],
            open: pred,
            high: upperBand,
            low: lowerBand,
            close: pred,
            volume: 0, // No volume for predictions
            isPrediction: true,
          } as any);
        });
      }

      setChartData(processedData);
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to fetch chart data';
      setError(errorMessage);
      console.error('Chart fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const CustomCandlestick = (props: any) => {
    const { payload, x, y, width, height } = props;
    if (!payload) return null;

    const { open, high, low, close, isPrediction } = payload;
    const isGreen = close >= open;
    const color = isPrediction ? '#9c27b0' : (isGreen ? '#4caf50' : '#f44336');
    const opacity = isPrediction ? 0.7 : 1;

    const bodyHeight = Math.abs(close - open);
    const bodyY = Math.min(close, open);
    
    return (
      <g opacity={opacity}>
        {/* High-Low line */}
        <line
          x1={x + width / 2}
          y1={y + (high - Math.max(close, open)) * height / (high - low)}
          x2={x + width / 2}
          y2={y + (high - Math.min(close, open)) * height / (high - low)}
          stroke={color}
          strokeWidth={1}
        />
        
        {/* Body */}
        <rect
          x={x + width * 0.2}
          y={y + (high - Math.max(close, open)) * height / (high - low)}
          width={width * 0.6}
          height={bodyHeight * height / (high - low)}
          fill={isGreen ? color : 'transparent'}
          stroke={color}
          strokeWidth={1}
        />
      </g>
    );
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const isPrediction = data.isPrediction;
      
      return (
        <Box sx={{ 
          bgcolor: 'background.paper', 
          p: 2, 
          border: 1, 
          borderColor: 'divider',
          borderRadius: 1 
        }}>
          <Typography variant="body2" fontWeight="bold">
            {isPrediction ? '🔮 Prediction' : '📊 Historical'} - {label}
          </Typography>
          <Typography variant="body2">Open: ₹{data.open?.toFixed(2)}</Typography>
          <Typography variant="body2">High: ₹{data.high?.toFixed(2)}</Typography>
          <Typography variant="body2">Low: ₹{data.low?.toFixed(2)}</Typography>
          <Typography variant="body2">Close: ₹{data.close?.toFixed(2)}</Typography>
          {!isPrediction && <Typography variant="body2">Volume: {data.volume?.toLocaleString()}</Typography>}
          {data.sma20 && <Typography variant="body2">SMA20: ₹{data.sma20.toFixed(2)}</Typography>}
        </Box>
      );
    }
    return null;
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'BULLISH': return '#4caf50';
      case 'BEARISH': return '#f44336';
      default: return '#ff9800';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'BULLISH': return <TrendingUp />;
      case 'BEARISH': return <TrendingDown />;
      default: return <ShowChart />;
    }
  };

  return (
    <Box sx={{ width: '100%', minHeight: 400 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2} flexWrap="wrap">
        {predictionData && (
          <Chip
            icon={getTrendIcon(predictionData.trend)}
            label={`${predictionData.trend} (${(predictionData.confidence_score * 100).toFixed(1)}%)`}
            sx={{ color: 'white', bgcolor: getTrendColor(predictionData.trend) }}
          />
        )}
      </Box>

        {/* Controls */}
        <Box display="flex" gap={2} mb={2} flexWrap="wrap">
          <FormControlLabel
            control={<Switch checked={showVolume} onChange={(e) => setShowVolume(e.target.checked)} />}
            label="Volume"
          />
          <FormControlLabel
            control={<Switch checked={showMA} onChange={(e) => setShowMA(e.target.checked)} />}
            label="Moving Averages"
          />
          <FormControlLabel
            control={<Switch checked={showBB} onChange={(e) => setShowBB(e.target.checked)} />}
            label="Bollinger Bands"
          />
          <FormControlLabel
            control={<Switch checked={showPrediction} onChange={(e) => setShowPrediction(e.target.checked)} />}
            label="Predictions"
          />
        </Box>

        {loading && (
          <Box display="flex" justifyContent="center" p={4}>
            <CircularProgress />
          </Box>
        )}

        {error && (
          <Typography color="error" align="center">
            {error}
          </Typography>
        )}

        {chartData.length > 0 && !loading && (
          <Grid container spacing={2}>
            {/* Main Price Chart */}
            <Grid item xs={12}>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => new Date(value).toLocaleDateString()}
                  />
                  <YAxis 
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => `₹${value}`}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  
                  {/* Bollinger Bands */}
                  {showBB && (
                    <>
                      <Area
                        type="monotone"
                        dataKey="upperBB"
                        stroke="#e0e0e0"
                        fill="transparent"
                        strokeDasharray="2 2"
                      />
                      <Area
                        type="monotone"
                        dataKey="lowerBB"
                        stroke="#e0e0e0"
                        fill="transparent"
                        strokeDasharray="2 2"
                      />
                    </>
                  )}
                  
                  {/* Moving Averages */}
                  {showMA && (
                    <>
                      <Line
                        type="monotone"
                        dataKey="sma20"
                        stroke="#ff9800"
                        strokeWidth={2}
                        dot={false}
                        name="SMA 20"
                      />
                      <Line
                        type="monotone"
                        dataKey="sma50"
                        stroke="#2196f3"
                        strokeWidth={2}
                        dot={false}
                        name="SMA 50"
                      />
                    </>
                  )}
                  
                  {/* Price Line for simplicity (can be replaced with custom candlesticks) */}
                  <Line
                    type="monotone"
                    dataKey="close"
                    stroke="#1976d2"
                    strokeWidth={2}
                    dot={false}
                    name="Close Price"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </Grid>

            {/* Volume Chart */}
            {showVolume && (
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Volume Profile
                </Typography>
                <ResponsiveContainer width="100%" height={150}>
                  <BarChart data={chartData.filter(d => !(d as any).isPrediction)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="date" 
                      tick={{ fontSize: 10 }}
                      tickFormatter={(value) => new Date(value).toLocaleDateString()}
                    />
                    <YAxis 
                      tick={{ fontSize: 10 }}
                      tickFormatter={(value) => `${(value/1000000).toFixed(1)}M`}
                    />
                    <Tooltip 
                      formatter={(value: number) => [value.toLocaleString(), 'Volume']}
                    />
                    <Bar dataKey="volume" fill="#607d8b" />
                  </BarChart>
                </ResponsiveContainer>
              </Grid>
            )}
          </Grid>
        )}
    </Box>
  );
};

export default EnhancedCandlestickChart;
