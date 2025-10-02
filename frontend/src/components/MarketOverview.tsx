import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Chip,
  LinearProgress,
  Card,
  CardContent,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { TrendingUp, TrendingDown, Assessment } from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8002';

interface MarketData {
  symbol: string;
  last_price: number;
  change?: number;
  changePercent?: number;
  volume?: number;
}

interface MarketOverviewProps {
  selectedStock?: string;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

const MarketOverview: React.FC<MarketOverviewProps> = ({ selectedStock }) => {
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    fetchMarketData();
  }, []);

  const fetchMarketData = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.get(`${API_BASE_URL}/api/stocks`);
      const stocks = response.data.slice(0, 8); // Top 8 stocks for overview
      
      // Add mock change data for demonstration
      const enrichedData = stocks.map((stock: any) => ({
        ...stock,
        change: (Math.random() - 0.5) * 20, // Random change for demo
        changePercent: (Math.random() - 0.5) * 5,
        volume: Math.floor(Math.random() * 1000000),
      }));
      
      setMarketData(enrichedData);
    } catch (err) {
      setError('Failed to fetch market data');
      console.error('Market data fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          📊 Market Overview
        </Typography>
        <LinearProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          📊 Market Overview
        </Typography>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  // Calculate market summary
  const gainers = marketData.filter(stock => (stock.changePercent || 0) > 0);
  const losers = marketData.filter(stock => (stock.changePercent || 0) < 0);
  const unchanged = marketData.filter(stock => Math.abs(stock.changePercent || 0) <= 0.1);

  const marketSummary = [
    { name: 'Gainers', value: gainers.length, color: '#4caf50' },
    { name: 'Losers', value: losers.length, color: '#f44336' },
    { name: 'Unchanged', value: unchanged.length, color: '#ff9800' },
  ];

  // Prepare volume data for bar chart
  const volumeData = marketData.slice(0, 6).map(stock => ({
    symbol: stock.symbol,
    volume: stock.volume || 0,
    price: stock.last_price,
  }));

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) return `${(volume / 1000000).toFixed(1)}M`;
    if (volume >= 1000) return `${(volume / 1000).toFixed(1)}K`;
    return volume.toString();
  };

  return (
    <Box>
      {/* Market Summary Stats */}
      <Box sx={{ mb: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
        <Typography variant="subtitle2" gutterBottom fontWeight="bold">
          Market Summary
        </Typography>
        <Box display="flex" justifyContent="space-around" textAlign="center">
          <Box>
            <Typography variant="h6" color="success.main" fontWeight="bold">{gainers.length}</Typography>
            <Typography variant="caption" color="textSecondary">Gainers</Typography>
          </Box>
          <Box>
            <Typography variant="h6" color="error.main" fontWeight="bold">{losers.length}</Typography>
            <Typography variant="caption" color="textSecondary">Losers</Typography>
          </Box>
          <Box>
            <Typography variant="h6" color="warning.main" fontWeight="bold">{unchanged.length}</Typography>
            <Typography variant="caption" color="textSecondary">Unchanged</Typography>
          </Box>
          <Box>
            <Typography variant="h6" fontWeight="bold" color={
              gainers.length > losers.length ? 'success.main' : 
              losers.length > gainers.length ? 'error.main' : 'warning.main'
            }>
              {gainers.length > losers.length ? 'Bullish' : 
               losers.length > gainers.length ? 'Bearish' : 'Neutral'}
            </Typography>
            <Typography variant="caption" color="textSecondary">Trend</Typography>
          </Box>
        </Box>
      </Box>

      <Grid container spacing={2}>
        {/* Volume Analysis */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Trading Volume Analysis
              </Typography>
              <Box sx={{ height: 250 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={volumeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="symbol" tick={{ fontSize: 12 }} />
                    <YAxis tickFormatter={formatVolume} tick={{ fontSize: 12 }} />
                    <Tooltip 
                      formatter={(value: any) => [formatVolume(value), 'Volume']}
                      labelFormatter={(label) => `Stock: ${label}`}
                    />
                    <Bar dataKey="volume" fill="#1976d2" />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Top Movers */}
        <Grid item xs={12}>
          <Typography variant="subtitle2" gutterBottom fontWeight="bold" sx={{ mt: 2 }}>
            Top Movers
          </Typography>
          <Grid container spacing={1}>
            {marketData.slice(0, 8).map((stock) => (
              <Grid item xs={6} sm={4} md={3} key={stock.symbol}>
                <Box 
                  sx={{ 
                    p: 1.5, 
                    textAlign: 'center',
                    bgcolor: selectedStock === stock.symbol ? 'primary.light' : 'background.paper',
                    border: '1px solid',
                    borderColor: selectedStock === stock.symbol ? 'primary.main' : 'grey.300',
                    borderRadius: 1,
                  }}
                >
                  <Typography variant="caption" fontWeight="bold" color="textSecondary">
                    {stock.symbol}
                  </Typography>
                  <Typography variant="body1" fontWeight="bold">
                    ₹{stock.last_price.toFixed(2)}
                  </Typography>
                  <Box display="flex" alignItems="center" justifyContent="center" gap={0.5}>
                    {(stock.changePercent || 0) > 0 ? (
                      <TrendingUp fontSize="small" color="success" />
                    ) : (
                      <TrendingDown fontSize="small" color="error" />
                    )}
                    <Typography 
                      variant="caption" 
                      color={(stock.changePercent || 0) > 0 ? 'success.main' : 'error.main'}
                      fontWeight="bold"
                    >
                      {((stock.changePercent || 0) >= 0 ? '+' : '')}{(stock.changePercent || 0).toFixed(2)}%
                    </Typography>
                  </Box>
                </Box>
              </Grid>
            ))}
          </Grid>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MarketOverview;
