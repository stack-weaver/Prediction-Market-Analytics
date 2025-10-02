import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  CircularProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';
import { TrendingUp, TrendingDown, AccountBalance } from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8002';

interface PortfolioItem {
  symbol: string;
  shares: number;
  current_price: number;
  value: number;
  weight: number;
}

interface PortfolioData {
  portfolio: PortfolioItem[];
  total_value: number;
  symbols: string[];
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#ff7c7c'];

const PortfolioDashboard: React.FC = () => {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [portfolioType, setPortfolioType] = useState<string>('default');

  const portfolioOptions = {
    default: 'RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK',
    tech: 'TCS,INFY,WIPRO,HCLTECH,TECHM',
    banking: 'HDFCBANK,ICICIBANK,SBIN,KOTAKBANK,AXISBANK',
    diversified: 'RELIANCE,TCS,HDFCBANK,BHARTIARTL,ITC,HINDUNILVR,ASIANPAINT,MARUTI'
  };

  const fetchPortfolioData = async (symbols: string) => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.get(`${API_BASE_URL}/api/portfolio/allocation?symbols=${symbols}`);
      setPortfolioData(response.data);
    } catch (err) {
      setError('Failed to fetch portfolio data');
      console.error('Portfolio fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const symbols = portfolioOptions[portfolioType as keyof typeof portfolioOptions];
    fetchPortfolioData(symbols);
  }, [portfolioType]);

  const handlePortfolioChange = (event: SelectChangeEvent<string>) => {
    setPortfolioType(event.target.value);
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Paper sx={{ p: 2, bgcolor: 'background.paper' }}>
          <Typography variant="body2" fontWeight="bold">
            {data.symbol}
          </Typography>
          <Typography variant="body2">
            Value: {formatCurrency(data.value)}
          </Typography>
          <Typography variant="body2">
            Weight: {data.weight}%
          </Typography>
          <Typography variant="body2">
            Shares: {data.shares}
          </Typography>
        </Paper>
      );
    }
    return null;
  };

  const pieData = portfolioData?.portfolio.map((item, index) => ({
    ...item,
    fill: COLORS[index % COLORS.length]
  })) || [];

  const barData = portfolioData?.portfolio.map(item => ({
    symbol: item.symbol,
    value: item.value,
    weight: item.weight
  })) || [];

  // Calculate some portfolio metrics
  const avgWeight = portfolioData ? 100 / portfolioData.portfolio.length : 0;
  const maxWeight = portfolioData ? Math.max(...portfolioData.portfolio.map(p => p.weight)) : 0;
  const minWeight = portfolioData ? Math.min(...portfolioData.portfolio.map(p => p.weight)) : 0;

  return (
    <Card sx={{ height: '100%', minWidth: 0 }}>
      <CardContent sx={{ p: { xs: 1, sm: 2 } }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2} flexWrap="wrap">
          <Typography variant="h6" gutterBottom sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }}>
            🥧 Portfolio Dashboard
          </Typography>
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Portfolio Type</InputLabel>
            <Select
              value={portfolioType}
              label="Portfolio Type"
              onChange={handlePortfolioChange}
            >
              <MenuItem value="default">Balanced</MenuItem>
              <MenuItem value="tech">Technology</MenuItem>
              <MenuItem value="banking">Banking</MenuItem>
              <MenuItem value="diversified">Diversified</MenuItem>
            </Select>
          </FormControl>
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

        {portfolioData && !loading && (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* Portfolio Summary */}
            <Box>
              <Box display="flex" gap={2} flexWrap="wrap">
                <Chip
                  icon={<AccountBalance />}
                  label={`Total Value: ${formatCurrency(portfolioData.total_value)}`}
                  color="primary"
                  variant="outlined"
                />
                <Chip
                  icon={<TrendingUp />}
                  label={`Holdings: ${portfolioData.portfolio.length} stocks`}
                  color="success"
                  variant="outlined"
                />
                <Chip
                  label={`Avg Weight: ${avgWeight.toFixed(1)}%`}
                  variant="outlined"
                />
              </Box>
            </Box>

            <Grid container spacing={3}>
            {/* Pie Chart */}
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Allocation Breakdown
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ symbol, weight }) => `${symbol} (${weight}%)`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="weight"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </Grid>

            {/* Bar Chart */}
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Value Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="symbol" 
                    tick={{ fontSize: 12 }}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis 
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => {
                      if (value >= 1000000) {
                        return `₹${(value/1000000).toFixed(1)}M`;
                      } else if (value >= 1000) {
                        return `₹${(value/1000).toFixed(1)}K`;
                      } else {
                        return `₹${value.toFixed(0)}`;
                      }
                    }}
                  />
                  <Tooltip 
                    formatter={(value: number) => [formatCurrency(value), 'Value']}
                  />
                  <Bar dataKey="value" fill="#1976d2" />
                </BarChart>
              </ResponsiveContainer>
            </Grid>

            {/* Portfolio Table */}
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Holdings Detail
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell><strong>Symbol</strong></TableCell>
                      <TableCell align="right"><strong>Shares</strong></TableCell>
                      <TableCell align="right"><strong>Price</strong></TableCell>
                      <TableCell align="right"><strong>Value</strong></TableCell>
                      <TableCell align="right"><strong>Weight</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {portfolioData.portfolio.map((item) => (
                      <TableRow key={item.symbol} hover>
                        <TableCell>
                          <Typography variant="body2" fontWeight="bold">
                            {item.symbol}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">{item.shares}</TableCell>
                        <TableCell align="right">
                          {formatCurrency(item.current_price)}
                        </TableCell>
                        <TableCell align="right">
                          {formatCurrency(item.value)}
                        </TableCell>
                        <TableCell align="right">
                          <Chip
                            label={`${item.weight}%`}
                            size="small"
                            color={item.weight > avgWeight ? 'primary' : 'default'}
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>

            {/* Portfolio Metrics */}
            <Grid item xs={12}>
              <Box display="flex" gap={2} flexWrap="wrap">
                <Chip
                  icon={<TrendingUp />}
                  label={`Max Weight: ${maxWeight.toFixed(1)}%`}
                  color="warning"
                  variant="outlined"
                />
                <Chip
                  icon={<TrendingDown />}
                  label={`Min Weight: ${minWeight.toFixed(1)}%`}
                  color="info"
                  variant="outlined"
                />
                <Chip
                  label={`Weight Range: ${(maxWeight - minWeight).toFixed(1)}%`}
                  variant="outlined"
                />
              </Box>
            </Grid>
            </Grid>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default PortfolioDashboard;
