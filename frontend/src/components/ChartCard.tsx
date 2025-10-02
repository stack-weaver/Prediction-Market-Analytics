import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Tabs,
  Tab,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Area,
  AreaChart,
  ComposedChart,
} from 'recharts';
import { ChartData } from '../types';

interface ChartCardProps {
  chartData: ChartData | null;
  symbol: string;
  loading: boolean;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`chart-tabpanel-${index}`}
      aria-labelledby={`chart-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const ChartCard: React.FC<ChartCardProps> = ({ chartData, symbol, loading }) => {
  const [tabValue, setTabValue] = React.useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Price Chart
          </Typography>
          <LinearProgress />
        </CardContent>
      </Card>
    );
  }

  if (!chartData) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Price Chart
          </Typography>
          <Typography color="text.secondary">
            Select a stock to see chart
          </Typography>
        </CardContent>
      </Card>
    );
  }

  // Prepare data for charts
  const priceData = chartData.dates.map((date, index) => ({
    date,
    price: chartData.prices[index],
    high: chartData.high[index],
    low: chartData.low[index],
    open: chartData.open[index],
    volume: chartData.volumes[index],
    // Candlestick data
    ohlc: [chartData.open[index], chartData.high[index], chartData.low[index], chartData.prices[index]],
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box sx={{ 
          bgcolor: 'background.paper', 
          p: 1, 
          border: '1px solid #ccc',
          borderRadius: 1 
        }}>
          <Typography variant="body2">{`Date: ${label}`}</Typography>
          <Typography variant="body2" color="primary">
            {`Price: ₹${payload[0].value.toFixed(2)}`}
          </Typography>
        </Box>
      );
    }
    return null;
  };

  const CandlestickTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Box sx={{ 
          bgcolor: 'background.paper', 
          p: 1, 
          border: '1px solid #ccc',
          borderRadius: 1 
        }}>
          <Typography variant="body2">{`Date: ${label}`}</Typography>
          <Typography variant="body2" color="primary">
            {`Open: ₹${data.open.toFixed(2)}`}
          </Typography>
          <Typography variant="body2" color="primary">
            {`High: ₹${data.high.toFixed(2)}`}
          </Typography>
          <Typography variant="body2" color="primary">
            {`Low: ₹${data.low.toFixed(2)}`}
          </Typography>
          <Typography variant="body2" color="primary">
            {`Close: ₹${data.price.toFixed(2)}`}
          </Typography>
        </Box>
      );
    }
    return null;
  };

  return (
    <Card sx={{ width: '100%' }}>
      <CardContent sx={{ p: { xs: 1, sm: 2 } }}>
        <Typography variant="h6" gutterBottom sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }}>
          Advanced Charts - {symbol}
        </Typography>
        
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="chart tabs">
            <Tab label="Line Chart" />
            <Tab label="OHLC" />
            <Tab label="Area Chart" />
            <Tab label="Volume" />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <Box sx={{ height: 400 }}>
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
                />
                <Tooltip content={<CustomTooltip />} />
                <Line 
                  type="monotone" 
                  dataKey="price" 
                  stroke="#1976d2" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Box sx={{ height: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={priceData}>
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
                <Tooltip content={<CandlestickTooltip />} />
                <Line 
                  type="monotone" 
                  dataKey="high" 
                  stroke="#4CAF50" 
                  strokeWidth={1}
                  dot={false}
                  name="High"
                />
                <Line 
                  type="monotone" 
                  dataKey="low" 
                  stroke="#F44336" 
                  strokeWidth={1}
                  dot={false}
                  name="Low"
                />
                <Line 
                  type="monotone" 
                  dataKey="open" 
                  stroke="#FF9800" 
                  strokeWidth={2}
                  dot={false}
                  name="Open"
                />
                <Line 
                  type="monotone" 
                  dataKey="price" 
                  stroke="#1976d2" 
                  strokeWidth={3}
                  dot={false}
                  name="Close"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </Box>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Box sx={{ height: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={priceData}>
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
                <Area 
                  type="monotone" 
                  dataKey="price" 
                  stroke="#1976d2" 
                  fill="#1976d2"
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </Box>
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <Box sx={{ height: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={priceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="volume" fill="#dc004e" />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </TabPanel>
      </CardContent>
    </Card>
  );
};

export default ChartCard;
