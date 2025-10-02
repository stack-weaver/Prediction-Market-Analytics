import { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Grid,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Divider,
  useMediaQuery,
  IconButton,
  Container,
  Paper,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Dashboard,
  TrendingUp,
  Analytics,
  AccountBalance,
  Article,
  Menu as MenuIcon,
  Close as CloseIcon,
  ShowChart,
  Assessment,
  ViewInAr,
} from '@mui/icons-material';
import StockSelector from './components/StockSelector';
import PredictionCard from './components/PredictionCard';
import AnalysisCard from './components/AnalysisCard';
import ChartCard from './components/ChartCard';
import CorrelationHeatmap from './components/CorrelationHeatmap';
import PortfolioDashboard from './components/PortfolioDashboard';
import EnhancedCandlestickChart from './components/EnhancedCandlestickChart';
import PriceChart from './components/PriceChart';
import MarketOverview from './components/MarketOverview';
import NewsSentimentDashboard from './components/NewsSentimentDashboard';
import Interactive3DVisualization from './components/Interactive3DVisualization';
import DatasetInfo from './components/DatasetInfo';
import { StockInfo, PredictionResponse, AnalysisResponse } from './types';
import { getStocks, getPrediction, getAnalysis, getChartData } from './services/api';

const drawerWidth = 280;

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f8fafc',
      paper: '#ffffff',
    },
    grey: {
      50: '#f8fafc',
      100: '#f1f5f9',
      200: '#e2e8f0',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h6: {
      fontWeight: 600,
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
          border: '1px solid #e2e8f0',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
          border: '1px solid #e2e8f0',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
  },
  spacing: 8,
});

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
      id={`dashboard-tabpanel-${index}`}
      aria-labelledby={`dashboard-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

function App() {
  const [stocks, setStocks] = useState<StockInfo[]>([]);
  const [selectedStock, setSelectedStock] = useState<string>('');
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [chartData, setChartData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [activeTab, setActiveTab] = useState(0);

  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  useEffect(() => {
    loadStocks();
  }, []);

  useEffect(() => {
    if (selectedStock) {
      loadStockData();
    }
  }, [selectedStock]);

  const loadStocks = async () => {
    try {
      const stocksData = await getStocks();
      setStocks(stocksData);
      if (stocksData.length > 0) {
        setSelectedStock(stocksData[0].symbol);
      }
    } catch (error) {
      console.error('Error loading stocks:', error);
    }
  };

  const loadStockData = async () => {
    if (!selectedStock) return;
    
    setLoading(true);
    try {
      const [predictionData, analysisData, chartDataResponse] = await Promise.all([
        getPrediction(selectedStock),
        getAnalysis(selectedStock),
        getChartData(selectedStock)
      ]);
      
      setPrediction(predictionData);
      setAnalysis(analysisData);
      setChartData(chartDataResponse);
    } catch (error) {
      console.error('Error loading stock data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
    // Scroll to top when changing tabs
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const menuItems = [
    { text: 'Overview', icon: <Dashboard />, tab: 0 },
    { text: 'Predictions', icon: <TrendingUp />, tab: 1 },
    { text: 'Charts', icon: <ShowChart />, tab: 2 },
    { text: 'Analysis', icon: <Analytics />, tab: 3 },
    { text: 'Portfolio', icon: <AccountBalance />, tab: 4 },
    { text: 'News & Sentiment', icon: <Article />, tab: 5 },
    { text: '3D Visualization', icon: <ViewInAr />, tab: 6 },
  ];

  const drawer = (
    <Box>
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
          📈 IntelliStock Pro
        </Typography>
        <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mb: 1 }}>
          AI-Powered Stock Prediction & Analytics
        </Typography>
        
        {/* Developer Badge */}
        <Box 
          component="a"
          href="https://github.com/HimanshuSalunke"
          target="_blank"
          rel="noopener noreferrer"
          sx={{ 
            mt: 1,
            p: 1,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            borderRadius: 2,
            textAlign: 'center',
            boxShadow: '0 2px 8px rgba(102, 126, 234, 0.3)',
            position: 'relative',
            overflow: 'hidden',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            textDecoration: 'none',
            display: 'block',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 4px 12px rgba(102, 126, 234, 0.5)',
            },
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: '-100%',
            width: '100%',
            height: '100%',
            background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)',
            animation: 'shimmer 3s infinite',
          },
          '@keyframes shimmer': {
            '0%': { left: '-100%' },
            '100%': { left: '100%' },
          },
        }}>
          <Typography 
            variant="caption" 
            sx={{ 
              color: 'white',
              fontWeight: 'bold',
              fontSize: '0.8rem',
              textTransform: 'uppercase',
              letterSpacing: '1px',
              textShadow: '1px 1px 2px rgba(0,0,0,0.3)',
              display: 'block'
            }}
          >
            🚀 HIMANSHU SALUNKE
          </Typography>
          <Typography 
            variant="caption" 
            sx={{ 
              color: 'rgba(255,255,255,0.9)',
              fontSize: '0.65rem',
              fontStyle: 'italic'
            }}
          >
            Data Science & AI Enthusiast
          </Typography>
        </Box>
      </Box>
      
      <Box sx={{ p: 2 }}>
        <StockSelector
          stocks={stocks}
          selectedStock={selectedStock}
          onStockChange={setSelectedStock}
          loading={loading}
        />
      </Box>

      <Divider />

      <List sx={{ px: 1 }}>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={activeTab === item.tab}
              onClick={() => {
                setActiveTab(item.tab);
                window.scrollTo({ top: 0, behavior: 'smooth' });
                if (isMobile) setMobileOpen(false);
              }}
              sx={{
                borderRadius: 2,
                mb: 0.5,
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'white',
                  '& .MuiListItemIcon-root': {
                    color: 'white',
                  },
                },
              }}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex' }}>
        {/* App Bar */}
        <AppBar
          position="fixed"
          sx={{
            width: { md: `calc(100% - ${drawerWidth}px)` },
            ml: { md: `${drawerWidth}px` },
          }}
        >
          <Toolbar>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2, display: { md: 'none' } }}
            >
              <MenuIcon />
            </IconButton>
            <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="h6" noWrap component="div">
                {selectedStock ? `${selectedStock} - IntelliStock Pro Analysis` : 'IntelliStock Pro'}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ 
                background: 'linear-gradient(45deg, #FF6B35 30%, #F7931E 90%)',
                borderRadius: 2,
                px: 2,
                py: 0.5,
                boxShadow: '0 3px 5px 2px rgba(255, 105, 135, .3)',
                animation: 'pulse 2s infinite',
                '@keyframes pulse': {
                  '0%': { 
                    transform: 'scale(1)',
                    boxShadow: '0 3px 5px 2px rgba(255, 105, 135, .3)',
                  },
                  '50%': { 
                    transform: 'scale(1.05)',
                    boxShadow: '0 4px 8px 3px rgba(255, 105, 135, .4)',
                  },
                  '100%': { 
                    transform: 'scale(1)',
                    boxShadow: '0 3px 5px 2px rgba(255, 105, 135, .3)',
                  },
                },
              }}>
                <Typography 
                  variant="subtitle1" 
                  sx={{ 
                    color: 'white',
                    fontWeight: 'bold',
                    fontSize: '1rem',
                    textShadow: '1px 1px 2px rgba(0,0,0,0.3)',
                    letterSpacing: '0.5px'
                  }}
                >
                  HIMANSHU SALUNKE
                </Typography>
              </Box>
              <Typography variant="caption" sx={{ 
                color: 'white', 
                opacity: 0.9,
                fontStyle: 'italic',
                fontSize: '0.75rem'
              }}>
                AI/ML Developer
              </Typography>
            </Box>
          </Toolbar>
        </AppBar>

        {/* Navigation Drawer */}
        <Box
          component="nav"
          sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
        >
          <Drawer
            variant="temporary"
            open={mobileOpen}
            onClose={handleDrawerToggle}
            ModalProps={{
              keepMounted: true,
            }}
            sx={{
              display: { xs: 'block', md: 'none' },
              '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
            }}
          >
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', p: 1 }}>
              <IconButton onClick={handleDrawerToggle}>
                <CloseIcon />
              </IconButton>
            </Box>
            {drawer}
          </Drawer>
          <Drawer
            variant="permanent"
            sx={{
              display: { xs: 'none', md: 'block' },
              '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
            }}
            open
          >
            {drawer}
          </Drawer>
        </Box>

        {/* Main Content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            width: { md: `calc(100% - ${drawerWidth}px)` },
            minHeight: '100vh',
            backgroundColor: 'background.default',
          }}
        >
          <Toolbar />
          
          <Container maxWidth="xl" sx={{ py: 3 }}>
            {/* Overview Tab */}
            <TabPanel value={activeTab} index={0}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {/* Stock Header */}
                {selectedStock && (
                  <Paper sx={{ p: 2, bgcolor: 'primary.main', color: 'white' }}>
                    <Typography variant="h5" fontWeight="bold">
                      {selectedStock} - Stock Analysis Overview
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                      Real-time insights powered by AI predictions and technical analysis
                    </Typography>
                  </Paper>
                )}

                {/* Quick Summary Cards */}
                <Box>
                  <Typography variant="h6" gutterBottom sx={{ mb: 2, fontWeight: 600 }}>
                    📊 Quick Summary
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {/* Quick Prediction Summary */}
                    <Paper sx={{ p: 2, border: '1px solid', borderColor: 'success.main', bgcolor: 'success.light' }}>
                      <Typography variant="subtitle1" fontWeight="bold" color="success.contrastText" gutterBottom>
                        🔮 AI Prediction Summary
                      </Typography>
                      {prediction ? (
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
                          <Box>
                            <Typography variant="caption" color="success.contrastText" sx={{ opacity: 0.8 }}>Next Day Target</Typography>
                            <Typography variant="h6" fontWeight="bold" color="success.contrastText">
                              ₹{prediction.ensemble_prediction[0].toFixed(2)}
                            </Typography>
                          </Box>
                          <Box>
                            <Typography variant="caption" color="success.contrastText" sx={{ opacity: 0.8 }}>Confidence</Typography>
                            <Typography variant="h6" fontWeight="bold" color="success.contrastText">
                              {(prediction.confidence_score * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                          <Box>
                            <Typography variant="caption" color="success.contrastText" sx={{ opacity: 0.8 }}>Trend</Typography>
                            <Typography variant="h6" fontWeight="bold" color="success.contrastText">
                              {prediction.trend}
                            </Typography>
                          </Box>
                        </Box>
                      ) : (
                        <Typography color="success.contrastText" sx={{ opacity: 0.8 }}>
                          Select a stock to see AI predictions
                        </Typography>
                      )}
                    </Paper>

                    {/* Quick Analysis Summary */}
                    <Paper sx={{ p: 2, border: '1px solid', borderColor: 'info.main', bgcolor: 'info.light' }}>
                      <Typography variant="subtitle1" fontWeight="bold" color="info.contrastText" gutterBottom>
                        📈 Technical Summary
                      </Typography>
                      {analysis ? (
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
                          <Box>
                            <Typography variant="caption" color="info.contrastText" sx={{ opacity: 0.8 }}>RSI</Typography>
                            <Typography variant="h6" fontWeight="bold" color="info.contrastText">
                              {analysis.technical_indicators.rsi.toFixed(1)}
                            </Typography>
                          </Box>
                          <Box>
                            <Typography variant="caption" color="info.contrastText" sx={{ opacity: 0.8 }}>Volatility</Typography>
                            <Typography variant="h6" fontWeight="bold" color="info.contrastText">
                              {(analysis.risk_metrics.volatility * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                          <Box>
                            <Typography variant="caption" color="info.contrastText" sx={{ opacity: 0.8 }}>Sharpe Ratio</Typography>
                            <Typography variant="h6" fontWeight="bold" color="info.contrastText">
                              {analysis.risk_metrics.sharpe_ratio.toFixed(2)}
                            </Typography>
                          </Box>
                        </Box>
                      ) : (
                        <Typography color="info.contrastText" sx={{ opacity: 0.8 }}>
                          Select a stock to see technical analysis
                        </Typography>
                      )}
                    </Paper>
                  </Box>
                </Box>

                {/* Main Price Chart */}
                <Box>
                  <Typography variant="h6" gutterBottom sx={{ mb: 2, fontWeight: 600 }}>
                    📈 Price Chart
                  </Typography>
                  <Paper sx={{ p: 3 }}>
                    <PriceChart
                      chartData={chartData}
                      prediction={prediction}
                      symbol={selectedStock}
                      loading={loading}
                    />
                  </Paper>
                </Box>

                {/* Market Analysis Section */}
                <Box>
                  <Typography variant="h6" gutterBottom sx={{ mb: 2, fontWeight: 600 }}>
                    🔍 Market Analysis
                  </Typography>
                  <Paper sx={{ p: 3, mb: 2 }}>
                    <MarketOverview selectedStock={selectedStock} />
                  </Paper>
                </Box>

                {/* Dataset Information */}
                <DatasetInfo />

                {/* Quick Actions */}
                <Paper sx={{ p: 2, bgcolor: 'grey.50', border: '1px solid', borderColor: 'grey.200' }}>
                  <Typography variant="subtitle2" gutterBottom fontWeight="bold">
                    💡 Quick Actions
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                    <Box sx={{ flex: 1, minWidth: 200 }}>
                      <Typography variant="caption" color="textSecondary">
                        Switch to Predictions tab for detailed ML model analysis
                      </Typography>
                    </Box>
                    <Box sx={{ flex: 1, minWidth: 200 }}>
                      <Typography variant="caption" color="textSecondary">
                        Visit Charts tab for advanced technical indicators
                      </Typography>
                    </Box>
                    <Box sx={{ flex: 1, minWidth: 200 }}>
                      <Typography variant="caption" color="textSecondary">
                        Check Portfolio tab for risk management tools
                      </Typography>
                    </Box>
                  </Box>
                </Paper>
              </Box>
            </TabPanel>

            {/* Predictions Tab */}
            <TabPanel value={activeTab} index={1}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {/* Predictions Header */}
                {selectedStock && (
                  <Paper sx={{ p: 2, bgcolor: 'success.main', color: 'white' }}>
                    <Typography variant="h5" fontWeight="bold">
                      🔮 AI Predictions for {selectedStock}
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                      Advanced machine learning models predict future price movements
                    </Typography>
                  </Paper>
                )}

                {/* Main Prediction Card */}
                <Box>
                  <Typography variant="h6" gutterBottom sx={{ mb: 2, fontWeight: 600 }}>
                    📈 ML Model Predictions
                  </Typography>
                  <PredictionCard prediction={prediction} loading={loading} />
                </Box>

                {/* Prediction Confidence Analysis */}
                <Box>
                  <Typography variant="h6" gutterBottom sx={{ mb: 2, fontWeight: 600 }}>
                    🎯 Confidence Analysis
                  </Typography>
                  <Paper sx={{ p: 3 }}>
                    {prediction ? (
                      <Box>
                        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
                          <Box sx={{ textAlign: 'center', p: 3, bgcolor: 'primary.light', borderRadius: 2, minWidth: 200 }}>
                            <Typography variant="h2" color="primary.contrastText" fontWeight="bold">
                              {(prediction.confidence_score * 100).toFixed(1)}%
                            </Typography>
                            <Typography variant="h6" color="primary.contrastText">
                              Model Confidence
                            </Typography>
                          </Box>
                        </Box>
                        
                        <Box sx={{ display: 'flex', justifyContent: 'space-around', textAlign: 'center', flexWrap: 'wrap', gap: 2 }}>
                          <Box sx={{ minWidth: 120 }}>
                            <Typography variant="caption" color="textSecondary">Models Agreement</Typography>
                            <Typography variant="h5" fontWeight="bold">{Object.keys(prediction.predictions).length}</Typography>
                            <Typography variant="body2" color="textSecondary">ML Models</Typography>
                          </Box>
                          <Box sx={{ minWidth: 120 }}>
                            <Typography variant="caption" color="textSecondary">Prediction Trend</Typography>
                            <Typography variant="h5" fontWeight="bold" color={
                              prediction.trend === 'BULLISH' ? 'success.main' : 
                              prediction.trend === 'BEARISH' ? 'error.main' : 'warning.main'
                            }>
                              {prediction.trend}
                            </Typography>
                            <Typography variant="body2" color="textSecondary">Market Direction</Typography>
                          </Box>
                          <Box sx={{ minWidth: 120 }}>
                            <Typography variant="caption" color="textSecondary">Next Day Target</Typography>
                            <Typography variant="h5" fontWeight="bold" color="primary.main">
                              ₹{prediction.ensemble_prediction[0].toFixed(2)}
                            </Typography>
                            <Typography variant="body2" color="textSecondary">Predicted Price</Typography>
                          </Box>
                        </Box>
                      </Box>
                    ) : (
                      <Box sx={{ textAlign: 'center', py: 4 }}>
                        <Typography color="textSecondary">
                          Select a stock to see detailed prediction confidence analysis
                        </Typography>
                      </Box>
                    )}
                  </Paper>
                </Box>

                {/* Model Performance Comparison */}
                <Box>
                  <Typography variant="h6" gutterBottom sx={{ mb: 2, fontWeight: 600 }}>
                    🤖 Individual Model Results
                  </Typography>
                  <Paper sx={{ p: 3 }}>
                    {prediction ? (
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        {Object.entries(prediction.predictions).map(([modelName, predictions]) => (
                          <Box key={modelName} sx={{ p: 2, border: '1px solid', borderColor: 'grey.300', borderRadius: 1 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Typography variant="subtitle1" fontWeight="bold">
                                {modelName.toUpperCase()}
                              </Typography>
                              <Typography variant="h6" color="primary.main" fontWeight="bold">
                                ₹{predictions[0].toFixed(2)}
                              </Typography>
                            </Box>
                            <Typography variant="body2" color="textSecondary">
                              5-day forecast: {predictions.map(p => `₹${p.toFixed(2)}`).join(' → ')}
                            </Typography>
                          </Box>
                        ))}
                      </Box>
                    ) : (
                      <Typography color="textSecondary" textAlign="center">
                        Individual model predictions will appear here
                      </Typography>
                    )}
                  </Paper>
                </Box>
              </Box>
            </TabPanel>

            {/* Charts Tab */}
            <TabPanel value={activeTab} index={2}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {/* Charts Header */}
                {selectedStock && (
                  <Paper sx={{ p: 2, bgcolor: 'warning.main', color: 'white' }}>
                    <Typography variant="h5" fontWeight="bold">
                      📈 Advanced Charts for {selectedStock}
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                      Professional trading charts with technical indicators and patterns
                    </Typography>
                  </Paper>
                )}

                {/* Enhanced Candlestick Chart */}
                <Box>
                  <Typography variant="h6" gutterBottom sx={{ mb: 2, fontWeight: 600 }}>
                    🕯️ Enhanced Candlestick Chart
                  </Typography>
                  <Paper sx={{ p: 3 }}>
                    {selectedStock ? (
                      <EnhancedCandlestickChart symbol={selectedStock} />
                    ) : (
                      <Box sx={{ textAlign: 'center', py: 4 }}>
                        <Typography color="textSecondary">
                          Select a stock to view advanced candlestick charts with technical indicators
                        </Typography>
                      </Box>
                    )}
                  </Paper>
                </Box>

                {/* Chart Features Info */}
                <Paper sx={{ p: 2, bgcolor: 'grey.50', border: '1px solid', borderColor: 'grey.200' }}>
                  <Typography variant="subtitle2" gutterBottom fontWeight="bold">
                    📊 Chart Features
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Typography variant="body2" color="textSecondary">
                      • Interactive candlestick patterns with OHLC data
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • Moving averages (SMA 20, SMA 50) and Bollinger Bands
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • Volume analysis and trading indicators
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • AI prediction overlays with confidence bands
                    </Typography>
                  </Box>
                </Paper>
              </Box>
            </TabPanel>

            {/* Analysis Tab */}
            <TabPanel value={activeTab} index={3}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {/* Analysis Header */}
                {selectedStock && (
                  <Paper sx={{ p: 2, bgcolor: 'info.main', color: 'white' }}>
                    <Typography variant="h5" fontWeight="bold">
                      📊 Technical Analysis for {selectedStock}
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                      Comprehensive technical indicators, risk metrics, and market analysis
                    </Typography>
                  </Paper>
                )}

                {/* Technical Analysis Card */}
                <Box>
                  <Typography variant="h6" gutterBottom sx={{ mb: 2, fontWeight: 600 }}>
                    🔍 Technical Indicators & Risk Metrics
                  </Typography>
                  <AnalysisCard analysis={analysis} loading={loading} />
                </Box>

                {/* Advanced Technical Charts */}
                <Box>
                  <Typography variant="h6" gutterBottom sx={{ mb: 2, fontWeight: 600 }}>
                    📈 Multi-Format Price Charts
                  </Typography>
                  <Paper sx={{ p: 3 }}>
                    <ChartCard
                      chartData={chartData}
                      symbol={selectedStock}
                      loading={loading}
                    />
                  </Paper>
                </Box>

                {/* Market Correlation Analysis */}
                <Box>
                  <Typography variant="h6" gutterBottom sx={{ mb: 2, fontWeight: 600 }}>
                    🔗 Stock Correlation Matrix
                  </Typography>
                  <Paper sx={{ p: 3 }}>
                    <CorrelationHeatmap />
                  </Paper>
                </Box>

                {/* Analysis Summary */}
                <Paper sx={{ p: 2, bgcolor: 'grey.50', border: '1px solid', borderColor: 'grey.200' }}>
                  <Typography variant="subtitle2" gutterBottom fontWeight="bold">
                    📋 Analysis Summary
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Typography variant="body2" color="textSecondary">
                      • Technical indicators include RSI, MACD, Bollinger Bands, and moving averages
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • Risk metrics cover volatility, Sharpe ratio, maximum drawdown, and VaR
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • Multiple chart formats: Line, OHLC, Area, and Volume analysis
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • Correlation matrix shows relationships between different stocks
                    </Typography>
                  </Box>
                </Paper>
              </Box>
            </TabPanel>

            {/* Portfolio Tab */}
            <TabPanel value={activeTab} index={4}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {/* Portfolio Header */}
                <Paper sx={{ p: 2, bgcolor: 'secondary.main', color: 'white' }}>
                  <Typography variant="h5" fontWeight="bold">
                    💼 Portfolio Management
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    Diversified portfolio analysis with risk management and optimization tools
                  </Typography>
                </Paper>

                {/* Portfolio Dashboard */}
                <Box>
                  <Typography variant="h6" gutterBottom sx={{ mb: 2, fontWeight: 600 }}>
                    📈 Portfolio Analytics
                  </Typography>
                  <Paper sx={{ p: 3 }}>
                    <PortfolioDashboard />
                  </Paper>
                </Box>

                {/* Portfolio Features */}
                <Paper sx={{ p: 2, bgcolor: 'grey.50', border: '1px solid', borderColor: 'grey.200' }}>
                  <Typography variant="subtitle2" gutterBottom fontWeight="bold">
                    🎯 Portfolio Features
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Typography variant="body2" color="textSecondary">
                      • Multiple portfolio strategies: Default, Tech Focus, Banking, Diversified
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • Real-time allocation analysis with interactive pie charts
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • Performance tracking with returns and risk metrics
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • Portfolio optimization suggestions and rebalancing insights
                    </Typography>
                  </Box>
                </Paper>
              </Box>
            </TabPanel>

            {/* News & Sentiment Tab */}
            <TabPanel value={activeTab} index={5}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {/* News Header */}
                <Paper sx={{ p: 2, bgcolor: 'error.main', color: 'white' }}>
                  <Typography variant="h5" fontWeight="bold">
                    📰 Market News & Sentiment
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    AI-powered sentiment analysis from financial news and market data
                  </Typography>
                </Paper>

                {/* News Sentiment Dashboard */}
                <Box>
                  <Typography variant="h6" gutterBottom sx={{ mb: 2, fontWeight: 600 }}>
                    📊 Sentiment Analysis Dashboard
                  </Typography>
                  <Paper sx={{ p: 3 }}>
                    <NewsSentimentDashboard />
                  </Paper>
                </Box>

                {/* Sentiment Features */}
                <Paper sx={{ p: 2, bgcolor: 'grey.50', border: '1px solid', borderColor: 'grey.200' }}>
                  <Typography variant="subtitle2" gutterBottom fontWeight="bold">
                    🤖 AI Sentiment Features
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Typography variant="body2" color="textSecondary">
                      • Real-time news sentiment analysis from multiple financial sources
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • Market sentiment gauge with positive/negative/neutral classification
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • Historical sentiment trends and correlation with price movements
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • Stock-specific sentiment analysis for targeted insights
                    </Typography>
                  </Box>
                </Paper>
              </Box>
            </TabPanel>

            {/* 3D Visualization Tab */}
            <TabPanel value={activeTab} index={6}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {/* 3D Header */}
                <Paper sx={{ p: 2, bgcolor: 'success.dark', color: 'white' }}>
                  <Typography variant="h5" fontWeight="bold">
                    🌐 3D Risk-Return Visualization
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    Interactive 3D visualization of portfolio risk, returns, and correlations
                  </Typography>
                </Paper>

                {/* 3D Visualization */}
                <Box>
                  <Typography variant="h6" gutterBottom sx={{ mb: 2, fontWeight: 600 }}>
                    🎮 Interactive 3D Analysis
                  </Typography>
                  <Paper sx={{ p: 3 }}>
                    <Interactive3DVisualization />
                  </Paper>
                </Box>

                {/* 3D Features */}
                <Paper sx={{ p: 2, bgcolor: 'grey.50', border: '1px solid', borderColor: 'grey.200' }}>
                  <Typography variant="subtitle2" gutterBottom fontWeight="bold">
                    🎯 3D Visualization Features
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Typography variant="body2" color="textSecondary">
                      • Interactive 3D risk surface with mouse/touch controls
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • Real-time correlation network visualization between stocks
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • Portfolio efficient frontier in 3D space
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      • Hardware-accelerated rendering with Three.js for smooth performance
                    </Typography>
                  </Box>
                </Paper>
              </Box>
            </TabPanel>
          </Container>
          
          {/* Footer */}
          <Box 
            component="footer" 
            sx={{ 
              mt: 4, 
              py: 3, 
              px: 2, 
              backgroundColor: 'background.paper',
              borderTop: 1,
              borderColor: 'divider'
            }}
          >
            <Container maxWidth="lg">
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  © 2025 IntelliStock Pro - AI-Powered Stock Prediction & Analytics Platform
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 2, flexWrap: 'wrap' }}>
                  <Typography variant="body2" color="text.secondary">
                    Developed by
                  </Typography>
                  <Box sx={{
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    borderRadius: 3,
                    px: 2,
                    py: 0.5,
                    boxShadow: '0 2px 6px rgba(102, 126, 234, 0.3)',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 4px 12px rgba(102, 126, 234, 0.4)',
                    }
                  }}>
                    <Typography 
                      component="a" 
                      href="https://github.com/HimanshuSalunke" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      sx={{ 
                        color: 'white',
                        textDecoration: 'none',
                        fontWeight: 'bold',
                        fontSize: '0.9rem',
                        textTransform: 'uppercase',
                        letterSpacing: '0.5px',
                        textShadow: '1px 1px 2px rgba(0,0,0,0.3)',
                      }}
                    >
                      🚀 HIMANSHU SALUNKE
                    </Typography>
                  </Box>
                  <Typography 
                    component="a" 
                    href="https://www.linkedin.com/in/himanshuksalunke/" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    sx={{ 
                      color: 'primary.main', 
                      textDecoration: 'none',
                      fontWeight: 'bold',
                      '&:hover': { 
                        textDecoration: 'underline',
                        color: 'primary.dark'
                      }
                    }}
                  >
                    LinkedIn →
                  </Typography>
                </Box>
              </Box>
            </Container>
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;