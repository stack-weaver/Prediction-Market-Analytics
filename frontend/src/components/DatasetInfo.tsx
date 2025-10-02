import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Dataset,
  Timeline,
  TrendingUp,
  Storage,
  Assessment,
  Schedule,
} from '@mui/icons-material';

const DatasetInfo: React.FC = () => {
  const stocksList = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
    'ICICIBANK', 'KOTAKBANK', 'ITC', 'SBIN', 'BHARTIARTL',
    'ASIANPAINT', 'MARUTI', 'BAJFINANCE', 'LT', 'HCLTECH',
    'AXISBANK', 'ULTRACEMCO', 'TITAN', 'NESTLEIND', 'WIPRO',
    'TECHM', 'SUNPHARMA', 'NTPC', 'POWERGRID', 'TATASTEEL',
    'JSWSTEEL', 'COALINDIA', 'INDUSINDBK', 'BAJAJ-AUTO', 'HINDALCO',
    'ADANIENT', 'ADANIPORTS', 'GRASIM', 'BRITANNIA', 'DIVISLAB',
    'DRREDDY', 'EICHERMOT', 'BPCL', 'CIPLA', 'HEROMOTOCO',
    'APOLLOHOSP', 'BAJAJFINSV', 'HDFCLIFE', 'SBILIFE', 'LTIM',
    'TATACONSUM', 'TATAMOTORS', 'UPL', 'MM', 'ONGC'
  ];

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Dataset sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
            📊 Dataset Information
          </Typography>
        </Box>

        <Grid container spacing={3}>
          {/* Dataset Overview */}
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1, color: 'primary.main' }}>
              📈 NSE Historical Dataset
            </Typography>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <Timeline color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Time Period" 
                  secondary="January 2022 - December 2024 (3 Years)"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <TrendingUp color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Total Stocks" 
                  secondary="50+ NSE Listed Companies"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Storage color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Data Points" 
                  secondary="Daily & Minute-level OHLCV Data"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Assessment color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Features" 
                  secondary="15+ Technical Indicators + Price Data"
                />
              </ListItem>
            </List>
          </Grid>

          {/* Data Characteristics */}
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1, color: 'primary.main' }}>
              🔍 Data Characteristics
            </Typography>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                <strong>Data Source:</strong> National Stock Exchange (NSE) India
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                <strong>Update Frequency:</strong> Historical data (not real-time)
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                <strong>Data Quality:</strong> Cleaned and validated market data
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                <strong>Coverage:</strong> Major blue-chip and mid-cap stocks
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Schedule sx={{ mr: 1, fontSize: 16, color: 'text.secondary' }} />
              <Typography variant="body2" color="text.secondary">
                <strong>Last Updated:</strong> December 2024
              </Typography>
            </Box>
          </Grid>
        </Grid>

        <Divider sx={{ my: 2 }} />

        {/* Stock List */}
        <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 2, color: 'primary.main' }}>
          📋 Included Stocks ({stocksList.length} Companies)
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
          {stocksList.map((stock) => (
            <Chip
              key={stock}
              label={stock}
              size="small"
              variant="outlined"
              sx={{ 
                fontSize: '0.75rem',
                height: 24,
                '&:hover': {
                  backgroundColor: 'primary.main',
                  color: 'white',
                }
              }}
            />
          ))}
        </Box>

        <Box sx={{ mt: 2, p: 2, backgroundColor: 'info.light', borderRadius: 1 }}>
          <Typography variant="body2" color="info.contrastText">
            <strong>Note:</strong> This system uses historical NSE data for training ML models. 
            Predictions are based on historical patterns and technical analysis. 
            For real-time trading, please use live market data feeds.
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default DatasetInfo;
