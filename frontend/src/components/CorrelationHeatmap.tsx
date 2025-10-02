import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  CircularProgress,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8002';

interface CorrelationData {
  symbols: string[];
  matrix: number[][];
  period: string;
}

const HeatmapContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  padding: theme.spacing(2),
}));

const HeatmapGrid = styled('div')<{ size: number }>(({ size }) => ({
  display: 'grid',
  gridTemplateColumns: `repeat(${size}, 1fr)`,
  gap: '2px',
  maxWidth: '100%',
  width: '100%',
  overflow: 'hidden',
}));

const HeatmapCell = styled('div')<{ correlation: number }>(({ correlation, theme }) => {
  const getColor = (value: number) => {
    if (value > 0.7) return '#d32f2f'; // Strong positive - Red
    if (value > 0.3) return '#ff9800'; // Moderate positive - Orange  
    if (value > -0.3) return '#4caf50'; // Weak correlation - Green
    if (value > -0.7) return '#2196f3'; // Moderate negative - Blue
    return '#9c27b0'; // Strong negative - Purple
  };

  return {
    backgroundColor: getColor(correlation),
    color: 'white',
    padding: '8px 4px',
    textAlign: 'center',
    fontSize: '0.75rem',
    fontWeight: 'bold',
    borderRadius: '4px',
    cursor: 'pointer',
    transition: 'transform 0.2s',
    '&:hover': {
      transform: 'scale(1.1)',
      zIndex: 1,
    },
  };
});

const SymbolLabel = styled(Typography)(({ theme }) => ({
  fontSize: '0.7rem',
  fontWeight: 'bold',
  padding: '4px',
  textAlign: 'center',
  backgroundColor: theme.palette.grey[800],
  color: 'white',
  borderRadius: '4px',
}));

const LegendContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  marginTop: theme.spacing(2),
  flexWrap: 'wrap',
  justifyContent: 'center',
}));

const LegendItem = styled(Box)<{ color: string }>(({ color }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: '4px',
  '&::before': {
    content: '""',
    width: '16px',
    height: '16px',
    backgroundColor: color,
    borderRadius: '2px',
  },
}));

const CorrelationHeatmap: React.FC = () => {
  const [correlationData, setCorrelationData] = useState<CorrelationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [period, setPeriod] = useState<number>(90);
  const [error, setError] = useState<string>('');

  const fetchCorrelationData = async (days: number) => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.get(`${API_BASE_URL}/api/correlation?days=${days}`);
      setCorrelationData(response.data);
    } catch (err) {
      setError('Failed to fetch correlation data');
      console.error('Correlation fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCorrelationData(period);
  }, [period]);

  const handlePeriodChange = (event: SelectChangeEvent<number>) => {
    setPeriod(event.target.value as number);
  };

  const getCorrelationDescription = (value: number): string => {
    if (value > 0.7) return 'Strong Positive';
    if (value > 0.3) return 'Moderate Positive';
    if (value > -0.3) return 'Weak Correlation';
    if (value > -0.7) return 'Moderate Negative';
    return 'Strong Negative';
  };

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="subtitle2" fontWeight="bold">
          Stock Correlation Analysis
        </Typography>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Period</InputLabel>
          <Select
            value={period}
            label="Period"
            onChange={handlePeriodChange}
          >
            <MenuItem value={30}>30 Days</MenuItem>
            <MenuItem value={60}>60 Days</MenuItem>
            <MenuItem value={90}>90 Days</MenuItem>
            <MenuItem value={180}>180 Days</MenuItem>
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

        {correlationData && !loading && (
          <HeatmapContainer>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              Correlation Matrix - {correlationData.period}
            </Typography>
            
            <HeatmapGrid size={correlationData.symbols.length + 1}>
              {/* Empty top-left cell */}
              <div></div>
              
              {/* Column headers */}
              {correlationData.symbols.map((symbol) => (
                <SymbolLabel key={`col-${symbol}`}>
                  {symbol}
                </SymbolLabel>
              ))}
              
              {/* Matrix rows */}
              {correlationData.symbols.map((rowSymbol, i) => (
                <React.Fragment key={`row-${rowSymbol}`}>
                  {/* Row header */}
                  <SymbolLabel>{rowSymbol}</SymbolLabel>
                  
                  {/* Correlation cells */}
                  {correlationData.matrix[i].map((correlation, j) => (
                    <Tooltip
                      key={`cell-${i}-${j}`}
                      title={`${rowSymbol} vs ${correlationData.symbols[j]}: ${correlation.toFixed(3)} (${getCorrelationDescription(correlation)})`}
                      arrow
                    >
                      <HeatmapCell correlation={correlation}>
                        {correlation.toFixed(2)}
                      </HeatmapCell>
                    </Tooltip>
                  ))}
                </React.Fragment>
              ))}
            </HeatmapGrid>

            {/* Legend */}
            <LegendContainer>
              <Typography variant="body2" fontWeight="bold">
                Correlation Strength:
              </Typography>
              <LegendItem color="#d32f2f">
                <Typography variant="caption">Strong Positive (0.7+)</Typography>
              </LegendItem>
              <LegendItem color="#ff9800">
                <Typography variant="caption">Moderate Positive (0.3-0.7)</Typography>
              </LegendItem>
              <LegendItem color="#4caf50">
                <Typography variant="caption">Weak (-0.3 to 0.3)</Typography>
              </LegendItem>
              <LegendItem color="#2196f3">
                <Typography variant="caption">Moderate Negative (-0.7 to -0.3)</Typography>
              </LegendItem>
              <LegendItem color="#9c27b0">
                <Typography variant="caption">Strong Negative (-0.7-)</Typography>
              </LegendItem>
            </LegendContainer>
          </HeatmapContainer>
        )}
    </Box>
  );
};

export default CorrelationHeatmap;
