import React from 'react';
import {
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Typography,
  CircularProgress,
} from '@mui/material';
import { StockInfo } from '../types';

interface StockSelectorProps {
  stocks: StockInfo[];
  selectedStock: string;
  onStockChange: (symbol: string) => void;
  loading: boolean;
}

const StockSelector: React.FC<StockSelectorProps> = ({
  stocks,
  selectedStock,
  onStockChange,
  loading,
}) => {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Select Stock for Analysis
      </Typography>
      <FormControl fullWidth disabled={loading}>
        <InputLabel>Stock Symbol</InputLabel>
        <Select
          value={selectedStock}
          label="Stock Symbol"
          onChange={(e) => onStockChange(e.target.value)}
        >
          {stocks.map((stock) => (
            <MenuItem key={stock.symbol} value={stock.symbol}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                <span>{stock.symbol}</span>
                <span style={{ color: '#666', fontSize: '0.9em' }}>
                  ₹{stock.last_price.toFixed(2)}
                </span>
              </Box>
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
          <CircularProgress size={24} />
        </Box>
      )}
    </Box>
  );
};

export default StockSelector;
