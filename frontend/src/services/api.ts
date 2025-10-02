import axios from 'axios';
import { StockInfo, PredictionResponse, AnalysisResponse, ChartData } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8002';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

export const getStocks = async (): Promise<StockInfo[]> => {
  try {
    const response = await api.get('/api/stocks');
    return response.data;
  } catch (error) {
    console.error('Error fetching stocks:', error);
    throw error;
  }
};

export const getPrediction = async (symbol: string, daysAhead: number = 5): Promise<PredictionResponse> => {
  try {
    const response = await api.post('/api/predict', {
      symbol,
      days_ahead: daysAhead,
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching prediction:', error);
    throw error;
  }
};

export const getAnalysis = async (symbol: string): Promise<AnalysisResponse> => {
  try {
    const response = await api.get(`/api/analysis/${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching analysis:', error);
    throw error;
  }
};

export const getChartData = async (symbol: string, days: number = 30): Promise<ChartData> => {
  try {
    const response = await api.get(`/api/chart/${symbol}?days=${days}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching chart data:', error);
    throw error;
  }
};
