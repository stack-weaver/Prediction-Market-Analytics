export interface StockInfo {
  symbol: string;
  name: string;
  sector: string;
  last_price: number;
  models_available: string[];
}

export interface PredictionResponse {
  symbol: string;
  current_price: number;
  predictions: Record<string, number[]>;
  ensemble_prediction: number[];
  confidence_score: number;
  trend: string;
}

export interface AnalysisResponse {
  symbol: string;
  current_price: number;
  model_performance: Record<string, Record<string, number>>;
  risk_metrics: Record<string, number>;
  technical_indicators: Record<string, number>;
}

export interface ChartData {
  dates: string[];
  prices: number[];
  volumes: number[];
  high: number[];
  low: number[];
  open: number[];
}
