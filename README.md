# IntelliStock Pro: AI-Powered Stock Prediction & Analytics Platform

A comprehensive full-stack machine learning platform that combines **6 advanced AI models** (LSTM Neural Networks, Random Forest, XGBoost, ARIMA, Prophet, Transformer) to predict NSE stock prices with **ensemble forecasting**. Built with **React TypeScript frontend** and **FastAPI Python backend**, featuring real-time technical analysis using **15+ indicators** (RSI, MACD, Bollinger Bands), **news sentiment analysis**, **portfolio optimization**, and **interactive 3D visualizations**. Trained on **3 years of NSE historical data** (2022-2024) covering **50+ Indian stocks** with both daily and minute-level OHLCV data for robust predictions.

## 📸 Project Screenshots

### 🎯 Main Dashboard
![Main Dashboard](images/dashboard-overview.png?v=2)
*Complete overview with stock selection, predictions, and real-time analytics*

### 📊 Prediction Results
![Prediction Results](images/prediction-results.png?v=2)
*AI predictions from 6 different models with confidence scores and forecasts*

### 📈 Advanced Charts
![Advanced Charts](images/advanced-charts.png?v=2)
*Interactive candlestick charts with technical indicators and trading signals*

### 🔍 Technical Analysis
![Technical Analysis](images/technical-analysis.png?v=2)
*Comprehensive technical analysis with RSI, MACD, Bollinger Bands, and more*

### 📰 Sentiment Analysis
![Sentiment Analysis](images/sentiment-analysis.png?v=2)
*AI-powered sentiment analysis from financial news sources*

### 💼 Portfolio Dashboard
![Portfolio Dashboard](images/portfolio-dashboard.png?v=2)
*Portfolio optimization and risk assessment tools*

---

## 🌟 Live Demo

### 🎮 **Try It Yourself:**
1. Follow the [Quick Start](#-quick-start) guide
2. Download the dataset from Kaggle
3. Train the models (30-60 minutes)
4. Explore all features in the web interface

### 🎯 **What You'll Experience:**
- 🔮 **AI Predictions** from 6 different models
- 📊 **Interactive Charts** with technical indicators  
- 📰 **News Sentiment** impact analysis
- 💼 **Portfolio Optimization** tools
- 📱 **Responsive Design** on any device

## 👨‍💻 Developer

**Himanshu Salunke** - Data Science & AI Enthusiast  
🌐 [GitHub](https://github.com/HimanshuSalunke) | 💼 [LinkedIn](https://www.linkedin.com/in/himanshuksalunke/)

*Passionate about building real-world AI/ML projects and turning data into insights.*

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install
cd frontend && npm install && cd ..
```

### 2. Start the Complete Application
```bash
# Start both backend and frontend servers
npm start
```
This will start:
- 📡 **Backend API**: http://localhost:8002 (or next available port)
- 🎨 **Frontend App**: http://localhost:3000
- 📖 **API Docs**: http://localhost:8002/docs

### 3. Alternative: Individual Commands
```bash
# Backend only
python api_server.py

# Frontend only (in separate terminal)
cd frontend && npm start

# CLI Tools
python main.py train    # Train models
python main.py predict  # Interactive predictions
python main.py analyze  # Stock analysis
python main.py test     # Model testing
```

## 🚀 Key Features

### 🤖 **AI/ML Models Architecture**
| Model | Type | Architecture | Training Config |
|-------|------|-------------|-----------------|
| 🧠 **LSTM** | Deep Learning | 2-layer LSTM (128→64 units) + Dropout(0.2) | 50 epochs, Adam optimizer, MSE loss |
| 🌳 **Random Forest** | Ensemble | 100 estimators, max_depth=10 | Bootstrap sampling, feature importance |
| ⚡ **XGBoost** | Gradient Boosting | learning_rate=0.1, max_depth=6 | GPU acceleration, early stopping |
| 📈 **ARIMA** | Statistical | Auto ARIMA (p,d,q) selection | AIC optimization, seasonal decomposition |
| 🔮 **Prophet** | Facebook AI | Additive model with trends | Weekly/yearly seasonality, holidays |
| 🎯 **Transformer** | Attention-based | Multi-head attention, 4 layers | Self-attention mechanism, positional encoding |

### 🔧 **Model Configuration Details**
#### **LSTM Neural Network**
- **Architecture**: Input(38 features) → LSTM(128) → Dropout(0.2) → LSTM(64) → Dense(32) → Output(5 predictions)
- **Training**: 50 epochs, batch_size=32, learning_rate=0.001
- **Optimization**: Adam optimizer with gradient clipping
- **Regularization**: Dropout layers, early stopping (patience=10)

#### **Ensemble Methodology**
- **Prediction Combination**: Weighted averaging based on historical performance
- **Confidence Scoring**: Standard deviation across model predictions
- **Validation**: Walk-forward analysis with 80/20 train-test split
- **Performance Metrics**: RMSE, MAPE, R², Sharpe ratio

### 📊 **Analytics Dashboard**
- 📈 **Enhanced Candlestick Charts** - Interactive TradingView-style charts with volume bars, moving averages (SMA, EMA), Bollinger Bands, and prediction overlays
- 🎯 **Ensemble Predictions** - Combines predictions from all 6 models using weighted averaging with confidence scoring and standard deviation bands
- 📰 **News Sentiment Engine** - Scrapes financial news from Economic Times, LiveMint, MoneyControl using NLP sentiment analysis to predict market impact
- 💼 **Portfolio Optimizer** - Modern Portfolio Theory implementation with risk-return optimization, Sharpe ratio calculation, and efficient frontier plotting
- 🔍 **Technical Indicators Suite** - RSI, MACD, Bollinger Bands, Stochastic, Williams %R, CCI, Ichimoku Cloud, PSAR, ATR, MFI, Ultimate Oscillator
- 📱 **Material-UI Interface** - Responsive React components with dark/light themes, drawer navigation, and real-time data updates

### ⚡ **Performance & Architecture**
- 🚀 **GPU Acceleration** - CUDA-enabled PyTorch training for LSTM and Transformer models with automatic CPU fallback
- 💾 **Intelligent Caching** - LRU cache for predictions, model persistence with pickle serialization, and Redis-ready architecture
- 🎨 **Modern Tech Stack** - Vite build system, TypeScript type safety, Material-UI v5 components, and Three.js 3D visualizations
- 📊 **Concurrent Processing** - FastAPI async endpoints, WebSocket real-time updates, and parallel model training
- 🔒 **Production Ready** - Comprehensive error boundaries, logging with Python logging module, input validation, and graceful degradation

## 🛠️ Tech Stack & Dependencies

### **🐍 Backend (Python 3.10.0)**
#### **Core ML & Data Science**
- 🔥 **PyTorch 1.13.1+cu117** - Deep learning with CUDA support
- 🤖 **scikit-learn 1.3.2** - Machine learning algorithms
- 📊 **pandas 2.3.2** - Data manipulation and analysis
- 🔢 **numpy 1.24.4** - Numerical computing
- 📉 **scipy 1.11.4** - Scientific computing

#### **Time Series & Financial Analysis**
- 📈 **statsmodels 0.14.0** - Statistical modeling (ARIMA)
- 🔮 **Prophet 1.1.4** - Facebook's time series forecasting
- 📊 **TA-Lib 0.6.7** - Technical analysis indicators
- 🏦 **arch 6.2.0** - Financial econometrics

#### **Web Framework & API**
- ⚡ **FastAPI 0.104.1** - Modern async web framework
- 🚀 **uvicorn 0.24.0** - ASGI server
- 📋 **Pydantic 2.5.0** - Data validation
- 🌐 **yfinance** - Current price fetching

#### **Visualization & Analysis**
- 📈 **matplotlib 3.7.2** - Plotting library
- 🎨 **seaborn 0.12.2** - Statistical visualization
- 📊 **plotly 5.15.0** - Interactive charts

#### **Portfolio Optimization**
- 🔢 **cvxpy 1.3.2** - Convex optimization
- 💼 **PyPortfolioOpt 1.5.5** - Modern Portfolio Theory

#### **Model Explainability & Tuning**
- 🎯 **Optuna 3.3.0** - Hyperparameter optimization
- 🔍 **SHAP 0.42.1** - Model explainability
- 🧪 **LIME 0.2.0.1** - Local interpretable explanations

#### **Utilities & Development**
- 🔧 **joblib 1.3.2** - Model persistence
- 📊 **tqdm 4.65.0** - Progress bars
- 🧪 **pytest 7.4.0** - Testing framework
- 🎨 **black 23.7.0** - Code formatting
- 📝 **flake8 6.0.0** - Code linting

### **⚛️ Frontend (Node.js 22.18.0)**
#### **Core Framework**
- ⚛️ **React 19.1.1** - UI library with latest features
- 📘 **TypeScript 5.3.0** - Type-safe JavaScript
- ⚡ **Vite 5.0.0** - Next-generation build tool
- 🎯 **@vitejs/plugin-react 4.2.1** - React integration

#### **UI Components & Styling**
- 🎨 **Material-UI 5.18.0** - React component library
- 💫 **@emotion/react 11.14.0** - CSS-in-JS styling
- 💫 **@emotion/styled 11.14.1** - Styled components
- 🎭 **@mui/icons-material 5.18.0** - Material Design icons

#### **Data Visualization**
- 📊 **Recharts 3.2.1** - React charting library
- 🎮 **Three.js 0.180.0** - 3D graphics and visualization
- 📈 **Interactive charts** - Custom candlestick implementations

#### **HTTP & State Management**
- 🌐 **Axios 1.12.2** - HTTP client
- 🔄 **React Hooks** - State management
- 📡 **WebSocket support** - Real-time updates

#### **Development & Testing**
- 🧪 **Vitest 1.2.0** - Unit testing framework
- 🧪 **@testing-library/react 16.3.0** - Component testing
- 📝 **ESLint 8.56.0** - Code linting
- 📝 **@typescript-eslint 6.19.0** - TypeScript linting

#### **Build Optimization**
- 🚀 **ESBuild** - Fast bundling
- 📦 **Code splitting** - Vendor chunks (React, MUI, Three.js, Charts)
- 🗜️ **Tree shaking** - Dead code elimination
- 📊 **Source maps** - Development debugging

### **🏗️ Architecture & Configuration**
#### **Build System**
- 📦 **Vite Config** - Optimized for ML dashboard
- 🎯 **Path aliases** - @components, @services, @types
- 🔄 **Proxy setup** - API calls to backend (localhost:8002)
- 🌐 **CORS enabled** - Cross-origin requests

#### **TypeScript Configuration**
- 🎯 **Target: ES2020** - Modern JavaScript features
- 📦 **Module: ESNext** - Latest module system
- 🔍 **Strict mode** - Enhanced type checking
- 📁 **Path mapping** - Absolute imports

#### **Development Tools**
- 🔄 **Concurrently 8.2.2** - Run backend + frontend
- 🌍 **Cross-env 7.0.3** - Cross-platform environment variables
- 🎨 **Hot reload** - Development server
- 📊 **Bundle analysis** - Performance monitoring

## 📁 Project Structure

```
├── main.py                 # Main entry point
├── scripts/               # Core scripts
│   ├── training.py        # Model training
│   ├── prediction.py      # Predictions
│   ├── analysis.py        # Analysis
│   └── testing.py         # Model testing
├── src/                   # Source code
│   ├── data/              # Data management
│   ├── features/          # Feature engineering
│   └── models/             # ML models
└── data/                  # Dataset and models
    ├── raw/dataset/       # NSE dataset (2022-2024)
    └── models/            # Trained models
```

## 🎯 Usage Examples

```bash
# Train specific models for specific stocks
python main.py train --stocks RELIANCE TCS --models lstm xgboost

# Interactive prediction demo
python main.py predict

# Analyze specific stock
python main.py analyze --stock RELIANCE

# Test model performance
python main.py test
```

## 📈 Dataset Information

### **NSE Historical Market Dataset (2022-2024)**
- **Source**: [Kaggle - NSE Nifty50 Index Daily & Minute Level Data](https://www.kaggle.com/datasets/tomtillo/nse-nifty50-index-daily-minute-level-data)
- **Time Period**: January 2022 - December 2024 (1,095 days of market data)
- **Data Granularity**: Both daily OHLCV and minute-level tick data for intraday analysis
- **Market Coverage**: 50+ NSE blue-chip and mid-cap stocks representing 65% of Indian market capitalization
- **Data Volume**: ~2.5 million data points across all timeframes and stocks
- **Quality Assurance**: Pre-cleaned data with corporate action adjustments and split/bonus corrections

### **Available Stocks:**
RELIANCE, TCS, HDFCBANK, INFY, HINDUNILVR, ICICIBANK, KOTAKBANK, ITC, SBIN, BHARTIARTL, MARUTI, BAJAJ-AUTO, TATAMOTORS, HEROMOTOCO, EICHERMOT, HCLTECH, WIPRO, TECHM, ULTRACEMCO, ASIANPAINT, GRASIM, JSWSTEEL, TATASTEEL, HINDALCO, NTPC, POWERGRID, COALINDIA, ONGC, BPCL, IOC, GAIL, ADANIENT, ADANIPORTS, APOLLOHOSP, CIPLA, DRREDDY, DIVISLAB, SUNPHARMA, NESTLEIND, BRITANNIA, TATACONSUM, TITAN, BAJFINANCE, BAJAJFINSV, SBILIFE, HDFCLIFE, LTIM, UPL, INDUSINDBK, AXISBANK

### **Feature Engineering Pipeline (38 Features)**
#### **Raw Market Data (5 features)**
- **OHLCV**: Open, High, Low, Close, Volume with corporate action adjustments

#### **Technical Indicators (15+ indicators)**
| Indicator | Parameters | Purpose |
|-----------|------------|---------|
| **RSI** | Period=14 | Momentum oscillator (0-100) |
| **MACD** | Fast=12, Slow=26, Signal=9 | Trend following momentum |
| **Bollinger Bands** | Period=20, StdDev=2 | Volatility and mean reversion |
| **Stochastic** | %K=14, %D=3 | Momentum oscillator |
| **Williams %R** | Period=14 | Momentum indicator |
| **CCI** | Period=20 | Commodity Channel Index |
| **Ichimoku Cloud** | 9,26,52 periods | Trend and support/resistance |
| **PSAR** | AF=0.02, Max=0.2 | Parabolic Stop and Reverse |
| **ATR** | Period=14 | Average True Range (volatility) |
| **MFI** | Period=14 | Money Flow Index |
| **Ultimate Oscillator** | 7,14,28 periods | Multi-timeframe momentum |

#### **Moving Averages (10 features)**
- **SMA**: 5, 10, 20, 50, 200-day Simple Moving Averages
- **EMA**: 5, 10, 20, 50, 200-day Exponential Moving Averages

#### **Volume Analysis (3 features)**
- **OBV**: On-Balance Volume for trend confirmation
- **Volume Rate of Change**: Volume momentum
- **Volume Moving Average**: 20-day volume trend

#### **Lag Features (5 features)**
- **Price Lags**: 1, 2, 3, 4, 5-day historical prices for sequence learning

#### **Derived Features (5 features)**
- **Price Change %**: Daily return percentage
- **Volatility**: 20-day rolling standard deviation
- **High-Low Spread**: Daily trading range
- **Close Position**: (Close - Low) / (High - Low)
- **Volume Ratio**: Current volume / 20-day average volume

#### **Target Variables (Multi-horizon forecasting)**
- **Next 1-day**: T+1 price prediction
- **Next 3-day**: T+3 price prediction  
- **Next 5-day**: T+5 price prediction
- **Direction**: Bullish/Bearish classification
- **Volatility**: Expected price volatility

## 🔧 System Configuration & Requirements

### **💻 System Requirements**
#### **Minimum Requirements**
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.10.0+ (tested on 3.10.0)
- **Node.js**: 18.0+ (tested on 22.18.0)
- **RAM**: 8GB (16GB recommended for full dataset)
- **Storage**: 10GB free space
- **GPU**: Optional (CUDA 11.7+ for GPU acceleration)

#### **Recommended Specifications**
- **CPU**: Intel i5/AMD Ryzen 5 or better
- **RAM**: 16GB+ for optimal performance
- **GPU**: NVIDIA GTX 1060+ with 6GB VRAM
- **SSD**: For faster data loading and model training

### **🔧 Configuration Files**
#### **Backend Configuration (`config/settings.py`)**
```python
# Model Training Parameters
LOOKBACK_DAYS = 60          # Historical data window
PREDICTION_DAYS = 5         # Forecast horizon
TRAIN_TEST_SPLIT = 0.8      # 80% training, 20% testing
BATCH_SIZE = 32             # Neural network batch size
EPOCHS = 50                 # Training epochs
LEARNING_RATE = 0.001       # Adam optimizer learning rate

# Technical Indicators
RSI_PERIOD = 14             # RSI calculation period
MACD_FAST = 12             # MACD fast EMA
MACD_SLOW = 26             # MACD slow EMA
MACD_SIGNAL = 9            # MACD signal line
BOLLINGER_PERIOD = 20      # Bollinger Bands period
BOLLINGER_STD = 2          # Standard deviation multiplier

# Risk Management
VAR_CONFIDENCE = 0.05      # 5% Value at Risk
SHARPE_RISK_FREE_RATE = 0.06  # 6% Indian risk-free rate
MAX_PORTFOLIO_RISK = 0.15  # 15% maximum portfolio risk
```

#### **Frontend Configuration (`vite.config.ts`)**
```typescript
// Development server
server: {
  port: 3000,                    // Frontend port
  proxy: {
    '/api': 'http://localhost:8002'  // Backend proxy
  }
}

// Build optimization
build: {
  target: 'esnext',              // Modern JavaScript
  chunkSizeWarningLimit: 1000,   // ML libraries are large
  rollupOptions: {
    output: {
      manualChunks: {              // Code splitting
        'react-vendor': ['react', 'react-dom'],
        'mui-vendor': ['@mui/material', '@mui/icons-material'],
        'three-vendor': ['three'],
        'charts-vendor': ['recharts']
      }
    }
  }
}
```

### **🌐 API Endpoints**
#### **Core Endpoints**
- `GET /` - API information and health check
- `GET /health` - System health status
- `GET /docs` - Interactive API documentation (Swagger UI)

#### **Stock Data**
- `GET /api/stocks` - List all available stocks
- `GET /api/chart/{symbol}` - Historical price data
- `POST /api/predict` - Generate ML predictions

#### **Analysis**
- `GET /api/analysis/{symbol}` - Technical analysis
- `GET /api/correlation` - Stock correlation matrix
- `GET /api/news/sentiment` - News sentiment analysis

#### **Portfolio**
- `GET /api/portfolio/allocation` - Portfolio optimization
- `GET /api/portfolio/risk` - Risk assessment

#### **WebSocket**
- `WS /ws` - Real-time data updates (planned feature)

## ⚠️ Important Notes

### **First Time Setup:**
1. **Dataset Required**: Download from Kaggle before running
2. **Model Training**: Required on first run (30-60 minutes)
3. **GPU Recommended**: For faster training (optional)
4. **Memory**: 8GB+ RAM recommended for full dataset

### **Model Training Time Estimates:**
- **LSTM**: ~10-15 minutes per stock
- **Random Forest**: ~2-3 minutes per stock  
- **XGBoost**: ~3-5 minutes per stock
- **ARIMA**: ~1-2 minutes per stock
- **Prophet**: ~2-3 minutes per stock
- **Transformer**: ~15-20 minutes per stock

## 🚀 Deployment

### **Free Deployment Options:**
1. **Frontend**: Deploy to Vercel/Netlify (free)
2. **Backend**: Deploy to Railway/Render (free tier)
3. **Models**: Train on deployment or use lightweight versions

### **Production Setup:**
```bash
# For production deployment
pip install gunicorn
gunicorn api_server:app --host 0.0.0.0 --port $PORT
```

## 🛠️ Troubleshooting

### **Common Issues:**

**1. Dataset Not Found**
```bash
# Make sure dataset is in correct location
data/raw/dataset/2022/day/*.csv
data/raw/dataset/2022/minute/*.csv
```

**2. Models Not Trained**
```bash
# Train models first
python main.py train
```

**3. Memory Issues**
```bash
# Train fewer stocks at once
python main.py train --stocks RELIANCE TCS --models lstm random_forest
```

**4. GPU Issues**
```bash
# Install CPU-only PyTorch if GPU issues
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 📝 Additional Notes

- **No External APIs**: Runs completely offline after dataset download
- **GPU Acceleration**: Supported for LSTM and Transformer models
- **Professional Logging**: Comprehensive error handling and monitoring
- **Caching**: Intelligent caching for improved performance
- **Scalable**: Easy to add new models and features