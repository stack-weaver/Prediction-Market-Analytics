# IntelliStock Pro: AI-Powered Stock Prediction & Analytics Platform

A professional machine learning system for predicting Indian stock prices using multiple algorithms and technical analysis.

## 📸 Project Screenshots

### 🎯 Main Dashboard
![Main Dashboard](images/dashboard-overview.png)
*Complete overview with stock selection, predictions, and real-time analytics*

### 📊 Prediction Results
![Prediction Results](images/prediction-results.png)
*AI predictions from 6 different models with confidence scores and forecasts*

### 📈 Advanced Charts
![Advanced Charts](images/advanced-charts.png)
*Interactive candlestick charts with technical indicators and trading signals*

### 🔍 Technical Analysis
![Technical Analysis](images/technical-analysis.png)
*Comprehensive technical analysis with RSI, MACD, Bollinger Bands, and more*

### 📰 Sentiment Analysis
![Sentiment Analysis](images/sentiment-analysis.png)
*Real-time news sentiment analysis affecting stock prices*

### 💼 Portfolio Dashboard
![Portfolio Dashboard](images/portfolio-dashboard.png)
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

### 🤖 **AI/ML Models**
| Model | Type | Strength |
|-------|------|----------|
| 🧠 **LSTM** | Deep Learning | Sequential pattern recognition |
| 🌳 **Random Forest** | Ensemble | Robust predictions |
| ⚡ **XGBoost** | Gradient Boosting | High performance |
| 📈 **ARIMA** | Statistical | Time series analysis |
| 🔮 **Prophet** | Facebook AI | Trend decomposition |
| 🎯 **Transformer** | Attention-based | Complex relationships |

### 📊 **Analytics Dashboard**
- 📈 **Interactive Charts** - Candlestick, volume, technical indicators
- 🎯 **Real-time Predictions** - Multi-model ensemble forecasting
- 📰 **Sentiment Analysis** - News impact on stock prices
- 💼 **Portfolio Optimization** - Risk assessment and allocation
- 🔍 **Technical Analysis** - 15+ indicators (RSI, MACD, Bollinger Bands)
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile

### ⚡ **Performance Features**
- 🚀 **Fast Training** - Optimized algorithms
- 💾 **Smart Caching** - Reduced computation time
- 🎨 **Modern UI** - Material Design components
- 📊 **Real-time Updates** - Live data processing
- 🔒 **Error Handling** - Robust error management

## 🛠️ Tech Stack

### **Backend**
- 🐍 **Python 3.10** - Core language
- 🔥 **PyTorch** - Deep learning framework
- 🤖 **scikit-learn** - Machine learning library
- 📊 **pandas** - Data manipulation
- 🔢 **numpy** - Numerical computing
- 📈 **matplotlib/seaborn** - Data visualization
- 📉 **statsmodels** - Statistical analysis
- 🔮 **Prophet** - Time series forecasting
- 📊 **TA-Lib** - Technical analysis
- ⚡ **FastAPI** - Modern web framework
- 🚀 **uvicorn** - ASGI server

### **Frontend**
- ⚛️ **React 18** - UI framework
- 📘 **TypeScript** - Type safety
- 🎨 **Material-UI** - Component library
- 📊 **Recharts** - Chart library
- 🎮 **Three.js** - 3D visualization
- ⚡ **Vite** - Build tool
- 🎯 **Axios** - HTTP client

### **Data & ML**
- 📈 **NSE Dataset** - Indian stock market data
- 🧠 **6 ML Models** - Ensemble predictions
- 📊 **15+ Indicators** - Technical analysis
- 📰 **News API** - Sentiment analysis
- 💼 **Portfolio Tools** - Risk assessment

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

### **NSE Nifty 50 Dataset (2022-2024)**
- **Source**: [Kaggle - NSE Nifty50 Index Daily & Minute Level Data](https://www.kaggle.com/datasets/tomtillo/nse-nifty50-index-daily-minute-level-data)
- **Time Period**: January 2022 - December 2024 (3 Years)
- **Frequency**: Daily & Minute-level OHLCV data
- **Stocks**: 50+ NSE listed companies

### **Available Stocks:**
RELIANCE, TCS, HDFCBANK, INFY, HINDUNILVR, ICICIBANK, KOTAKBANK, ITC, SBIN, BHARTIARTL, MARUTI, BAJAJ-AUTO, TATAMOTORS, HEROMOTOCO, EICHERMOT, HCLTECH, WIPRO, TECHM, ULTRACEMCO, ASIANPAINT, GRASIM, JSWSTEEL, TATASTEEL, HINDALCO, NTPC, POWERGRID, COALINDIA, ONGC, BPCL, IOC, GAIL, ADANIENT, ADANIPORTS, APOLLOHOSP, CIPLA, DRREDDY, DIVISLAB, SUNPHARMA, NESTLEIND, BRITANNIA, TATACONSUM, TITAN, BAJFINANCE, BAJAJFINSV, SBILIFE, HDFCLIFE, LTIM, UPL, INDUSINDBK, AXISBANK

### **Data Features:**
- **OHLCV Data**: Open, High, Low, Close, Volume
- **Technical Indicators**: 15+ indicators (RSI, MACD, Bollinger Bands, etc.)
- **Time Series**: Sequential data for ML training
- **Quality**: Clean, validated market data

## 🔧 Configuration

All configuration is handled through dataclasses in the scripts for easy customization.

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