# IntelliStock Pro: AI-Powered Stock Prediction & Analytics Platform

A professional machine learning system for predicting Indian stock prices using multiple algorithms and technical analysis.

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

## 📊 Features

- **LSTM Neural Networks** - Deep learning for time series
- **Random Forest** - Ensemble learning
- **XGBoost** - Gradient boosting
- **ARIMA** - Statistical forecasting
- **Prophet** - Facebook's forecasting tool
- **Transformer** - Attention-based neural networks

## 🛠️ Tech Stack

- **Python 3.10** + **PyTorch** + **scikit-learn**
- **pandas** + **numpy** + **matplotlib**
- **statsmodels** + **prophet** + **ta-lib**

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

## 📈 Dataset

Uses real NSE dataset (2022-2024) with 50+ stocks including:
- RELIANCE, TCS, HDFCBANK, INFY, HINDUNILVR
- ICICIBANK, KOTAKBANK, ITC, SBIN, BHARTIARTL
- And many more...

## 🔧 Configuration

All configuration is handled through dataclasses in the scripts for easy customization.

## 📝 Notes

- No API dependencies - runs directly with Python scripts
- GPU acceleration supported for LSTM and XGBoost
- Professional error handling and logging
- Caching for improved performance