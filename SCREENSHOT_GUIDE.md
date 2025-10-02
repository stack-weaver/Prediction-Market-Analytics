# 📸 Screenshot Guide for IntelliStock Pro

## 🎯 Required Screenshots

To complete the README, please take the following screenshots of your running application:

### 1. **Main Dashboard** (`images/dashboard-overview.png`)
**What to capture:**
- Open http://localhost:3000
- Select a stock (e.g., RELIANCE)
- Show the Overview tab with:
  - Stock selector at top
  - Key metrics cards
  - Price chart
  - Market overview section
  - Dataset information

**Recommended size:** 1920x1080 (Full HD)
**Format:** PNG

---

### 2. **Prediction Results** (`images/prediction-results.png`)
**What to capture:**
- Navigate to "Predictions" tab
- Select a stock with trained models
- Show:
  - Current price vs predicted price
  - 5-day forecast
  - Confidence score
  - All 6 model predictions (LSTM, Random Forest, XGBoost, ARIMA, Prophet, Transformer)
  - Individual model forecasts

**Focus:** Make sure all 6 models are visible

---

### 3. **Advanced Charts** (`images/advanced-charts.png`)
**What to capture:**
- Navigate to "Charts" tab
- Show the Enhanced Candlestick Chart with:
  - Candlestick pattern
  - Volume bars
  - Moving averages
  - Bollinger Bands
  - Prediction overlay
  - Technical indicators panel

**Tip:** Choose a stock with interesting price movements

---

### 4. **Technical Analysis** (`images/technical-analysis.png`)
**What to capture:**
- Navigate to "Analysis" tab
- Show:
  - Technical indicators (RSI, MACD, etc.)
  - Risk metrics (Volatility, Sharpe Ratio, etc.)
  - Model performance comparison
  - Correlation heatmap

**Focus:** Ensure all metrics are visible and populated

---

### 5. **Sentiment Analysis** (`images/sentiment-analysis.png`)
**What to capture:**
- Navigate to "Sentiment" tab
- Show:
  - Market sentiment summary
  - Sentiment distribution chart
  - Recent news articles with sentiment scores
  - Time range selector
  - Sentiment trends

**Note:** Make sure news data is loaded

---

### 6. **Portfolio Dashboard** (`images/portfolio-dashboard.png`)
**What to capture:**
- Navigate to "Portfolio" tab
- Show:
  - Portfolio allocation chart
  - Performance metrics
  - Risk assessment
  - Value distribution
  - Recommended allocation

**Tip:** The default portfolio should show sample data

---

## 📝 Screenshot Tips

### **Best Practices:**
1. **Full Screen**: Use full browser window (F11)
2. **Clean UI**: Hide browser bookmarks/extensions
3. **Good Data**: Use stocks with interesting patterns
4. **High Quality**: PNG format, high resolution
5. **Consistent**: Same browser, same zoom level

### **Recommended Tools:**
- **Windows**: Snipping Tool or Win + Shift + S
- **Mac**: Cmd + Shift + 4
- **Browser**: Developer tools screenshot feature

### **Image Optimization:**
```bash
# After taking screenshots, you can optimize them:
# 1. Resize to reasonable dimensions (1200px wide max)
# 2. Compress to reduce file size
# 3. Ensure good visibility of text and charts
```

---

## 🚀 After Taking Screenshots

1. **Save images** in the `images/` folder with exact names:
   - `dashboard-overview.png`
   - `prediction-results.png`
   - `advanced-charts.png`
   - `technical-analysis.png`
   - `sentiment-analysis.png`
   - `portfolio-dashboard.png`

2. **Add to git:**
   ```bash
   git add images/
   git commit -m "📸 Add project screenshots"
   git push
   ```

3. **Verify**: Check that images display correctly in GitHub README

---

## 🎨 Alternative: Create Demo GIFs

For even better presentation, consider creating animated GIFs showing:
- Navigation between tabs
- Stock selection process
- Real-time updates
- Interactive chart features

**Tools for GIFs:**
- **ScreenToGif** (Windows)
- **LICEcap** (Cross-platform)
- **Kap** (Mac)

---

**Happy screenshotting! 📸✨**
