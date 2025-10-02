"""
News Sentiment Analysis Module
Integrates real-time news data and sentiment analysis for stock prediction enhancement
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import re
import time
from textblob import TextBlob
import yfinance as yf
from bs4 import BeautifulSoup
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    """
    Comprehensive news sentiment analysis for stock market prediction
    """
    
    def __init__(self):
        self.news_sources = {
            'economic_times': 'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
            'business_standard': 'https://www.business-standard.com/rss/markets-106.rss',
            'moneycontrol': 'https://www.moneycontrol.com/rss/business.xml',
            'livemint': 'https://www.livemint.com/rss/markets',
            'financial_express': 'https://www.financialexpress.com/market/rss'
        }
        
        # Stock-related keywords for filtering
        self.stock_keywords = [
            'stock', 'share', 'equity', 'market', 'trading', 'investment',
            'earnings', 'revenue', 'profit', 'loss', 'dividend', 'ipo',
            'merger', 'acquisition', 'quarterly', 'results', 'nse', 'bse'
        ]
        
        # Sentiment keywords
        self.positive_keywords = [
            'growth', 'profit', 'gain', 'rise', 'surge', 'bullish', 'positive',
            'strong', 'beat', 'exceed', 'outperform', 'upgrade', 'buy'
        ]
        
        self.negative_keywords = [
            'loss', 'decline', 'fall', 'drop', 'bearish', 'negative', 'weak',
            'miss', 'underperform', 'downgrade', 'sell', 'concern', 'risk'
        ]
    
    def fetch_news_from_rss(self, source_name: str, url: str, max_articles: int = 50) -> List[Dict]:
        """Fetch news articles from RSS feed"""
        try:
            logger.info(f"Fetching news from {source_name}")
            feed = feedparser.parse(url)
            
            articles = []
            for entry in feed.entries[:max_articles]:
                article = {
                    'title': entry.get('title', ''),
                    'description': entry.get('description', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'source': source_name,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Parse published date
                try:
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        article['published_date'] = datetime(*entry.published_parsed[:6])
                    else:
                        article['published_date'] = datetime.now()
                except:
                    article['published_date'] = datetime.now()
                
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from {source_name}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news from {source_name}: {e}")
            return []
    
    def fetch_all_news(self, max_articles_per_source: int = 30) -> List[Dict]:
        """Fetch news from all sources concurrently"""
        all_articles = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_source = {
                executor.submit(self.fetch_news_from_rss, source, url, max_articles_per_source): source
                for source, url in self.news_sources.items()
            }
            
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    articles = future.result(timeout=30)
                    all_articles.extend(articles)
                except Exception as e:
                    logger.error(f"Error processing news from {source}: {e}")
        
        # Sort by published date
        all_articles.sort(key=lambda x: x['published_date'], reverse=True)
        
        logger.info(f"Total articles fetched: {len(all_articles)}")
        return all_articles
    
    def is_stock_related(self, text: str) -> bool:
        """Check if article is stock/market related"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.stock_keywords)
    
    def filter_stock_news(self, articles: List[Dict]) -> List[Dict]:
        """Filter articles to only stock-related news"""
        filtered = []
        
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            combined_text = f"{title} {description}"
            
            if self.is_stock_related(combined_text):
                filtered.append(article)
        
        logger.info(f"Filtered to {len(filtered)} stock-related articles")
        return filtered
    
    def analyze_sentiment_textblob(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Convert to our scale
            sentiment_score = polarity  # Keep -1 to 1 scale
            confidence = 1 - subjectivity  # Higher objectivity = higher confidence
            
            return {
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'polarity': polarity,
                'subjectivity': subjectivity
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'polarity': 0.0,
                'subjectivity': 0.5
            }
    
    def analyze_sentiment_keywords(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using keyword matching"""
        text_lower = text.lower()
        
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        
        total_keywords = positive_count + negative_count
        
        if total_keywords == 0:
            return {'sentiment_score': 0.0, 'confidence': 0.0}
        
        sentiment_score = (positive_count - negative_count) / total_keywords
        confidence = min(total_keywords / 10.0, 1.0)  # Max confidence at 10+ keywords
        
        return {
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'positive_keywords': positive_count,
            'negative_keywords': negative_count
        }
    
    def analyze_article_sentiment(self, article: Dict) -> Dict:
        """Analyze sentiment for a single article"""
        title = article.get('title', '')
        description = article.get('description', '')
        combined_text = f"{title} {description}"
        
        # TextBlob analysis
        textblob_result = self.analyze_sentiment_textblob(combined_text)
        
        # Keyword analysis
        keyword_result = self.analyze_sentiment_keywords(combined_text)
        
        # Combine results (weighted average)
        textblob_weight = 0.7
        keyword_weight = 0.3
        
        final_sentiment = (
            textblob_result['sentiment_score'] * textblob_weight +
            keyword_result['sentiment_score'] * keyword_weight
        )
        
        final_confidence = (
            textblob_result['confidence'] * textblob_weight +
            keyword_result['confidence'] * keyword_weight
        )
        
        # Add sentiment analysis to article
        article['sentiment'] = {
            'score': final_sentiment,
            'confidence': final_confidence,
            'textblob': textblob_result,
            'keywords': keyword_result,
            'classification': self.classify_sentiment(final_sentiment)
        }
        
        return article
    
    def classify_sentiment(self, score: float) -> str:
        """Classify sentiment score into categories"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_news_sentiment(self, hours_back: int = 24) -> Dict[str, Any]:
        """Analyze sentiment for recent news"""
        logger.info(f"Starting news sentiment analysis for last {hours_back} hours")
        
        # Fetch news
        all_articles = self.fetch_all_news()
        
        # Filter to recent articles
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_articles = [
            article for article in all_articles
            if article['published_date'] >= cutoff_time
        ]
        
        # Filter to stock-related news
        stock_articles = self.filter_stock_news(recent_articles)
        
        # Analyze sentiment for each article
        analyzed_articles = []
        for article in stock_articles:
            try:
                analyzed_article = self.analyze_article_sentiment(article)
                analyzed_articles.append(analyzed_article)
            except Exception as e:
                logger.error(f"Error analyzing article sentiment: {e}")
                continue
        
        # Calculate aggregate sentiment metrics
        sentiment_summary = self.calculate_sentiment_summary(analyzed_articles)
        
        return {
            'total_articles': len(all_articles),
            'recent_articles': len(recent_articles),
            'stock_articles': len(stock_articles),
            'analyzed_articles': len(analyzed_articles),
            'articles': analyzed_articles,
            'sentiment_summary': sentiment_summary,
            'analysis_timestamp': datetime.now().isoformat(),
            'hours_analyzed': hours_back
        }
    
    def calculate_sentiment_summary(self, articles: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate sentiment metrics"""
        if not articles:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'sentiment_distribution': {}
            }
        
        sentiments = [article['sentiment']['score'] for article in articles]
        confidences = [article['sentiment']['confidence'] for article in articles]
        classifications = [article['sentiment']['classification'] for article in articles]
        
        # Weighted average sentiment (by confidence)
        if sum(confidences) > 0:
            weighted_sentiment = sum(s * c for s, c in zip(sentiments, confidences)) / sum(confidences)
        else:
            weighted_sentiment = np.mean(sentiments)
        
        # Count classifications
        positive_count = classifications.count('positive')
        negative_count = classifications.count('negative')
        neutral_count = classifications.count('neutral')
        
        # Sentiment distribution
        sentiment_distribution = {
            'positive': positive_count / len(articles),
            'negative': negative_count / len(articles),
            'neutral': neutral_count / len(articles)
        }
        
        return {
            'overall_sentiment': weighted_sentiment,
            'confidence': np.mean(confidences),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'sentiment_distribution': sentiment_distribution,
            'sentiment_std': np.std(sentiments),
            'articles_analyzed': len(articles)
        }
    
    def get_symbol_specific_sentiment(self, symbol: str, articles: List[Dict]) -> Dict[str, Any]:
        """Get sentiment specific to a stock symbol"""
        symbol_articles = []
        
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            
            # Check if symbol is mentioned
            if (symbol.lower() in title or symbol.lower() in description or
                any(keyword in title or keyword in description 
                    for keyword in [symbol.lower(), f"{symbol.lower()} stock", f"{symbol.lower()} share"])):
                symbol_articles.append(article)
        
        if not symbol_articles:
            return {
                'symbol': symbol,
                'articles_found': 0,
                'sentiment_score': 0.0,
                'confidence': 0.0
            }
        
        sentiment_summary = self.calculate_sentiment_summary(symbol_articles)
        sentiment_summary['symbol'] = symbol
        sentiment_summary['articles_found'] = len(symbol_articles)
        
        return sentiment_summary
    
    def get_market_sentiment_features(self, hours_back: int = 24) -> Dict[str, float]:
        """Get sentiment features for ML model integration"""
        try:
            sentiment_data = self.analyze_news_sentiment(hours_back)
            summary = sentiment_data['sentiment_summary']
            
            return {
                'news_sentiment_score': summary['overall_sentiment'],
                'news_sentiment_confidence': summary['confidence'],
                'news_positive_ratio': summary['sentiment_distribution']['positive'],
                'news_negative_ratio': summary['sentiment_distribution']['negative'],
                'news_articles_count': summary['articles_analyzed'],
                'news_sentiment_volatility': summary.get('sentiment_std', 0.0)
            }
        except Exception as e:
            logger.error(f"Error getting sentiment features: {e}")
            return {
                'news_sentiment_score': 0.0,
                'news_sentiment_confidence': 0.0,
                'news_positive_ratio': 0.33,
                'news_negative_ratio': 0.33,
                'news_articles_count': 0,
                'news_sentiment_volatility': 0.0
            }

# Example usage and testing
if __name__ == "__main__":
    # Test the news sentiment analyzer
    analyzer = NewsSentimentAnalyzer()
    
    logger.info("Testing News Sentiment Analyzer...")
    
    # Analyze recent news sentiment
    result = analyzer.analyze_news_sentiment(hours_back=12)
    
    print(f"Analysis Results:")
    print(f"Total articles: {result['total_articles']}")
    print(f"Stock-related articles: {result['stock_articles']}")
    print(f"Overall sentiment: {result['sentiment_summary']['overall_sentiment']:.3f}")
    print(f"Confidence: {result['sentiment_summary']['confidence']:.3f}")
    print(f"Positive articles: {result['sentiment_summary']['positive_count']}")
    print(f"Negative articles: {result['sentiment_summary']['negative_count']}")
    
    # Get sentiment features for ML
    features = analyzer.get_market_sentiment_features()
    print(f"ML Features: {features}")
    
    logger.info("News sentiment analysis test completed!")
