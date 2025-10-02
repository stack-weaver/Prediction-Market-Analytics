import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  Paper,
  Divider,
} from '@mui/material';
import {
  Refresh,
  TrendingUp,
  TrendingDown,
  Article,
  SentimentSatisfied,
  AccessTime,
  OpenInNew,
} from '@mui/icons-material';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip } from 'recharts';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

interface SentimentSummary {
  overall_sentiment: number;
  confidence: number;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  sentiment_distribution: {
    positive: number;
    negative: number;
    neutral: number;
  };
  articles_analyzed: number;
}

interface NewsArticle {
  title: string;
  description: string;
  link: string;
  source: string;
  published_date: string;
  sentiment: {
    score: number;
    confidence: number;
    classification: string;
  };
}

interface SentimentData {
  sentiment_analysis: {
    total_articles: number;
    recent_articles: number;
    stock_articles: number;
    analyzed_articles: number;
    articles: NewsArticle[];
    sentiment_summary: SentimentSummary;
    analysis_timestamp: string;
    hours_analyzed: number;
  };
  summary: SentimentSummary;
  hours_analyzed: number;
  timestamp: string;
}

const SENTIMENT_COLORS = {
  positive: '#4caf50',
  negative: '#f44336',
  neutral: '#ff9800',
};

const NewsSentimentDashboard: React.FC = () => {
  const [sentimentData, setSentimentData] = useState<SentimentData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [hoursBack, setHoursBack] = useState<number>(24);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchSentimentData = async (hours: number) => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.get(`/api/news/sentiment?hours=${hours}`, {
        timeout: 15000,
      });
      
      if (!response.data || !response.data.summary || !response.data.sentiment_analysis) {
        throw new Error('Invalid response structure from sentiment API');
      }
      
      setSentimentData(response.data);
      setLastUpdate(new Date());
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to fetch sentiment data';
      setError(errorMessage);
      console.error('Sentiment fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSentimentData(hoursBack);
  }, [hoursBack]);

  const handleHoursChange = (event: SelectChangeEvent<number>) => {
    setHoursBack(event.target.value as number);
  };

  const handleRefresh = () => {
    fetchSentimentData(hoursBack);
  };

  const getSentimentColor = (score: number): string => {
    if (score > 0.1) return SENTIMENT_COLORS.positive;
    if (score < -0.1) return SENTIMENT_COLORS.negative;
    return SENTIMENT_COLORS.neutral;
  };

  const getSentimentLabel = (score: number): string => {
    if (score > 0.3) return 'Very Positive';
    if (score > 0.1) return 'Positive';
    if (score > -0.1) return 'Neutral';
    if (score > -0.3) return 'Negative';
    return 'Very Negative';
  };

  const getSentimentIcon = (classification: string) => {
    switch (classification) {
      case 'positive': return <TrendingUp color="success" />;
      case 'negative': return <TrendingDown color="error" />;
      default: return <SentimentSatisfied color="action" />;
    }
  };

  // Prepare data for charts with safe access
  const pieData = sentimentData?.summary ? [
    { name: 'Positive', value: sentimentData.summary.positive_count || 0, color: SENTIMENT_COLORS.positive },
    { name: 'Negative', value: sentimentData.summary.negative_count || 0, color: SENTIMENT_COLORS.negative },
    { name: 'Neutral', value: sentimentData.summary.neutral_count || 0, color: SENTIMENT_COLORS.neutral },
  ] : [];

  const distributionData = sentimentData?.summary?.sentiment_distribution ? [
    { name: 'Positive', percentage: (sentimentData.summary.sentiment_distribution.positive || 0) * 100 },
    { name: 'Negative', percentage: (sentimentData.summary.sentiment_distribution.negative || 0) * 100 },
    { name: 'Neutral', percentage: (sentimentData.summary.sentiment_distribution.neutral || 0) * 100 },
  ] : [];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* Header Section */}
      <Paper sx={{ p: 3, bgcolor: 'primary.main', color: 'white', overflow: 'visible' }}>
        <Box 
          display="flex" 
          justifyContent="space-between" 
          alignItems="flex-start" 
          flexWrap="wrap" 
          gap={3}
          sx={{ minHeight: 80 }}
        >
          <Box sx={{ flex: 1, minWidth: 250 }}>
            <Typography variant="h5" fontWeight="bold" gutterBottom>
              📰 Market Sentiment Analysis
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.9 }}>
              AI-powered sentiment analysis from financial news sources
            </Typography>
          </Box>
          
          <Box 
            display="flex" 
            alignItems="center" 
            gap={2}
            sx={{ 
              flexShrink: 0,
              minWidth: 'fit-content',
              mt: { xs: 2, sm: 0 }
            }}
          >
            {lastUpdate && (
              <Box textAlign="right">
                <Typography variant="caption" sx={{ opacity: 0.8 }}>
                  Last Updated
                </Typography>
                <Typography variant="body2" fontWeight="bold">
                  {lastUpdate.toLocaleTimeString()}
                </Typography>
              </Box>
            )}
            
            <FormControl 
              size="small" 
              sx={{ 
                minWidth: 140, 
                bgcolor: 'white', 
                borderRadius: 1,
                '& .MuiOutlinedInput-root': {
                  height: 40,
                  '& fieldset': {
                    borderColor: 'rgba(0, 0, 0, 0.23)',
                  },
                  '&:hover fieldset': {
                    borderColor: 'primary.main',
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: 'primary.main',
                  },
                },
                '& .MuiInputLabel-root': {
                  color: 'text.primary',
                  backgroundColor: 'white',
                  paddingX: 0.5,
                  '&.Mui-focused': {
                    color: 'primary.main',
                  },
                  '&.MuiInputLabel-shrink': {
                    backgroundColor: 'white',
                    paddingX: 0.5,
                  },
                },
                '& .MuiSelect-select': {
                  paddingY: 1,
                },
              }}
            >
              <InputLabel>Time Range</InputLabel>
              <Select
                value={hoursBack}
                label="Time Range"
                onChange={handleHoursChange}
              >
                <MenuItem value={6}>6 Hours</MenuItem>
                <MenuItem value={12}>12 Hours</MenuItem>
                <MenuItem value={24}>24 Hours</MenuItem>
                <MenuItem value={48}>48 Hours</MenuItem>
              </Select>
            </FormControl>
            
            <Tooltip title={loading ? "Refreshing..." : "Refresh Data"}>
              <span>
                <IconButton 
                  size="small" 
                  onClick={handleRefresh} 
                  disabled={loading}
                  sx={{ 
                    bgcolor: 'rgba(255,255,255,0.2)', 
                    color: 'white',
                    '&:hover': { 
                      bgcolor: 'rgba(255,255,255,0.3)' 
                    },
                    '&:disabled': {
                      bgcolor: 'rgba(255,255,255,0.1)',
                      color: 'rgba(255,255,255,0.5)',
                    }
                  }}
                >
                  <Refresh sx={{ 
                    animation: loading ? 'spin 1s linear infinite' : 'none',
                    '@keyframes spin': {
                      '0%': {
                        transform: 'rotate(0deg)',
                      },
                      '100%': {
                        transform: 'rotate(360deg)',
                      },
                    },
                  }} />
                </IconButton>
              </span>
            </Tooltip>
          </Box>
        </Box>

        {loading && <LinearProgress sx={{ mt: 2, bgcolor: 'rgba(255,255,255,0.2)' }} />}
      </Paper>

      {error && (
        <Paper sx={{ p: 2, bgcolor: 'error.light', color: 'error.contrastText' }}>
          <Typography align="center" fontWeight="bold">
            ⚠️ {error}
          </Typography>
        </Paper>
      )}

      {sentimentData && !loading && (
        <>
          {/* Key Metrics Cards */}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            
            {/* Overall Sentiment Score - Enhanced */}
            <Paper sx={{ p: 3, textAlign: 'center', bgcolor: 'background.paper', border: '2px solid', borderColor: getSentimentColor(sentimentData.summary.overall_sentiment) }}>
              <Typography variant="h6" gutterBottom fontWeight="bold">
                🎯 Overall Market Sentiment
              </Typography>
              
              <Box display="flex" alignItems="center" justifyContent="center" gap={2} mb={2}>
                <Box sx={{ 
                  p: 2, 
                  borderRadius: '50%', 
                  bgcolor: getSentimentColor(sentimentData.summary.overall_sentiment),
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  {getSentimentIcon(
                    sentimentData.summary.overall_sentiment > 0.1 ? 'positive' :
                    sentimentData.summary.overall_sentiment < -0.1 ? 'negative' : 'neutral'
                  )}
                </Box>
                <Box textAlign="left">
                  <Typography 
                    variant="h2" 
                    fontWeight="bold"
                    sx={{ color: getSentimentColor(sentimentData.summary.overall_sentiment) }}
                  >
                    {sentimentData.summary.overall_sentiment.toFixed(3)}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    Sentiment Score
                  </Typography>
                </Box>
              </Box>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 2 }}>
                <Box textAlign="center">
                  <Chip
                    label={getSentimentLabel(sentimentData.summary.overall_sentiment)}
                    sx={{ 
                      color: 'white',
                      bgcolor: getSentimentColor(sentimentData.summary.overall_sentiment),
                      fontWeight: 'bold',
                      fontSize: '0.9rem'
                    }}
                  />
                </Box>
                <Box textAlign="center">
                  <Typography variant="body2" color="textSecondary">Confidence</Typography>
                  <Typography variant="h6" fontWeight="bold" color="primary.main">
                    {(sentimentData.summary.confidence * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <Box textAlign="center">
                  <Typography variant="body2" color="textSecondary">Articles</Typography>
                  <Typography variant="h6" fontWeight="bold" color="primary.main">
                    {sentimentData.summary.articles_analyzed}
                  </Typography>
                </Box>
              </Box>
            </Paper>

            {/* Quick Stats Row */}
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(3, 1fr)' }, gap: 2 }}>
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light', color: 'success.contrastText' }}>
                <Typography variant="h4" fontWeight="bold">
                  {sentimentData.summary.positive_count}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Positive Articles
                </Typography>
                <Typography variant="caption" sx={{ opacity: 0.8 }}>
                  {(sentimentData.summary.sentiment_distribution.positive * 100).toFixed(1)}%
                </Typography>
              </Paper>
              
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'error.light', color: 'error.contrastText' }}>
                <Typography variant="h4" fontWeight="bold">
                  {sentimentData.summary.negative_count}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Negative Articles
                </Typography>
                <Typography variant="caption" sx={{ opacity: 0.8 }}>
                  {(sentimentData.summary.sentiment_distribution.negative * 100).toFixed(1)}%
                </Typography>
              </Paper>
              
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.light', color: 'warning.contrastText' }}>
                <Typography variant="h4" fontWeight="bold">
                  {sentimentData.summary.neutral_count}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Neutral Articles
                </Typography>
                <Typography variant="caption" sx={{ opacity: 0.8 }}>
                  {(sentimentData.summary.sentiment_distribution.neutral * 100).toFixed(1)}%
                </Typography>
              </Paper>
            </Box>

            {/* Charts and Analysis Section */}
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
              
              {/* Enhanced Pie Chart */}
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom fontWeight="bold">
                  📊 Sentiment Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={100}
                      paddingAngle={3}
                      dataKey="value"
                      label={({ name, value, percent }) => `${name}: ${value} (${((percent as number) * 100).toFixed(1)}%)`}
                      labelLine={false}
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <RechartsTooltip 
                      formatter={(value, name) => [value, `${name} Articles`]}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </Paper>

              {/* Enhanced Bar Chart */}
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom fontWeight="bold">
                  📈 Percentage Breakdown
                </Typography>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={distributionData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                    <YAxis 
                      tick={{ fontSize: 12 }}
                      tickFormatter={(value) => `${value}%`}
                      domain={[0, 100]}
                    />
                    <RechartsTooltip 
                      formatter={(value) => [`${(value as number).toFixed(1)}%`, 'Percentage']}
                    />
                    <Bar 
                      dataKey="percentage" 
                      fill="#1976d2"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </Paper>
            </Box>

            {/* Analysis Insights */}
            <Paper sx={{ p: 3, bgcolor: 'grey.50' }}>
              <Typography variant="h6" gutterBottom fontWeight="bold">
                🔍 Analysis Insights
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }, gap: 2 }}>
                <Box textAlign="center">
                  <Typography variant="body2" color="textSecondary">Time Period</Typography>
                  <Typography variant="h6" fontWeight="bold" color="primary.main">
                    {sentimentData.hours_analyzed} Hours
                  </Typography>
                </Box>
                <Box textAlign="center">
                  <Typography variant="body2" color="textSecondary">Dominant Sentiment</Typography>
                  <Typography variant="h6" fontWeight="bold" sx={{ color: getSentimentColor(sentimentData.summary.overall_sentiment) }}>
                    {getSentimentLabel(sentimentData.summary.overall_sentiment)}
                  </Typography>
                </Box>
                <Box textAlign="center">
                  <Typography variant="body2" color="textSecondary">Market Mood</Typography>
                  <Typography variant="h6" fontWeight="bold" color={
                    sentimentData.summary.positive_count > sentimentData.summary.negative_count ? 'success.main' : 
                    sentimentData.summary.negative_count > sentimentData.summary.positive_count ? 'error.main' : 'warning.main'
                  }>
                    {sentimentData.summary.positive_count > sentimentData.summary.negative_count ? 'Optimistic' : 
                     sentimentData.summary.negative_count > sentimentData.summary.positive_count ? 'Pessimistic' : 'Mixed'}
                  </Typography>
                </Box>
                <Box textAlign="center">
                  <Typography variant="body2" color="textSecondary">Data Quality</Typography>
                  <Typography variant="h6" fontWeight="bold" color={
                    sentimentData.summary.confidence > 0.7 ? 'success.main' : 
                    sentimentData.summary.confidence > 0.5 ? 'warning.main' : 'error.main'
                  }>
                    {sentimentData.summary.confidence > 0.7 ? 'High' : 
                     sentimentData.summary.confidence > 0.5 ? 'Medium' : 'Low'}
                  </Typography>
                </Box>
              </Box>
            </Paper>

            {/* Enhanced Recent Articles */}
            <Paper sx={{ p: 3 }}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6" fontWeight="bold">
                  📰 Latest News Articles
                </Typography>
                <Chip 
                  label={`${sentimentData?.sentiment_analysis?.articles?.length || 0} Articles`}
                  color="primary"
                  size="small"
                />
              </Box>
              
              <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                {!sentimentData?.sentiment_analysis?.articles || sentimentData.sentiment_analysis.articles.length === 0 ? (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Typography color="textSecondary" variant="body2">
                      📭 No articles found for the selected time period
                    </Typography>
                    <Typography color="textSecondary" variant="caption">
                      Try selecting a longer time range or refresh the data
                    </Typography>
                  </Box>
                ) : (
                  <List>
                    {sentimentData.sentiment_analysis.articles.slice(0, 15).map((article, index) => (
                    <ListItem 
                      key={index} 
                      divider={index < 14}
                      sx={{ 
                        '&:hover': { bgcolor: 'grey.50' },
                        borderRadius: 1,
                        mb: 1
                      }}
                    >
                      <Box sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: 2, 
                        width: '100%',
                        p: 1
                      }}>
                        {/* Sentiment Icon */}
                        <Box sx={{ 
                          p: 1, 
                          borderRadius: '50%', 
                          bgcolor: getSentimentColor(article.sentiment.score),
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          minWidth: 40,
                          height: 40
                        }}>
                          {getSentimentIcon(article.sentiment.classification)}
                        </Box>
                        
                        {/* Article Content */}
                        <Box sx={{ flex: 1, minWidth: 0 }}>
                          <Typography 
                            variant="body1" 
                            fontWeight="bold" 
                            sx={{ 
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              display: '-webkit-box',
                              WebkitLineClamp: 2,
                              WebkitBoxOrient: 'vertical',
                              mb: 0.5
                            }}
                          >
                            {article.title}
                          </Typography>
                          
                          <Box display="flex" alignItems="center" gap={1} mb={1}>
                            <Typography variant="caption" color="textSecondary">
                              {article.source}
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                              • {new Date(article.published_date).toLocaleDateString()}
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                              • {new Date(article.published_date).toLocaleTimeString()}
                            </Typography>
                          </Box>
                          
                          <Box display="flex" alignItems="center" gap={1}>
                            <Chip
                              label={article.sentiment.classification.toUpperCase()}
                              size="small"
                              sx={{ 
                                bgcolor: getSentimentColor(article.sentiment.score),
                                color: 'white',
                                fontSize: '0.7rem',
                                fontWeight: 'bold'
                              }}
                            />
                            <Typography variant="caption" color="textSecondary">
                              Score: {article.sentiment.score.toFixed(3)}
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                              • Confidence: {(article.sentiment.confidence * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                        </Box>
                        
                        {/* Action Button */}
                        <Tooltip title="Read Full Article">
                          <IconButton 
                            size="small" 
                            onClick={() => window.open(article.link, '_blank')}
                            sx={{ 
                              bgcolor: 'primary.light',
                              color: 'primary.contrastText',
                              '&:hover': { bgcolor: 'primary.main' }
                            }}
                          >
                            <OpenInNew fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </ListItem>
                    ))}
                  </List>
                )}
              </Box>
            </Paper>
          </Box>
        </>
      )}
    </Box>
  );
};

export default NewsSentimentDashboard;
