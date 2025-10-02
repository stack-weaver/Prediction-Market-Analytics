import React, { Component, ReactNode } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Alert,
  AlertTitle,
} from '@mui/material';
import { Refresh, BugReport } from '@mui/icons-material';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: any;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({
      error,
      errorInfo,
    });
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Box sx={{ p: 3, maxWidth: 600, mx: 'auto', mt: 4 }}>
          <Paper sx={{ p: 3 }}>
            <Alert severity="error" sx={{ mb: 2 }}>
              <AlertTitle>Something went wrong</AlertTitle>
              <Typography variant="body2" sx={{ mb: 2 }}>
                We encountered an unexpected error. This has been logged and our team will investigate.
              </Typography>
              
              {process.env.NODE_ENV === 'development' && this.state.error && (
                <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
                  <Typography variant="caption" component="pre" sx={{ fontSize: '0.75rem' }}>
                    {this.state.error.toString()}
                    {this.state.errorInfo?.componentStack}
                  </Typography>
                </Box>
              )}
            </Alert>

            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Button
                variant="contained"
                startIcon={<Refresh />}
                onClick={this.handleRetry}
              >
                Try Again
              </Button>
              
              <Button
                variant="outlined"
                startIcon={<BugReport />}
                onClick={() => window.location.reload()}
              >
                Reload Page
              </Button>
            </Box>

            <Box sx={{ mt: 3, p: 2, bgcolor: 'info.light', borderRadius: 1 }}>
              <Typography variant="body2" color="info.contrastText">
                💡 <strong>Troubleshooting Tips:</strong>
              </Typography>
              <Typography variant="body2" color="info.contrastText" sx={{ mt: 1 }}>
                • Check your internet connection
                <br />
                • Try refreshing the page
                <br />
                • Clear your browser cache
                <br />
                • Contact support if the problem persists
              </Typography>
            </Box>
          </Paper>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
