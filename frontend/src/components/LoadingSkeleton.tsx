import React from 'react';
import {
  Box,
  Skeleton,
  Card,
  CardContent,
  Grid,
  Paper,
} from '@mui/material';

interface LoadingSkeletonProps {
  variant?: 'card' | 'chart' | 'table' | 'prediction' | 'analysis';
  height?: number | string;
  count?: number;
}

const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({ 
  variant = 'card', 
  height = 200,
  count = 1 
}) => {
  const renderCardSkeleton = () => (
    <Card>
      <CardContent>
        <Skeleton variant="text" width="60%" height={32} sx={{ mb: 2 }} />
        <Skeleton variant="rectangular" width="100%" height={height} sx={{ mb: 2 }} />
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Skeleton variant="text" width="30%" />
          <Skeleton variant="text" width="25%" />
          <Skeleton variant="text" width="20%" />
        </Box>
      </CardContent>
    </Card>
  );

  const renderChartSkeleton = () => (
    <Box>
      <Skeleton variant="text" width="40%" height={32} sx={{ mb: 2 }} />
      <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
        <Skeleton variant="rectangular" width={80} height={32} />
        <Skeleton variant="rectangular" width={80} height={32} />
        <Skeleton variant="rectangular" width={80} height={32} />
      </Box>
      <Skeleton variant="rectangular" width="100%" height={height} />
    </Box>
  );

  const renderTableSkeleton = () => (
    <Box>
      <Skeleton variant="text" width="30%" height={32} sx={{ mb: 2 }} />
      {Array.from({ length: 5 }).map((_, index) => (
        <Box key={index} sx={{ display: 'flex', gap: 2, mb: 1 }}>
          <Skeleton variant="text" width="20%" />
          <Skeleton variant="text" width="15%" />
          <Skeleton variant="text" width="15%" />
          <Skeleton variant="text" width="20%" />
          <Skeleton variant="text" width="10%" />
        </Box>
      ))}
    </Box>
  );

  const renderPredictionSkeleton = () => (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
          <Skeleton variant="text" width="40%" height={32} />
          <Skeleton variant="rectangular" width={80} height={32} />
        </Box>
        
        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={6}>
            <Skeleton variant="text" width="60%" />
            <Skeleton variant="text" width="80%" height={40} />
          </Grid>
          <Grid item xs={6}>
            <Skeleton variant="text" width="60%" />
            <Skeleton variant="text" width="80%" height={40} />
          </Grid>
        </Grid>

        <Skeleton variant="text" width="30%" sx={{ mb: 1 }} />
        <Skeleton variant="rectangular" width="100%" height={8} sx={{ mb: 2 }} />

        <Skeleton variant="text" width="40%" sx={{ mb: 1 }} />
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          {Array.from({ length: 5 }).map((_, index) => (
            <Skeleton key={index} variant="rectangular" width={100} height={32} />
          ))}
        </Box>
      </CardContent>
    </Card>
  );

  const renderAnalysisSkeleton = () => (
    <Card>
      <CardContent>
        <Skeleton variant="text" width="50%" height={32} sx={{ mb: 3 }} />
        
        {/* Technical Indicators */}
        <Skeleton variant="text" width="40%" sx={{ mb: 2 }} />
        <Grid container spacing={1} sx={{ mb: 3 }}>
          {Array.from({ length: 4 }).map((_, index) => (
            <Grid item xs={6} sm={3} key={index}>
              <Paper sx={{ p: 1 }}>
                <Skeleton variant="text" width="60%" />
                <Skeleton variant="text" width="80%" height={24} />
              </Paper>
            </Grid>
          ))}
        </Grid>

        {/* Risk Metrics */}
        <Skeleton variant="text" width="30%" sx={{ mb: 2 }} />
        <Grid container spacing={1} sx={{ mb: 3 }}>
          {Array.from({ length: 4 }).map((_, index) => (
            <Grid item xs={6} sm={3} key={index}>
              <Paper sx={{ p: 1 }}>
                <Skeleton variant="text" width="70%" />
                <Skeleton variant="text" width="50%" height={24} />
              </Paper>
            </Grid>
          ))}
        </Grid>

        {/* Model Performance */}
        <Skeleton variant="text" width="35%" sx={{ mb: 2 }} />
        {Array.from({ length: 3 }).map((_, index) => (
          <Paper key={index} sx={{ p: 1, mb: 1 }}>
            <Skeleton variant="text" width="20%" />
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Skeleton variant="text" width="15%" />
              <Skeleton variant="text" width="15%" />
              <Skeleton variant="text" width="15%" />
            </Box>
          </Paper>
        ))}
      </CardContent>
    </Card>
  );

  const renderSkeleton = () => {
    switch (variant) {
      case 'chart':
        return renderChartSkeleton();
      case 'table':
        return renderTableSkeleton();
      case 'prediction':
        return renderPredictionSkeleton();
      case 'analysis':
        return renderAnalysisSkeleton();
      default:
        return renderCardSkeleton();
    }
  };

  if (count === 1) {
    return renderSkeleton();
  }

  return (
    <>
      {Array.from({ length: count }).map((_, index) => (
        <Box key={index} sx={{ mb: 2 }}>
          {renderSkeleton()}
        </Box>
      ))}
    </>
  );
};

export default LoadingSkeleton;
