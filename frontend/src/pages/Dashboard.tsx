import React, { useEffect, useState } from 'react';
import { Container, Typography, Box, Paper, CircularProgress, Alert } from '@mui/material';
import { DashboardSummary } from '../services/types';
import { useWebSocket } from '../hooks/useWebSocket';
import StatsCards from '../components/dashboard/StatsCards';
import api from '../services/api';

const Dashboard: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { connected, data: wsData } = useWebSocket();

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await api.get('/api/dashboard/summary');
      setDashboardData(response.data);
      setError(null);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch dashboard data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  // Handle WebSocket updates
  useEffect(() => {
    if (wsData?.type === 'dashboard_update') {
      setDashboardData(wsData.data);
    }
  }, [wsData]);

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress size={60} />
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
          🚀 BreadthFlow Dashboard
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
          Real-time pipeline monitoring and analytics
        </Typography>
        
        {/* Connection Status */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
          <Box
            sx={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              backgroundColor: connected ? '#4caf50' : '#f44336',
              animation: connected ? 'pulse 2s infinite' : 'none',
              '@keyframes pulse': {
                '0%': { opacity: 1 },
                '50%': { opacity: 0.5 },
                '100%': { opacity: 1 }
              }
            }}
          />
          <Typography variant="body2" color="text.secondary">
            {connected ? 'Connected to real-time updates' : 'Disconnected from real-time updates'}
          </Typography>
        </Box>
      </Box>

      {/* Stats Cards */}
      {dashboardData && (
        <StatsCards stats={dashboardData.stats} />
      )}

      {/* Recent Runs */}
      {dashboardData && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h5" component="h2" gutterBottom sx={{ fontWeight: 'bold' }}>
            📋 Recent Pipeline Runs
          </Typography>
          <Box sx={{ overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e0e0e0' }}>
                  <th style={{ padding: '12px', textAlign: 'left' }}>Command</th>
                  <th style={{ padding: '12px', textAlign: 'left' }}>Status</th>
                  <th style={{ padding: '12px', textAlign: 'left' }}>Duration</th>
                  <th style={{ padding: '12px', textAlign: 'left' }}>Start Time</th>
                </tr>
              </thead>
              <tbody>
                {dashboardData.recent_runs.map((run) => (
                  <tr key={run.run_id} style={{ borderBottom: '1px solid #f0f0f0' }}>
                    <td style={{ padding: '12px' }}>{run.command}</td>
                    <td style={{ padding: '12px' }}>
                      <span
                        style={{
                          padding: '4px 8px',
                          borderRadius: '4px',
                          fontSize: '12px',
                          fontWeight: 'bold',
                          backgroundColor: 
                            run.status === 'completed' ? '#e8f5e8' :
                            run.status === 'failed' ? '#ffebee' :
                            run.status === 'running' ? '#e3f2fd' : '#f5f5f5',
                          color:
                            run.status === 'completed' ? '#2e7d32' :
                            run.status === 'failed' ? '#c62828' :
                            run.status === 'running' ? '#1976d2' : '#666'
                        }}
                      >
                        {run.status.toUpperCase()}
                      </span>
                    </td>
                    <td style={{ padding: '12px' }}>
                      {run.duration ? `${run.duration.toFixed(2)}s` : '-'}
                    </td>
                    <td style={{ padding: '12px' }}>
                      {new Date(run.start_time).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Box>
        </Paper>
      )}

      {/* Last Updated */}
      {dashboardData && (
        <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', mt: 2 }}>
          Last updated: {new Date(dashboardData.last_updated).toLocaleString()}
        </Typography>
      )}
    </Container>
  );
};

export default Dashboard;
