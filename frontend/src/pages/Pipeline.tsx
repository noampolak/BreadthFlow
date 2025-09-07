import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Grid,
  Card,
  CardContent,
} from '@mui/material';
import { PlayArrow, Stop, Refresh } from '@mui/icons-material';
import api from '../services/api';
import { useWebSocket } from '../hooks/useWebSocket';

interface PipelineConfig {
  mode: string;
  interval: string;
  timeframe: string;
  symbols?: string;
  data_source: string;
}

interface PipelineRun {
  run_id: string;
  command: string;
  status: string;
  start_time: string;
  end_time?: string;
  duration?: number;
  error_message?: string;
  run_metadata?: any;
}

interface PipelineStatus {
  state: string;
  total_runs: number;
  successful_runs: number;
  failed_runs: number;
  stopped_runs: number;
  uptime_seconds?: number;
  last_run_time?: string;
}

const Pipeline: React.FC = () => {
  const [config, setConfig] = useState<PipelineConfig>({
    mode: 'demo',
    interval: '5m',
    timeframe: '1day',
    symbols: '',
    data_source: 'yfinance',
  });
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const { connected, data: wsData } = useWebSocket();


  const fetchStatus = async () => {
    try {
      const response = await api.get('/api/pipeline/status');
      setStatus(response.data);
    } catch (err: any) {
      console.error('Failed to fetch pipeline status:', err);
    }
  };

  const fetchRuns = async () => {
    try {
      const response = await api.get('/api/pipeline/runs?limit=10');
      setRuns(response.data);
    } catch (err: any) {
      console.error('Failed to fetch pipeline runs:', err);
    }
  };

  const startPipeline = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      
      const response = await api.post('/api/pipeline/start', config);
      setSuccess(`Pipeline started successfully: ${response.data.run_id}`);
      await fetchStatus();
      await fetchRuns();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start pipeline');
    } finally {
      setLoading(false);
    }
  };

  const stopPipeline = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      
      const response = await api.post('/api/pipeline/stop');
      setSuccess(`Pipeline stopped successfully: ${response.data.run_id}`);
      await fetchStatus();
      await fetchRuns();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to stop pipeline');
    } finally {
      setLoading(false);
    }
  };

  const refreshData = async () => {
    try {
      await Promise.all([fetchStatus(), fetchRuns()]);
    } catch (err) {
      console.error('Error refreshing pipeline data:', err);
      setError('Failed to refresh pipeline data');
    }
  };

  useEffect(() => {
    refreshData();
    const interval = setInterval(refreshData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Handle WebSocket updates
  useEffect(() => {
    if (wsData?.type === 'pipeline_status_update') {
      refreshData();
    }
  }, [wsData]);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'running': return 'success';
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'stopped': return 'warning';
      default: return 'default';
    }
  };

  const isRunning = status?.state === 'running';

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
        🎮 Pipeline Management
      </Typography>
      <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
        Real-time pipeline control and monitoring
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

      {/* Status Cards */}
      {status && (
        <Box sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
          gap: 3,
          mb: 4
        }}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Pipeline State
              </Typography>
              <Typography variant="h4" component="div">
                <Chip 
                  label={status.state.toUpperCase()} 
                  color={getStatusColor(status.state) as any}
                  size="small"
                />
              </Typography>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Runs
              </Typography>
              <Typography variant="h4" component="div">
                {status.total_runs}
              </Typography>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Success Rate
              </Typography>
              <Typography variant="h4" component="div">
                {status.total_runs > 0 
                  ? `${Math.round((status.successful_runs / status.total_runs) * 100)}%`
                  : '0%'
                }
              </Typography>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Uptime
              </Typography>
              <Typography variant="h4" component="div">
                {status.uptime_seconds 
                  ? `${Math.floor(status.uptime_seconds / 60)}m ${status.uptime_seconds % 60}s`
                  : '-'
                }
              </Typography>
            </CardContent>
          </Card>
        </Box>
      )}

      {/* Control Panel */}
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="h5" component="h2" gutterBottom sx={{ fontWeight: 'bold' }}>
          🎛️ Pipeline Control
        </Typography>
        
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}

        <Box sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
          gap: 3,
          mb: 3
        }}>
          <FormControl fullWidth>
            <InputLabel>Mode</InputLabel>
            <Select
              value={config.mode}
              label="Mode"
              onChange={(e) => setConfig({ ...config, mode: e.target.value })}
            >
              <MenuItem value="demo">Demo</MenuItem>
              <MenuItem value="demo_small">Demo Small</MenuItem>
              <MenuItem value="tech_leaders">Tech Leaders</MenuItem>
              <MenuItem value="all_symbols">All Symbols</MenuItem>
              <MenuItem value="custom">Custom</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth>
            <InputLabel>Interval</InputLabel>
            <Select
              value={config.interval}
              label="Interval"
              onChange={(e) => setConfig({ ...config, interval: e.target.value })}
            >
              <MenuItem value="1m">1 Minute</MenuItem>
              <MenuItem value="5m">5 Minutes</MenuItem>
              <MenuItem value="15m">15 Minutes</MenuItem>
              <MenuItem value="1h">1 Hour</MenuItem>
              <MenuItem value="6h">6 Hours</MenuItem>
              <MenuItem value="1d">1 Day</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth>
            <InputLabel>Timeframe</InputLabel>
            <Select
              value={config.timeframe}
              label="Timeframe"
              onChange={(e) => setConfig({ ...config, timeframe: e.target.value })}
            >
              <MenuItem value="1min">1 Minute</MenuItem>
              <MenuItem value="5min">5 Minutes</MenuItem>
              <MenuItem value="15min">15 Minutes</MenuItem>
              <MenuItem value="1hour">1 Hour</MenuItem>
              <MenuItem value="1day">1 Day</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth>
            <InputLabel>Data Source</InputLabel>
            <Select
              value={config.data_source}
              label="Data Source"
              onChange={(e) => setConfig({ ...config, data_source: e.target.value })}
            >
              <MenuItem value="yfinance">Yahoo Finance</MenuItem>
              <MenuItem value="alpha_vantage">Alpha Vantage</MenuItem>
              <MenuItem value="polygon">Polygon</MenuItem>
            </Select>
          </FormControl>
        </Box>

        {config.mode === 'custom' && (
          <Box sx={{ mb: 3 }}>
            <TextField
              fullWidth
              label="Custom Symbols (comma-separated)"
              value={config.symbols}
              onChange={(e) => setConfig({ ...config, symbols: e.target.value })}
              placeholder="AAPL,MSFT,GOOGL"
              helperText="Enter stock symbols separated by commas"
            />
          </Box>
        )}

        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Button
            variant="contained"
            color="success"
            startIcon={<PlayArrow />}
            onClick={startPipeline}
            disabled={loading || isRunning}
            size="large"
          >
            {loading ? <CircularProgress size={20} /> : 'Start Pipeline'}
          </Button>
          
          <Button
            variant="contained"
            color="error"
            startIcon={<Stop />}
            onClick={stopPipeline}
            disabled={loading || !isRunning}
            size="large"
          >
            {loading ? <CircularProgress size={20} /> : 'Stop Pipeline'}
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={refreshData}
            disabled={loading}
            size="large"
          >
            Refresh
          </Button>
        </Box>
      </Paper>

      {/* Recent Runs */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h5" component="h2" gutterBottom sx={{ fontWeight: 'bold' }}>
          📋 Recent Pipeline Runs
        </Typography>
        
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Run ID</TableCell>
                <TableCell>Command</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Start Time</TableCell>
                <TableCell>End Time</TableCell>
                <TableCell>Duration</TableCell>
                <TableCell>Error</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {runs.map((run) => (
                <TableRow key={run.run_id}>
                  <TableCell>{run.run_id}</TableCell>
                  <TableCell>{run.command}</TableCell>
                  <TableCell>
                    <Chip 
                      label={run.status.toUpperCase()} 
                      color={getStatusColor(run.status) as any}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>{new Date(run.start_time).toLocaleString()}</TableCell>
                  <TableCell>{run.end_time ? new Date(run.end_time).toLocaleString() : '-'}</TableCell>
                  <TableCell>{run.duration ? `${run.duration.toFixed(2)}s` : '-'}</TableCell>
                  <TableCell>{run.error_message || '-'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </Container>
  );
};

export default Pipeline;
