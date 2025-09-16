import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Tooltip,
} from '@mui/material';
import { Download, Refresh, TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material';
import api from '../services/api';
import { useWebSocket } from '../hooks/useWebSocket';

interface TradingSignal {
  symbol: string;
  signal_type: string;
  confidence: number;
  strength: string;
  date: string;
  timeframe: string;
  create_time: string;
}

interface SignalStats {
  total_signals: number;
  buy_signals: number;
  sell_signals: number;
  hold_signals: number;
  avg_confidence: number;
  strong_signals: number;
}

const Signals: React.FC = () => {
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [stats, setStats] = useState<SignalStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeframeFilter, setTimeframeFilter] = useState<string>('all');
  const [signalTypeFilter, setSignalTypeFilter] = useState<string>('all');
  const { connected, data: wsData } = useWebSocket();

  const fetchSignals = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await api.get('/api/signals/latest');
      setSignals(response.data.signals || []);
      setStats(response.data.stats || null);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch trading signals');
    } finally {
      setLoading(false);
    }
  };

  const exportSignals = async (format: 'csv' | 'json') => {
    try {
      const response = await api.get(`/api/signals/export?format=${format}`, {
        responseType: 'blob',
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `trading_signals.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      setError(`Failed to export signals: ${err.response?.data?.detail || err.message}`);
    }
  };

  useEffect(() => {
    fetchSignals();
    const interval = setInterval(fetchSignals, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Handle WebSocket updates
  useEffect(() => {
    if (wsData?.type === 'signals_update') {
      fetchSignals();
    }
  }, [wsData]);

  const getSignalIcon = (signalType: string) => {
    switch (signalType.toLowerCase()) {
      case 'buy': return <TrendingUp color="success" />;
      case 'sell': return <TrendingDown color="error" />;
      case 'hold': return <TrendingFlat color="warning" />;
      default: return <TrendingFlat />;
    }
  };

  const getSignalColor = (signalType: string) => {
    switch (signalType.toLowerCase()) {
      case 'buy': return 'success';
      case 'sell': return 'error';
      case 'hold': return 'warning';
      default: return 'default';
    }
  };

  const getStrengthColor = (strength: string) => {
    switch (strength.toLowerCase()) {
      case 'strong': return 'error';
      case 'medium': return 'warning';
      case 'weak': return 'info';
      default: return 'default';
    }
  };

  const filteredSignals = signals.filter(signal => {
    const timeframeMatch = timeframeFilter === 'all' || signal.timeframe === timeframeFilter;
    const signalTypeMatch = signalTypeFilter === 'all' || signal.signal_type.toLowerCase() === signalTypeFilter;
    return timeframeMatch && signalTypeMatch;
  });

  const uniqueTimeframes = Array.from(new Set(signals.map(s => s.timeframe)));
  const uniqueSignalTypes = Array.from(new Set(signals.map(s => s.signal_type.toLowerCase())));

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
        📈 Trading Signals
      </Typography>
      <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
        Real-time trading signal monitoring and analysis
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

      {/* Stats Cards */}
      {stats && (
        <Box sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(6, 1fr)' },
          gap: 3,
          mb: 4
        }}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Signals
              </Typography>
              <Typography variant="h4" component="div">
                {stats.total_signals}
              </Typography>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Buy Signals
              </Typography>
              <Typography variant="h4" component="div" color="success.main">
                {stats.buy_signals}
              </Typography>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Sell Signals
              </Typography>
              <Typography variant="h4" component="div" color="error.main">
                {stats.sell_signals}
              </Typography>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Hold Signals
              </Typography>
              <Typography variant="h4" component="div" color="warning.main">
                {stats.hold_signals}
              </Typography>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Avg Confidence
              </Typography>
              <Typography variant="h4" component="div">
                {stats.avg_confidence.toFixed(1)}%
              </Typography>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Strong Signals
              </Typography>
              <Typography variant="h4" component="div" color="error.main">
                {stats.strong_signals}
              </Typography>
            </CardContent>
          </Card>
        </Box>
      )}

      {/* Controls */}
      <Paper sx={{ p: 3, mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" component="h2" sx={{ fontWeight: 'bold' }}>
            🎛️ Signal Controls
          </Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title="Export as CSV">
              <IconButton onClick={() => exportSignals('csv')} color="primary">
                <Download />
              </IconButton>
            </Tooltip>
            <Tooltip title="Export as JSON">
              <IconButton onClick={() => exportSignals('json')} color="primary">
                <Download />
              </IconButton>
            </Tooltip>
            <Tooltip title="Refresh Signals">
              <IconButton onClick={fetchSignals} color="primary">
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

        <Box sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' },
          gap: 3
        }}>
          <FormControl fullWidth>
            <InputLabel>Timeframe Filter</InputLabel>
            <Select
              value={timeframeFilter}
              label="Timeframe Filter"
              onChange={(e) => setTimeframeFilter(e.target.value)}
            >
              <MenuItem value="all">All Timeframes</MenuItem>
              {uniqueTimeframes.map(timeframe => (
                <MenuItem key={timeframe} value={timeframe}>
                  {timeframe}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl fullWidth>
            <InputLabel>Signal Type Filter</InputLabel>
            <Select
              value={signalTypeFilter}
              label="Signal Type Filter"
              onChange={(e) => setSignalTypeFilter(e.target.value)}
            >
              <MenuItem value="all">All Signal Types</MenuItem>
              {uniqueSignalTypes.map(signalType => (
                <MenuItem key={signalType} value={signalType}>
                  {signalType.toUpperCase()}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Box sx={{ display: 'flex', alignItems: 'center', height: '100%' }}>
            <Typography variant="body2" color="text.secondary">
              Showing {filteredSignals.length} of {signals.length} signals
            </Typography>
          </Box>
        </Box>
      </Paper>

      {/* Signals Table */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h5" component="h2" gutterBottom sx={{ fontWeight: 'bold' }}>
          📊 Trading Signals ({filteredSignals.length})
        </Typography>
        
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Signal</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Strength</TableCell>
                  <TableCell>Timeframe</TableCell>
                  <TableCell>Date</TableCell>
                  <TableCell>Create Time</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredSignals.map((signal, index) => (
                  <TableRow key={`${signal.symbol}-${signal.create_time}-${index}`}>
                    <TableCell>
                      <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                        {signal.symbol}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {getSignalIcon(signal.signal_type)}
                        <Chip 
                          label={signal.signal_type.toUpperCase()} 
                          color={getSignalColor(signal.signal_type) as any}
                          size="small"
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {(signal.confidence * 100).toFixed(1)}%
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={signal.strength.toUpperCase()} 
                        color={getStrengthColor(signal.strength) as any}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={signal.timeframe} 
                        variant="outlined"
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {new Date(signal.date).toLocaleDateString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {new Date(signal.create_time).toLocaleString()}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}

        {filteredSignals.length === 0 && !loading && (
          <Box sx={{ textAlign: 'center', p: 4 }}>
            <Typography variant="h6" color="text.secondary">
              No signals found matching your filters
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Try adjusting your filters or wait for new signals to be generated
            </Typography>
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default Signals;
