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
  Button,
  LinearProgress,
} from '@mui/material';
import { Refresh, CheckCircle, Error, Warning, Info } from '@mui/icons-material';
import api from '../services/api';
import { useWebSocket } from '../hooks/useWebSocket';

interface ServiceStatus {
  name: string;
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  url?: string;
  response_time?: number;
  last_check: string;
  details?: string;
}

interface SystemHealth {
  overall_status: 'healthy' | 'warning' | 'error';
  services: ServiceStatus[];
  system_resources: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    network_status: 'connected' | 'disconnected';
  };
  database_status: {
    connected: boolean;
    response_time?: number;
    active_connections?: number;
  };
  last_updated: string;
}

const Infrastructure: React.FC = () => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { connected, data: wsData } = useWebSocket();

  const fetchHealth = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await api.get('/api/infrastructure/health');
      setHealth(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch infrastructure health');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Handle WebSocket updates
  useEffect(() => {
    if (wsData?.type === 'infrastructure_update') {
      fetchHealth();
    }
  }, [wsData]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle color="success" />;
      case 'warning': return <Warning color="warning" />;
      case 'error': return <Error color="error" />;
      default: return <Info color="info" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getUsageColor = (usage: number) => {
    if (usage < 50) return 'success';
    if (usage < 80) return 'warning';
    return 'error';
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
        🏗️ Infrastructure Status
      </Typography>
      <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
        System health monitoring and service status
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

      {/* Overall Status */}
      {health && (
        <Paper sx={{ p: 3, mb: 4 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h5" component="h2" sx={{ fontWeight: 'bold' }}>
              🎯 Overall System Status
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {getStatusIcon(health.overall_status)}
              <Chip 
                label={health.overall_status.toUpperCase()} 
                color={getStatusColor(health.overall_status) as any}
                size="medium"
              />
            </Box>
          </Box>
          <Typography variant="body2" color="text.secondary">
            Last updated: {new Date(health.last_updated).toLocaleString()}
          </Typography>
        </Paper>
      )}

      {/* System Resources */}
      {health && (
        <Box sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
          gap: 3,
          mb: 4
        }}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                CPU Usage
              </Typography>
              <Typography variant="h4" component="div" sx={{ mb: 1 }}>
                {health.system_resources.cpu_usage.toFixed(1)}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={health.system_resources.cpu_usage}
                color={getUsageColor(health.system_resources.cpu_usage) as any}
              />
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Memory Usage
              </Typography>
              <Typography variant="h4" component="div" sx={{ mb: 1 }}>
                {health.system_resources.memory_usage.toFixed(1)}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={health.system_resources.memory_usage}
                color={getUsageColor(health.system_resources.memory_usage) as any}
              />
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Disk Usage
              </Typography>
              <Typography variant="h4" component="div" sx={{ mb: 1 }}>
                {health.system_resources.disk_usage.toFixed(1)}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={health.system_resources.disk_usage}
                color={getUsageColor(health.system_resources.disk_usage) as any}
              />
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Network Status
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                {getStatusIcon(health.system_resources.network_status === 'connected' ? 'healthy' : 'error')}
                <Typography variant="h6" component="div">
                  {health.system_resources.network_status.toUpperCase()}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Box>
      )}

      {/* Database Status */}
      {health && (
        <Paper sx={{ p: 3, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom sx={{ fontWeight: 'bold' }}>
            🗄️ Database Status
          </Typography>
          <Box sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', sm: 'repeat(3, 1fr)' },
            gap: 3
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {getStatusIcon(health.database_status.connected ? 'healthy' : 'error')}
              <Typography variant="h6">
                Connection: {health.database_status.connected ? 'Connected' : 'Disconnected'}
              </Typography>
            </Box>
            <Typography variant="body1">
              Response Time: {health.database_status.response_time ? `${health.database_status.response_time}ms` : 'N/A'}
            </Typography>
            <Typography variant="body1">
              Active Connections: {health.database_status.active_connections || 'N/A'}
            </Typography>
          </Box>
        </Paper>
      )}

      {/* Services Status */}
      <Paper sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" component="h2" sx={{ fontWeight: 'bold' }}>
            🔧 Services Status
          </Typography>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={fetchHealth}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>

        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Service</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Response Time</TableCell>
                  <TableCell>Last Check</TableCell>
                  <TableCell>Details</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {health?.services.map((service) => (
                  <TableRow key={service.name}>
                    <TableCell>
                      <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                        {service.name}
                      </Typography>
                      {service.url && (
                        <Typography variant="caption" color="text.secondary" display="block">
                          {service.url}
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {getStatusIcon(service.status)}
                        <Chip 
                          label={service.status.toUpperCase()} 
                          color={getStatusColor(service.status) as any}
                          size="small"
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      {service.response_time ? `${service.response_time}ms` : '-'}
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {new Date(service.last_check).toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {service.details || '-'}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}

        {health?.services.length === 0 && !loading && (
          <Box sx={{ textAlign: 'center', p: 4 }}>
            <Typography variant="h6" color="text.secondary">
              No services found
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Services will appear here once they are registered and monitored
            </Typography>
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default Infrastructure;
