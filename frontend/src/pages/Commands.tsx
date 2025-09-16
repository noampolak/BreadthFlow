import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import HistoryIcon from '@mui/icons-material/History';
import api from '../services/api';
import { useWebSocket } from '../hooks/useWebSocket';

interface CommandRequest {
  command: string;
  parameters?: Record<string, any>;
  background: boolean;
}

interface CommandResponse {
  command_id: string;
  command: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  output?: string;
  error?: string;
  start_time: string;
  end_time?: string;
  duration?: number;
}

interface QuickFlow {
  id: string;
  name: string;
  description: string;
  commands: string[];
}

interface CommandTemplate {
  name: string;
  template: string;
  parameters: string[];
}

const Commands: React.FC = () => {
  const [command, setCommand] = useState('');
  const [parameters] = useState<Record<string, string>>({});
  const [background, setBackground] = useState('false');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [commandHistory, setCommandHistory] = useState<CommandResponse[]>([]);
  const [quickFlows, setQuickFlows] = useState<QuickFlow[]>([]);
  const [commandTemplates, setCommandTemplates] = useState<Record<string, CommandTemplate[]>>({});
  const [selectedFlow, setSelectedFlow] = useState<string>('');
  const { connected } = useWebSocket();

  useEffect(() => {
    fetchQuickFlows();
    fetchCommandTemplates();
    fetchCommandHistory();
  }, []);

  const fetchQuickFlows = async () => {
    try {
      const response = await api.get('/api/commands/quick-flows');
      setQuickFlows(response.data);
    } catch (err) {
      console.error('Failed to fetch quick flows:', err);
    }
  };

  const fetchCommandTemplates = async () => {
    try {
      const response = await api.get('/api/commands/templates');
      setCommandTemplates(response.data);
    } catch (err) {
      console.error('Failed to fetch command templates:', err);
    }
  };

  const fetchCommandHistory = async () => {
    try {
      const response = await api.get('/api/commands/history?limit=20');
      setCommandHistory(response.data);
    } catch (err) {
      console.error('Failed to fetch command history:', err);
    }
  };

  const executeCommand = async () => {
    if (!command.trim()) {
      setError('Please enter a command');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const request: CommandRequest = {
        command: command.trim(),
        parameters,
        background: background === 'true'
      };

      const response = await api.post('/api/commands/execute', request);
      
      if (response.data.status === 'completed') {
        setSuccess(`Command executed successfully in ${response.data.duration?.toFixed(2)}s`);
      } else if (response.data.status === 'failed') {
        setError(`Command failed: ${response.data.error}`);
      } else {
        setSuccess('Command started in background');
      }

      // Refresh command history
      fetchCommandHistory();
      
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to execute command');
    } finally {
      setLoading(false);
    }
  };

  const selectFlow = (flowId: string) => {
    const flow = quickFlows.find(f => f.id === flowId);
    if (flow && flow.commands.length > 0) {
      setCommand(flow.commands[0]);
      setSelectedFlow(flowId);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'running': return 'warning';
      case 'pending': return 'info';
      default: return 'default';
    }
  };

  const formatDuration = (duration?: number) => {
    if (!duration) return '-';
    return `${duration.toFixed(2)}s`;
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
        🎯 Commands
      </Typography>
      <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
        Execute BreadthFlow commands and manage command history
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

      {/* Quick Flows */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            🚀 Quick Flows
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Select a predefined flow to quickly execute common commands
          </Typography>
          
          <Box sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
            gap: 2
          }}>
            {quickFlows.map((flow) => (
              <Card 
                key={flow.id}
                variant="outlined" 
                sx={{ 
                  cursor: 'pointer',
                  border: selectedFlow === flow.id ? '2px solid #1976d2' : '1px solid #e0e0e0',
                  '&:hover': { borderColor: '#1976d2' }
                }}
                onClick={() => selectFlow(flow.id)}
              >
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    {flow.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {flow.description}
                  </Typography>
                </CardContent>
              </Card>
            ))}
          </Box>
        </CardContent>
      </Card>

      {/* Command Execution */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            ⚡ Command Execution
          </Typography>
          
          <Box sx={{ mb: 3 }}>
            <TextField
              fullWidth
              multiline
              rows={3}
              label="Command"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              placeholder="Enter BreadthFlow command (e.g., data fetch --symbols AAPL,MSFT --timeframe 1day)"
              sx={{ mb: 2 }}
            />
            
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2 }}>
              <FormControl>
                <InputLabel>Run in Background</InputLabel>
                <Select
                  value={background}
                  onChange={(e) => setBackground(e.target.value)}
                  size="small"
                  sx={{ minWidth: 150 }}
                >
                  <MenuItem value="false">Foreground</MenuItem>
                  <MenuItem value="true">Background</MenuItem>
                </Select>
              </FormControl>
              
              <Button
                variant="contained"
                startIcon={loading ? <CircularProgress size={20} /> : <PlayArrowIcon />}
                onClick={executeCommand}
                disabled={loading || !command.trim()}
                size="large"
              >
                {loading ? 'Executing...' : 'Execute Command'}
              </Button>
            </Box>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {success && (
            <Alert severity="success" sx={{ mb: 2 }}>
              {success}
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Command Templates */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            📋 Command Templates
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Use these templates as starting points for common operations
          </Typography>
          
          {Object.entries(commandTemplates).map(([category, templates]) => (
            <Accordion key={category} sx={{ mb: 1 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6" sx={{ textTransform: 'capitalize' }}>
                  {category.replace('_', ' ')} Commands
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{
                  display: 'grid',
                  gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' },
                  gap: 2
                }}>
                  {templates.map((template, index) => (
                    <Card key={index} variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom>
                          {template.name}
                        </Typography>
                        <Typography 
                          variant="body2" 
                          sx={{ 
                            fontFamily: 'monospace', 
                            backgroundColor: '#f5f5f5', 
                            p: 1, 
                            borderRadius: 1,
                            mb: 1
                          }}
                        >
                          {template.template}
                        </Typography>
                        <Button
                          size="small"
                          onClick={() => setCommand(template.template)}
                        >
                          Use Template
                        </Button>
                      </CardContent>
                    </Card>
                  ))}
                </Box>
              </AccordionDetails>
            </Accordion>
          ))}
        </CardContent>
      </Card>

      {/* Command History */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
            <HistoryIcon />
            <Typography variant="h5">
              Command History
            </Typography>
            <Button size="small" onClick={fetchCommandHistory}>
              Refresh
            </Button>
          </Box>
          
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Command</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Start Time</TableCell>
                  <TableCell>Duration</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {commandHistory.map((cmd) => (
                  <TableRow key={cmd.command_id}>
                    <TableCell>
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          fontFamily: 'monospace',
                          maxWidth: 300,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis'
                        }}
                      >
                        {cmd.command}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={cmd.status.toUpperCase()} 
                        color={getStatusColor(cmd.status) as any}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      {new Date(cmd.start_time).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      {formatDuration(cmd.duration)}
                    </TableCell>
                    <TableCell>
                      <Button size="small" disabled>
                        View Details
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Container>
  );
};

export default Commands;
