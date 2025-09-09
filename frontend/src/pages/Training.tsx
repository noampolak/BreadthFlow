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
  AccordionDetails,
  Autocomplete,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import HistoryIcon from '@mui/icons-material/History';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import DeleteIcon from '@mui/icons-material/Delete';
import api from '../services/api';
import { useWebSocket } from '../hooks/useWebSocket';

interface TrainingRequest {
  symbols: string[];
  timeframe: string;
  start_date: string;
  end_date: string;
  strategy: string;
  model_type: string;
  parameters?: Record<string, any>;
  test_split: number;
}

interface TrainingResponse {
  training_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  message: string;
  start_time: string;
  estimated_duration?: number;
}

interface TrainingHistory {
  training_id: string;
  symbols: string[];
  timeframe: string;
  strategy: string;
  model_type: string;
  status: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  start_time: string;
  end_time?: string;
  duration?: number;
  error_message?: string;
}

interface ModelInfo {
  model_id: string;
  name: string;
  strategy: string;
  model_type: string;
  symbols: string[];
  timeframe: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  created_at: string;
  last_used?: string;
  is_deployed: boolean;
}

interface TrainingConfig {
  strategies: Array<{id: string; name: string; description: string}>;
  model_types: Array<{id: string; name: string; description: string}>;
  timeframes: Array<{id: string; name: string; description: string}>;
  symbols: string[];
}

const Training: React.FC = () => {
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig | null>(null);
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistory[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [modelToDelete, setModelToDelete] = useState<string | null>(null);
  
  // Training form state
  const [symbols, setSymbols] = useState<string[]>(['AAPL', 'MSFT']);
  const [timeframe, setTimeframe] = useState('1day');
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState('2024-12-31');
  const [strategy, setStrategy] = useState('momentum');
  const [modelType, setModelType] = useState('random_forest');
  const [testSplit, setTestSplit] = useState(0.2);
  
  const { connected, data: wsData } = useWebSocket();

  useEffect(() => {
    fetchTrainingConfig();
    fetchTrainingHistory();
    fetchModels();
  }, []);

  const fetchTrainingConfig = async () => {
    try {
      const response = await api.get('/api/training/configurations');
      setTrainingConfig(response.data);
    } catch (err) {
      console.error('Failed to fetch training configurations:', err);
    }
  };

  const fetchTrainingHistory = async () => {
    try {
      const response = await api.get('/api/training/history?limit=20');
      setTrainingHistory(response.data);
    } catch (err) {
      console.error('Failed to fetch training history:', err);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await api.get('/api/training/models');
      setModels(response.data);
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  };

  const startTraining = async () => {
    if (!symbols.length || !timeframe || !startDate || !endDate || !strategy || !modelType) {
      setError('Please fill in all required fields');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const request: TrainingRequest = {
        symbols,
        timeframe,
        start_date: startDate,
        end_date: endDate,
        strategy,
        model_type: modelType,
        test_split: testSplit
      };

      const response = await api.post('/api/training/start', request);
      
      setSuccess(`Training started successfully! Estimated duration: ${response.data.estimated_duration} minutes`);
      
      // Refresh training history
      fetchTrainingHistory();
      
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start training');
    } finally {
      setLoading(false);
    }
  };

  const deleteModel = async (modelId: string) => {
    try {
      await api.delete(`/api/training/models/${modelId}`);
      setSuccess('Model deleted successfully');
      fetchModels();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete model');
    }
  };

  const handleDeleteClick = (modelId: string) => {
    setModelToDelete(modelId);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = () => {
    if (modelToDelete) {
      deleteModel(modelToDelete);
      setDeleteDialogOpen(false);
      setModelToDelete(null);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'running': return 'warning';
      case 'pending': return 'info';
      case 'cancelled': return 'default';
      default: return 'default';
    }
  };

  const formatDuration = (duration?: number) => {
    if (!duration) return '-';
    const minutes = Math.floor(duration / 60);
    const seconds = Math.floor(duration % 60);
    return `${minutes}m ${seconds}s`;
  };

  const formatMetric = (value?: number) => {
    if (value === undefined || value === null) return '-';
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
        🤖 Training
      </Typography>
      <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
        Train machine learning models for trading strategies
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

      {/* Training Configuration */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            🎯 Training Configuration
          </Typography>
          
          <Box sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' },
            gap: 3,
            mb: 3
          }}>
            <FormControl fullWidth>
              <Autocomplete
                multiple
                options={trainingConfig?.symbols || []}
                value={symbols}
                onChange={(_, newValue) => setSymbols(newValue)}
                renderInput={(params) => (
                  <TextField {...params} label="Symbols" placeholder="Select symbols" />
                )}
              />
            </FormControl>

            <FormControl fullWidth>
              <InputLabel>Timeframe</InputLabel>
              <Select
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value)}
                label="Timeframe"
              >
                {trainingConfig?.timeframes.map((tf) => (
                  <MenuItem key={tf.id} value={tf.id}>
                    {tf.name} - {tf.description}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              label="Start Date"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              InputLabelProps={{ shrink: true }}
            />

            <TextField
              label="End Date"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              InputLabelProps={{ shrink: true }}
            />

            <FormControl fullWidth>
              <InputLabel>Strategy</InputLabel>
              <Select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                label="Strategy"
              >
                {trainingConfig?.strategies.map((s) => (
                  <MenuItem key={s.id} value={s.id}>
                    {s.name} - {s.description}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel>Model Type</InputLabel>
              <Select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                label="Model Type"
              >
                {trainingConfig?.model_types.map((mt) => (
                  <MenuItem key={mt.id} value={mt.id}>
                    {mt.name} - {mt.description}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>

          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2 }}>
            <TextField
              label="Test Split"
              type="number"
              value={testSplit}
              onChange={(e) => setTestSplit(parseFloat(e.target.value))}
              inputProps={{ min: 0.1, max: 0.5, step: 0.1 }}
              sx={{ width: 150 }}
            />
            
            <Button
              variant="contained"
              startIcon={loading ? <CircularProgress size={20} /> : <ModelTrainingIcon />}
              onClick={startTraining}
              disabled={loading}
              size="large"
            >
              {loading ? 'Starting Training...' : 'Start Training'}
            </Button>
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

      {/* Training History */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
            <HistoryIcon />
            <Typography variant="h5">
              Training History
            </Typography>
            <Button size="small" onClick={fetchTrainingHistory}>
              Refresh
            </Button>
          </Box>
          
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Strategy</TableCell>
                  <TableCell>Model Type</TableCell>
                  <TableCell>Symbols</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Accuracy</TableCell>
                  <TableCell>Start Time</TableCell>
                  <TableCell>Duration</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {trainingHistory.map((training) => (
                  <TableRow key={training.training_id}>
                    <TableCell>{training.strategy}</TableCell>
                    <TableCell>{training.model_type}</TableCell>
                    <TableCell>{training.symbols.join(', ')}</TableCell>
                    <TableCell>
                      <Chip 
                        label={training.status.toUpperCase()} 
                        color={getStatusColor(training.status) as any}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{formatMetric(training.accuracy)}</TableCell>
                    <TableCell>
                      {new Date(training.start_time).toLocaleString()}
                    </TableCell>
                    <TableCell>{formatDuration(training.duration)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Trained Models */}
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            📊 Trained Models
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Manage your trained models and view performance metrics
          </Typography>
          
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Model Name</TableCell>
                  <TableCell>Strategy</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Accuracy</TableCell>
                  <TableCell>Precision</TableCell>
                  <TableCell>Recall</TableCell>
                  <TableCell>F1 Score</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {models.map((model) => (
                  <TableRow key={model.model_id}>
                    <TableCell>{model.name}</TableCell>
                    <TableCell>{model.strategy}</TableCell>
                    <TableCell>{model.model_type}</TableCell>
                    <TableCell>{formatMetric(model.accuracy)}</TableCell>
                    <TableCell>{formatMetric(model.precision)}</TableCell>
                    <TableCell>{formatMetric(model.recall)}</TableCell>
                    <TableCell>{formatMetric(model.f1_score)}</TableCell>
                    <TableCell>
                      {new Date(model.created_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      <Button
                        size="small"
                        color="error"
                        startIcon={<DeleteIcon />}
                        onClick={() => handleDeleteClick(model.model_id)}
                      >
                        Delete
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Model</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this model? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default Training;

