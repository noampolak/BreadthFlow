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
  Switch,
  FormControlLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SettingsIcon from '@mui/icons-material/Settings';
import HistoryIcon from '@mui/icons-material/History';
import DownloadIcon from '@mui/icons-material/Download';
import UploadIcon from '@mui/icons-material/Upload';
import RestoreIcon from '@mui/icons-material/Restore';
import api from '../services/api';
import { useWebSocket } from '../hooks/useWebSocket';

interface ParameterValue {
  name: string;
  value: any;
  default_value: any;
  description: string;
  parameter_type: 'string' | 'integer' | 'float' | 'boolean' | 'select' | 'multiselect';
  options?: string[];
  min_value?: number;
  max_value?: number;
  required: boolean;
  last_modified?: string;
}

interface ParameterGroup {
  group_name: string;
  display_name: string;
  description: string;
  parameters: ParameterValue[];
  last_modified?: string;
}

interface ParameterUpdate {
  group_name: string;
  parameter_name: string;
  value: any;
}

interface ParameterHistory {
  history_id: string;
  group_name: string;
  parameter_name: string;
  old_value: any;
  new_value: any;
  changed_by: string;
  change_time: string;
  change_reason?: string;
}

const Parameters: React.FC = () => {
  const [parameterGroups, setParameterGroups] = useState<ParameterGroup[]>([]);
  const [parameterHistory, setParameterHistory] = useState<ParameterHistory[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(0);
  const [pendingUpdates, setPendingUpdates] = useState<ParameterUpdate[]>([]);
  const [resetDialogOpen, setResetDialogOpen] = useState(false);
  const [groupToReset, setGroupToReset] = useState<string | null>(null);
  
  const { connected, data: wsData } = useWebSocket();

  useEffect(() => {
    fetchParameterGroups();
    fetchParameterHistory();
  }, []);

  const fetchParameterGroups = async () => {
    try {
      const response = await api.get('/api/parameters/groups');
      setParameterGroups(response.data);
    } catch (err) {
      console.error('Failed to fetch parameter groups:', err);
    }
  };

  const fetchParameterHistory = async () => {
    try {
      const response = await api.get('/api/parameters/history?limit=50');
      setParameterHistory(response.data);
    } catch (err) {
      console.error('Failed to fetch parameter history:', err);
    }
  };

  const updateParameter = (groupName: string, paramName: string, value: any) => {
    const existingUpdateIndex = pendingUpdates.findIndex(
      update => update.group_name === groupName && update.parameter_name === paramName
    );

    const update: ParameterUpdate = {
      group_name: groupName,
      parameter_name: paramName,
      value
    };

    if (existingUpdateIndex >= 0) {
      const newUpdates = [...pendingUpdates];
      newUpdates[existingUpdateIndex] = update;
      setPendingUpdates(newUpdates);
    } else {
      setPendingUpdates([...pendingUpdates, update]);
    }
  };

  const saveUpdates = async () => {
    if (pendingUpdates.length === 0) {
      setError('No changes to save');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      await api.put('/api/parameters/update', pendingUpdates);
      setSuccess(`Successfully updated ${pendingUpdates.length} parameters`);
      setPendingUpdates([]);
      fetchParameterGroups();
      fetchParameterHistory();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update parameters');
    } finally {
      setLoading(false);
    }
  };

  const resetGroup = async (groupName: string) => {
    try {
      await api.post(`/api/parameters/reset/${groupName}`);
      setSuccess(`Parameter group '${groupName}' reset to defaults`);
      fetchParameterGroups();
      fetchParameterHistory();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to reset parameters');
    }
  };

  const exportParameters = async () => {
    try {
      const response = await api.get('/api/parameters/export?format=json');
      const dataStr = JSON.stringify(response.data, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `breadthflow-parameters-${new Date().toISOString().split('T')[0]}.json`;
      link.click();
      URL.revokeObjectURL(url);
      setSuccess('Parameters exported successfully');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to export parameters');
    }
  };

  const renderParameterInput = (group: ParameterGroup, param: ParameterValue) => {
    const currentValue = pendingUpdates.find(
      update => update.group_name === group.group_name && update.parameter_name === param.name
    )?.value ?? param.value;

    switch (param.parameter_type) {
      case 'boolean':
        return (
          <FormControlLabel
            control={
              <Switch
                checked={currentValue}
                onChange={(e) => updateParameter(group.group_name, param.name, e.target.checked)}
              />
            }
            label={currentValue ? 'Enabled' : 'Disabled'}
          />
        );

      case 'select':
        return (
          <FormControl fullWidth size="small">
            <Select
              value={currentValue}
              onChange={(e) => updateParameter(group.group_name, param.name, e.target.value)}
            >
              {param.options?.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        );

      case 'multiselect':
        return (
          <Autocomplete
            multiple
            options={param.options || []}
            value={currentValue}
            onChange={(_, newValue) => updateParameter(group.group_name, param.name, newValue)}
            renderInput={(params) => (
              <TextField {...params} size="small" placeholder="Select options" />
            )}
          />
        );

      case 'integer':
        return (
          <TextField
            type="number"
            value={currentValue}
            onChange={(e) => updateParameter(group.group_name, param.name, parseInt(e.target.value))}
            inputProps={{ min: param.min_value, max: param.max_value }}
            size="small"
            fullWidth
          />
        );

      case 'float':
        return (
          <TextField
            type="number"
            value={currentValue}
            onChange={(e) => updateParameter(group.group_name, param.name, parseFloat(e.target.value))}
            inputProps={{ min: param.min_value, max: param.max_value, step: 0.01 }}
            size="small"
            fullWidth
          />
        );

      default: // string
        return (
          <TextField
            value={currentValue}
            onChange={(e) => updateParameter(group.group_name, param.name, e.target.value)}
            size="small"
            fullWidth
          />
        );
    }
  };

  const formatValue = (value: any) => {
    if (Array.isArray(value)) {
      return value.join(', ');
    }
    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No';
    }
    return String(value);
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
        ⚙️ Parameters
      </Typography>
      <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
        Configure system parameters and settings
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

      {/* Action Buttons */}
      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <Button
          variant="contained"
          startIcon={loading ? <CircularProgress size={20} /> : <SettingsIcon />}
          onClick={saveUpdates}
          disabled={loading || pendingUpdates.length === 0}
        >
          {loading ? 'Saving...' : `Save Changes (${pendingUpdates.length})`}
        </Button>
        
        <Button
          variant="outlined"
          startIcon={<DownloadIcon />}
          onClick={exportParameters}
        >
          Export
        </Button>
        
        <Button
          variant="outlined"
          startIcon={<UploadIcon />}
          disabled
        >
          Import
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

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
          <Tab label="Configuration" />
          <Tab label="History" />
        </Tabs>
      </Box>

      {/* Configuration Tab */}
      {activeTab === 0 && (
        <Box>
          {parameterGroups.map((group) => (
            <Accordion key={group.group_name} sx={{ mb: 2 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                  <Typography variant="h6">{group.display_name}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {group.description}
                  </Typography>
                  <Box sx={{ flexGrow: 1 }} />
                  <Button
                    size="small"
                    startIcon={<RestoreIcon />}
                    onClick={(e) => {
                      e.stopPropagation();
                      setGroupToReset(group.group_name);
                      setResetDialogOpen(true);
                    }}
                  >
                    Reset
                  </Button>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Parameter</TableCell>
                        <TableCell>Description</TableCell>
                        <TableCell>Current Value</TableCell>
                        <TableCell>Default Value</TableCell>
                        <TableCell>Type</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {group.parameters.map((param) => (
                        <TableRow key={param.name}>
                          <TableCell>
                            <Typography variant="subtitle2">
                              {param.name}
                              {param.required && <Chip label="Required" size="small" sx={{ ml: 1 }} />}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" color="text.secondary">
                              {param.description}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            {renderParameterInput(group, param)}
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" color="text.secondary">
                              {formatValue(param.default_value)}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip label={param.parameter_type} size="small" />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      )}

      {/* History Tab */}
      {activeTab === 1 && (
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
              <HistoryIcon />
              <Typography variant="h5">
                Parameter History
              </Typography>
              <Button size="small" onClick={fetchParameterHistory}>
                Refresh
              </Button>
            </Box>
            
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Group</TableCell>
                    <TableCell>Parameter</TableCell>
                    <TableCell>Old Value</TableCell>
                    <TableCell>New Value</TableCell>
                    <TableCell>Changed By</TableCell>
                    <TableCell>Time</TableCell>
                    <TableCell>Reason</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {parameterHistory.map((history) => (
                    <TableRow key={history.history_id}>
                      <TableCell>{history.group_name}</TableCell>
                      <TableCell>{history.parameter_name}</TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {formatValue(history.old_value)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatValue(history.new_value)}
                        </Typography>
                      </TableCell>
                      <TableCell>{history.changed_by}</TableCell>
                      <TableCell>
                        {new Date(history.change_time).toLocaleString()}
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {history.change_reason || '-'}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* Reset Confirmation Dialog */}
      <Dialog open={resetDialogOpen} onClose={() => setResetDialogOpen(false)}>
        <DialogTitle>Reset Parameters</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to reset all parameters in the "{groupToReset}" group to their default values? 
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={() => {
              if (groupToReset) {
                resetGroup(groupToReset);
                setResetDialogOpen(false);
                setGroupToReset(null);
              }
            }} 
            color="error" 
            variant="contained"
          >
            Reset
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default Parameters;

