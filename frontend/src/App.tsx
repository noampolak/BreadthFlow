import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link, useLocation } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box, AppBar, Toolbar, Typography, Button, Container } from '@mui/material';
import Dashboard from './pages/Dashboard';
import Pipeline from './pages/Pipeline';
import Signals from './pages/Signals';
import Infrastructure from './pages/Infrastructure';
import Commands from './pages/Commands';
import Training from './pages/Training';
import Parameters from './pages/Parameters';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h3: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          borderRadius: '12px',
        },
      },
    },
  },
});

const Navigation: React.FC = () => {
  const location = useLocation();
  
  const navItems = [
    { path: '/dashboard', label: 'Dashboard', icon: '📊' },
    { path: '/pipeline', label: 'Pipeline', icon: '🎮' },
    { path: '/signals', label: 'Signals', icon: '📈' },
    { path: '/commands', label: 'Commands', icon: '🎯' },
    { path: '/training', label: 'Training', icon: '🤖' },
    { path: '/parameters', label: 'Parameters', icon: '⚙️' },
    { path: '/infrastructure', label: 'Infrastructure', icon: '🏗️' },
  ];

  return (
    <AppBar position="static" elevation={0} sx={{ backgroundColor: 'white', borderBottom: '1px solid #e0e0e0' }}>
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1, color: '#1976d2', fontWeight: 'bold' }}>
          🚀 BreadthFlow Dashboard v2.0
        </Typography>
        {navItems.map((item) => (
          <Button
            key={item.path}
            component={Link}
            to={item.path}
            color="inherit"
            sx={{
              color: location.pathname === item.path ? '#1976d2' : '#666',
              mr: 2,
              fontWeight: location.pathname === item.path ? 'bold' : 'normal',
              '&:hover': {
                backgroundColor: 'rgba(25, 118, 210, 0.04)',
              }
            }}
          >
            {item.icon} {item.label}
          </Button>
        ))}
      </Toolbar>
    </AppBar>
  );
};

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ flexGrow: 1 }}>
          <Navigation />
          
          <Container maxWidth={false} sx={{ mt: 0 }}>
                    <Routes>
                      <Route path="/" element={<Navigate to="/dashboard" replace />} />
                      <Route path="/dashboard" element={<Dashboard />} />
                      <Route path="/pipeline" element={<Pipeline />} />
                      <Route path="/signals" element={<Signals />} />
                      <Route path="/commands" element={<Commands />} />
                      <Route path="/training" element={<Training />} />
                      <Route path="/parameters" element={<Parameters />} />
                      <Route path="/infrastructure" element={<Infrastructure />} />
                    </Routes>
          </Container>
        </Box>
      </Router>
    </ThemeProvider>
  );
};

export default App;