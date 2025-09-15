import React from 'react';
import { render, screen } from '@testing-library/react';

// Mock react-router-dom completely
jest.mock('react-router-dom', () => ({
  BrowserRouter: ({ children }) => children,
  Routes: ({ children }) => children,
  Route: ({ element }) => element,
  Navigate: ({ to }) => <div>Navigate to {to}</div>,
  Link: ({ to, children, ...props }) => <a href={to} {...props}>{children}</a>,
  useLocation: () => ({ pathname: '/dashboard' }),
}));

// Mock the page components to avoid complex dependencies
jest.mock('./pages/Dashboard', () => {
  return function MockDashboard() {
    return <div>Dashboard Component</div>;
  };
});

jest.mock('./pages/Pipeline', () => {
  return function MockPipeline() {
    return <div>Pipeline Component</div>;
  };
});

jest.mock('./pages/Signals', () => {
  return function MockSignals() {
    return <div>Signals Component</div>;
  };
});

jest.mock('./pages/Infrastructure', () => {
  return function MockInfrastructure() {
    return <div>Infrastructure Component</div>;
  };
});

jest.mock('./pages/Commands', () => {
  return function MockCommands() {
    return <div>Commands Component</div>;
  };
});

jest.mock('./pages/Training', () => {
  return function MockTraining() {
    return <div>Training Component</div>;
  };
});

jest.mock('./pages/Parameters', () => {
  return function MockParameters() {
    return <div>Parameters Component</div>;
  };
});

// Import App after mocking
import App from './App';

test('renders BreadthFlow dashboard', () => {
  render(<App />);
  const dashboardElement = screen.getByText(/BreadthFlow Dashboard/i);
  expect(dashboardElement).toBeInTheDocument();
});
