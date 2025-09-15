import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

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

test('renders BreadthFlow dashboard', () => {
  render(<App />);
  const dashboardElement = screen.getByText(/BreadthFlow Dashboard/i);
  expect(dashboardElement).toBeInTheDocument();
});
