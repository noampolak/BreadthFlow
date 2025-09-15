import React from 'react';
import { createRoot } from 'react-dom/client';

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

test('renders without crashing', () => {
  const div = document.createElement('div');
  const root = createRoot(div);
  
  // Just test that the component renders without crashing
  expect(() => {
    root.render(<App />);
  }).not.toThrow();
});
