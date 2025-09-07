// Dashboard Types
export interface DashboardStats {
  total_runs: number;
  successful_runs: number;
  failed_runs: number;
  recent_runs_24h: number;
  average_duration: number;
  success_rate: number;
  last_updated: string;
}

export interface RecentRun {
  run_id: string;
  command: string;
  status: string;
  start_time: string;
  end_time?: string;
  duration?: number;
  error_message?: string;
}

export interface DashboardSummary {
  stats: DashboardStats;
  recent_runs: RecentRun[];
  last_updated: string;
}

// Pipeline Types
export interface PipelineConfig {
  mode: string;
  interval: string;
  timeframe: string;
  symbols?: string;
  data_source: string;
}

export interface PipelineRun {
  run_id: string;
  command: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  start_time: string;
  end_time?: string;
  duration?: number;
  error_message?: string;
  run_metadata?: any;
}

export interface PipelineStatus {
  total_runs: number;
  successful_runs: number;
  failed_runs: number;
  running_runs: number;
  success_rate: number;
  average_duration: number;
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'heartbeat' | 'pipeline_update' | 'signal_update' | 'dashboard_update' | 'infrastructure_update' | 'pipeline_status_update' | 'signals_update';
  data?: any;
  timestamp: string;
  pipeline_id?: string;
  status?: string;
}
