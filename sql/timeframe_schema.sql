-- Timeframe-Enhanced Database Schema
-- This file contains database schema enhancements to support multi-timeframe operations
-- while maintaining backward compatibility with existing tables.

-- Create timeframe_configs table for storing timeframe-specific configurations
CREATE TABLE IF NOT EXISTS timeframe_configs (
    id SERIAL PRIMARY KEY,
    timeframe VARCHAR(10) NOT NULL UNIQUE,
    display_name VARCHAR(50) NOT NULL,
    description TEXT,
    config_json JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on timeframe for faster lookups
CREATE INDEX IF NOT EXISTS idx_timeframe_configs_timeframe ON timeframe_configs(timeframe);
CREATE INDEX IF NOT EXISTS idx_timeframe_configs_active ON timeframe_configs(is_active);

-- Insert default timeframe configurations
INSERT INTO timeframe_configs (timeframe, display_name, description, config_json) 
VALUES 
    ('1day', 'Daily', 'Traditional daily trading with end-of-day bars', 
     '{"ma_short_period": 20, "ma_long_period": 50, "rsi_period": 14, "commission_rate": 0.001, "max_position_size": 0.1}'::jsonb),
    ('1hour', '1 Hour', 'Intraday trading with hourly bars',
     '{"ma_short_period": 12, "ma_long_period": 24, "rsi_period": 14, "commission_rate": 0.0015, "max_position_size": 0.08}'::jsonb),
    ('15min', '15 Minutes', 'Medium frequency trading with 15-minute bars',
     '{"ma_short_period": 8, "ma_long_period": 16, "rsi_period": 14, "commission_rate": 0.002, "max_position_size": 0.06}'::jsonb),
    ('5min', '5 Minutes', 'High frequency trading with 5-minute bars',
     '{"ma_short_period": 6, "ma_long_period": 12, "rsi_period": 10, "commission_rate": 0.0025, "max_position_size": 0.05}'::jsonb),
    ('1min', '1 Minute', 'Ultra-high frequency trading with 1-minute bars',
     '{"ma_short_period": 5, "ma_long_period": 10, "rsi_period": 8, "commission_rate": 0.003, "max_position_size": 0.03}'::jsonb)
ON CONFLICT (timeframe) DO NOTHING;

-- Enhance existing pipeline_runs table with timeframe support
-- Add timeframe column if it doesn't exist
ALTER TABLE pipeline_runs 
ADD COLUMN IF NOT EXISTS timeframe VARCHAR(10) DEFAULT '1day';

-- Add data_source column for tracking different data sources
ALTER TABLE pipeline_runs 
ADD COLUMN IF NOT EXISTS data_source VARCHAR(50) DEFAULT 'yfinance';

-- Add symbols_count for quick statistics
ALTER TABLE pipeline_runs 
ADD COLUMN IF NOT EXISTS symbols_count INTEGER;

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_timeframe ON pipeline_runs(timeframe);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_data_source ON pipeline_runs(data_source);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_command_timeframe ON pipeline_runs(command, timeframe);

-- Create timeframe_data_summary table for tracking data availability by timeframe
CREATE TABLE IF NOT EXISTS timeframe_data_summary (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    data_source VARCHAR(50) NOT NULL DEFAULT 'yfinance',
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    records_count INTEGER NOT NULL DEFAULT 0,
    file_path TEXT,
    file_size_bytes BIGINT,
    storage_location VARCHAR(20) DEFAULT 'minio',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, data_source, start_date, end_date)
);

-- Create indexes for timeframe_data_summary
CREATE INDEX IF NOT EXISTS idx_timeframe_data_symbol_tf ON timeframe_data_summary(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_timeframe_data_timeframe ON timeframe_data_summary(timeframe);
CREATE INDEX IF NOT EXISTS idx_timeframe_data_date_range ON timeframe_data_summary(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_timeframe_data_updated ON timeframe_data_summary(updated_at);

-- Create signals_metadata table for tracking signal generation by timeframe
CREATE TABLE IF NOT EXISTS signals_metadata (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL, -- BUY, SELL, HOLD
    confidence DECIMAL(5,3),
    strength VARCHAR(20),
    signal_date DATE NOT NULL,
    price DECIMAL(12,4),
    volume BIGINT,
    indicators JSONB, -- Technical indicators used
    file_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

-- Create indexes for signals_metadata
CREATE INDEX IF NOT EXISTS idx_signals_run_id ON signals_metadata(run_id);
CREATE INDEX IF NOT EXISTS idx_signals_timeframe ON signals_metadata(timeframe);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_tf ON signals_metadata(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_signals_type ON signals_metadata(signal_type);
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals_metadata(signal_date);
CREATE INDEX IF NOT EXISTS idx_signals_confidence ON signals_metadata(confidence);

-- Create backtest_results table for tracking backtesting by timeframe
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    symbols TEXT[], -- Array of symbols tested
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(15,2) NOT NULL,
    final_capital DECIMAL(15,2),
    total_return DECIMAL(8,6),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,6),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5,3),
    avg_win DECIMAL(12,4),
    avg_loss DECIMAL(12,4),
    profit_factor DECIMAL(8,4),
    execution_params JSONB, -- Commission, slippage, etc.
    performance_metrics JSONB, -- Additional metrics
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

-- Create indexes for backtest_results
CREATE INDEX IF NOT EXISTS idx_backtest_run_id ON backtest_results(run_id);
CREATE INDEX IF NOT EXISTS idx_backtest_timeframe ON backtest_results(timeframe);
CREATE INDEX IF NOT EXISTS idx_backtest_date_range ON backtest_results(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_backtest_return ON backtest_results(total_return);
CREATE INDEX IF NOT EXISTS idx_backtest_sharpe ON backtest_results(sharpe_ratio);

-- Create timeframe_performance_stats view for dashboard analytics
CREATE OR REPLACE VIEW timeframe_performance_stats AS
SELECT 
    p.timeframe,
    tc.display_name as timeframe_display,
    COUNT(*) as total_runs,
    COUNT(CASE WHEN p.status = 'completed' THEN 1 END) as successful_runs,
    ROUND(
        COUNT(CASE WHEN p.status = 'completed' THEN 1 END) * 100.0 / COUNT(*), 
        2
    ) as success_rate,
    AVG(p.duration) as avg_duration,
    MIN(p.start_time) as first_run,
    MAX(p.start_time) as last_run,
    COUNT(DISTINCT DATE(p.start_time)) as active_days
FROM pipeline_runs p
LEFT JOIN timeframe_configs tc ON p.timeframe = tc.timeframe
WHERE p.timeframe IS NOT NULL
GROUP BY p.timeframe, tc.display_name, tc.id
ORDER BY tc.id;

-- Create signals_summary_by_timeframe view
CREATE OR REPLACE VIEW signals_summary_by_timeframe AS
SELECT 
    timeframe,
    signal_date,
    COUNT(*) as total_signals,
    COUNT(CASE WHEN signal_type = 'BUY' THEN 1 END) as buy_signals,
    COUNT(CASE WHEN signal_type = 'SELL' THEN 1 END) as sell_signals,
    COUNT(CASE WHEN signal_type = 'HOLD' THEN 1 END) as hold_signals,
    AVG(confidence) as avg_confidence,
    COUNT(DISTINCT symbol) as unique_symbols
FROM signals_metadata
GROUP BY timeframe, signal_date
ORDER BY signal_date DESC, timeframe;

-- Create data_availability_by_timeframe view
CREATE OR REPLACE VIEW data_availability_by_timeframe AS
SELECT 
    timeframe,
    data_source,
    COUNT(DISTINCT symbol) as unique_symbols,
    SUM(records_count) as total_records,
    MIN(start_date) as earliest_date,
    MAX(end_date) as latest_date,
    COUNT(*) as file_count,
    SUM(file_size_bytes) as total_size_bytes,
    ROUND(SUM(file_size_bytes) / (1024.0 * 1024.0), 2) as total_size_mb
FROM timeframe_data_summary
GROUP BY timeframe, data_source
ORDER BY timeframe, data_source;

-- Create function to update timeframe_data_summary when new data is stored
CREATE OR REPLACE FUNCTION update_timeframe_data_summary(
    p_symbol VARCHAR(20),
    p_timeframe VARCHAR(10),
    p_data_source VARCHAR(50),
    p_start_date DATE,
    p_end_date DATE,
    p_records_count INTEGER,
    p_file_path TEXT,
    p_file_size_bytes BIGINT
) RETURNS VOID AS $$
BEGIN
    INSERT INTO timeframe_data_summary (
        symbol, timeframe, data_source, start_date, end_date,
        records_count, file_path, file_size_bytes, updated_at
    ) VALUES (
        p_symbol, p_timeframe, p_data_source, p_start_date, p_end_date,
        p_records_count, p_file_path, p_file_size_bytes, CURRENT_TIMESTAMP
    )
    ON CONFLICT (symbol, timeframe, data_source, start_date, end_date)
    DO UPDATE SET
        records_count = EXCLUDED.records_count,
        file_path = EXCLUDED.file_path,
        file_size_bytes = EXCLUDED.file_size_bytes,
        updated_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Create function to log signal generation
CREATE OR REPLACE FUNCTION log_signal_generation(
    p_run_id VARCHAR(255),
    p_timeframe VARCHAR(10),
    p_signals JSONB
) RETURNS INTEGER AS $$
DECLARE
    signal_record JSONB;
    inserted_count INTEGER := 0;
BEGIN
    -- Loop through signals array
    FOR signal_record IN SELECT * FROM jsonb_array_elements(p_signals)
    LOOP
        INSERT INTO signals_metadata (
            run_id, timeframe, symbol, signal_type, confidence, 
            strength, signal_date, price, volume, indicators
        ) VALUES (
            p_run_id,
            p_timeframe,
            signal_record->>'symbol',
            signal_record->>'signal_type',
            (signal_record->>'confidence')::DECIMAL,
            signal_record->>'strength',
            (signal_record->>'date')::DATE,
            (signal_record->>'close')::DECIMAL,
            (signal_record->>'volume')::BIGINT,
            signal_record->'indicators'
        );
        inserted_count := inserted_count + 1;
    END LOOP;
    
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to log backtest results
CREATE OR REPLACE FUNCTION log_backtest_results(
    p_run_id VARCHAR(255),
    p_timeframe VARCHAR(10),
    p_symbols TEXT[],
    p_start_date DATE,
    p_end_date DATE,
    p_initial_capital DECIMAL,
    p_results JSONB
) RETURNS INTEGER AS $$
BEGIN
    INSERT INTO backtest_results (
        run_id, timeframe, symbols, start_date, end_date, initial_capital,
        final_capital, total_return, sharpe_ratio, max_drawdown,
        total_trades, winning_trades, losing_trades, win_rate,
        avg_win, avg_loss, profit_factor, execution_params, performance_metrics
    ) VALUES (
        p_run_id,
        p_timeframe,
        p_symbols,
        p_start_date,
        p_end_date,
        p_initial_capital,
        (p_results->>'final_capital')::DECIMAL,
        (p_results->>'total_return')::DECIMAL,
        (p_results->>'sharpe_ratio')::DECIMAL,
        (p_results->>'max_drawdown')::DECIMAL,
        (p_results->>'total_trades')::INTEGER,
        (p_results->>'winning_trades')::INTEGER,
        (p_results->>'losing_trades')::INTEGER,
        (p_results->>'win_rate')::DECIMAL,
        (p_results->>'avg_win')::DECIMAL,
        (p_results->>'avg_loss')::DECIMAL,
        (p_results->>'profit_factor')::DECIMAL,
        p_results->'execution_params',
        p_results->'performance_metrics'
    );
    
    RETURN 1;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add triggers for updated_at columns
DROP TRIGGER IF EXISTS update_timeframe_configs_updated_at ON timeframe_configs;
CREATE TRIGGER update_timeframe_configs_updated_at
    BEFORE UPDATE ON timeframe_configs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_timeframe_data_summary_updated_at ON timeframe_data_summary;
CREATE TRIGGER update_timeframe_data_summary_updated_at
    BEFORE UPDATE ON timeframe_data_summary
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed for your security model)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO pipeline_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO pipeline_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO pipeline_user;

-- Add helpful comments
COMMENT ON TABLE timeframe_configs IS 'Stores configuration parameters for different timeframes';
COMMENT ON TABLE timeframe_data_summary IS 'Tracks data availability and metadata by symbol and timeframe';
COMMENT ON TABLE signals_metadata IS 'Stores detailed signal generation metadata with timeframe tracking';
COMMENT ON TABLE backtest_results IS 'Stores comprehensive backtesting results by timeframe';
COMMENT ON VIEW timeframe_performance_stats IS 'Aggregated performance statistics by timeframe for dashboard';
COMMENT ON VIEW signals_summary_by_timeframe IS 'Daily signal summaries grouped by timeframe';
COMMENT ON VIEW data_availability_by_timeframe IS 'Data availability overview by timeframe and source';

-- Create indexes for performance optimization on JSONB columns
CREATE INDEX IF NOT EXISTS idx_timeframe_configs_config_gin ON timeframe_configs USING GIN (config_json);
CREATE INDEX IF NOT EXISTS idx_signals_indicators_gin ON signals_metadata USING GIN (indicators);
CREATE INDEX IF NOT EXISTS idx_backtest_execution_params_gin ON backtest_results USING GIN (execution_params);
CREATE INDEX IF NOT EXISTS idx_backtest_performance_metrics_gin ON backtest_results USING GIN (performance_metrics);
