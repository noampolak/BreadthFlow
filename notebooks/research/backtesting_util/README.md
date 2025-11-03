# Backtesting Utilities

Modular Python package for ultra-optimized backtesting with timing fixes.

## Structure

```
backtesting_util/
├── __init__.py          # Clean imports
├── config.py            # Configuration constants and paths
├── helpers.py           # Helper functions (trading days, momentum, prices)
├── backtest_engine.py   # Core backtesting function
├── data_loader.py       # Load models, stock data, features
├── batch_processor.py   # Batch processing for multiple models
├── main.py              # Main orchestrator function
└── README.md            # This file
```

## Quick Start

### Simple Usage (All Models)

```python
from backtesting_util import run_ultra_optimized_backtesting

# Run complete pipeline
results_df = run_ultra_optimized_backtesting(
    max_positions=20,
    verbose=True,
    trade_when_positions_zero=True
)
```

### Custom Usage

```python
from backtesting_util import (
    load_models_from_log,
    load_stock_data,
    load_features_cache,
    get_test_period,
    run_all_models
)

# Load data
models_dict = load_models_from_log()
stock_data = load_stock_data()
features_cache = load_features_cache()

# Define test period
test_start, test_end = get_test_period(stock_data, test_split=0.6)

# Run backtests
results_df = run_all_models(
    all_models=models_dict['all_models'],
    features_cache=features_cache,
    stock_data=stock_data,
    test_start_date=test_start,
    test_end_date=test_end,
    max_positions=20
)
```

### Single Model Backtest

```python
from backtesting_util import backtest_model_ultra_optimized

portfolio_df, trades_df, stats = backtest_model_ultra_optimized(
    model=my_model,
    features_df=features,
    stock_data=stock_data,
    target_window=5,
    start_date='2022-01-01',
    end_date='2024-12-31',
    max_positions=20,
    trade_when_positions_zero=True
)
```

## Key Features

### Timing Fixes Applied

1. **Momentum Calculation**: Uses previous trading day (not calendar day)
2. **Entry Prices**: Uses current day's OPEN price (not previous close)
3. **Exit Dates**: Uses trading days (k trading days ahead, not calendar days)
4. **Look-ahead Bias**: Completely removed

### Trading Logic

- **Trade Frequency**: Only when `positions == 0` (if `trade_when_positions_zero=True`)
- **Position Sizing**: Equal allocation across selected stocks
- **Transaction Costs**: Applied on both entry and exit
- **Cash Buffer**: Maintains configurable cash buffer

### Model Support

- RandomForest models
- XGBoost models
- Custom model types (if they follow sklearn API)

## Configuration

Default parameters can be modified in `config.py`:

```python
DEFAULT_INITIAL_CAPITAL = 100000
DEFAULT_TRANSACTION_COST = 0.003  # 0.3%
DEFAULT_TOP_PERCENT = 0.2  # Top/bottom 20%
DEFAULT_MAX_POSITIONS = 20
DEFAULT_CASH_BUFFER = 0.05  # 5%
DEFAULT_TEST_SPLIT = 0.6  # 60% train, 40% test
```

## Output Files

Results are saved to `data/research/backtesting/ultra_optimized_results/`:

- `{model_key}_portfolio_ultra_optimized.csv`: Daily portfolio values
- `{model_key}_trades_ultra_optimized.csv`: All executed trades
- `ultra_optimized_results_summary.csv`: Summary of all models

## Dependencies

- pandas
- numpy
- tqdm
- scikit-learn or xgboost (for models)
