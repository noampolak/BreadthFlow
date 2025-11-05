# Fetch Data Utilities

Modular Python package for downloading and managing financial data with automatic versioning.

## Features

- **Stock Prices**: Download S&P 500 stock OHLCV data from Yahoo Finance
- **SEC Fundamentals**: Download quarterly fundamental data from SEC EDGAR
- **S&P 500 Index**: Download S&P 500 index data for baseline comparison
- **Automatic Versioning**: Existing files are backed up with version numbers
- **Latest Copies**: Automatically maintains latest data in `stock_data_latest` directory

## Structure

```
fetch_data_util/
├── __init__.py          # Package exports
├── config.py            # Configuration constants
├── versioning.py        # File versioning utilities
├── stock_prices.py      # Stock price download
├── sec_data.py          # SEC fundamentals download
├── sp500_index.py       # S&P 500 index download
└── README.md            # This file
```

## Quick Start

### Stock Prices

```python
from fetch_data_util import fetch_stock_prices

# Download stock prices
stock_data, metadata = fetch_stock_prices(
    start_date='2014-01-01',
    end_date='2025-01-31',
    verbose=True
)

print(f"Downloaded {len(stock_data):,} records")
print(f"Saved to: {metadata['filepath']}")
print(f"Latest copy: {metadata['latest_filepath']}")
```

### SEC Fundamentals

```python
from fetch_data_util import fetch_sec_fundamentals
import pandas as pd

# Load price data first (needed for SEC calculations)
price_data = pd.read_pickle('data/research/stock_data_latest/sp500_stock_data_latest.pkl')

# Get tickers
tickers = price_data['symbol'].unique().tolist()

# Download SEC fundamentals with incremental saving and resume support
fundamentals, metadata = fetch_sec_fundamentals(
    tickers=tickers,
    price_df=price_data,
    start_date='2014-01-01',
    end_date='2025-01-31',
    verbose=True,
    resume=True  # Enable resume - will continue from checkpoint if interrupted
)
```

**Resume Feature:**
- Automatically saves progress after each ticker
- If interrupted, can resume from checkpoint on next run
- Checkpoint files: `*_in_progress.pkl` in data directory
- Latest checkpoint: `stock_data_latest/sec_fundamentals_latest_in_progress.pkl`

### S&P 500 Index

```python
from fetch_data_util import fetch_sp500_index

# Download S&P 500 index data
sp500_data, metadata = fetch_sp500_index(
    start_date='2020-12-30',
    end_date='2025-01-31',
    verbose=True
)
```

## File Versioning

The package automatically handles file versioning:

1. **New Downloads**: Files are saved with format:
   ```
   {prefix}_{start_year}-{end_year}_download_date_{DD_mm_YYYY}.pkl
   ```
   Example: `stocks_prices_2014-2025_download_date_31_01_2025.pkl`

2. **Existing Files**: If a file with the same name exists, it's backed up as:
   ```
   {original_filename}_v{version}.pkl
   ```
   Example: `stocks_prices_2014-2025_download_date_31_01_2025_v1.pkl`

3. **Latest Copies**: Each download also creates/updates files in `stock_data_latest/`:
   - `sp500_stock_data_latest.pkl`
   - `sec_fundamentals_latest.pkl`
   - `sp500_index_data_latest.pkl`

## Configuration

Default settings can be modified in `config.py`:

```python
DATA_DIR = 'data/research'
STOCK_PRICES_DIR = 'data/research/stock_prices'
SEC_DATA_DIR = 'data/research/sec_data'
SP500_INDEX_DIR = 'data/research/sp500_index'
LATEST_DIR = 'data/research/stock_data_latest'

YAHOO_RATE_LIMIT = 0.1  # seconds between Yahoo Finance requests
SEC_RATE_LIMIT = 0.08   # seconds between SEC API requests
```

## Output Files

### Stock Prices
- **Versioned**: `data/research/stock_prices/stocks_prices_{years}_download_date_{date}.pkl`
- **Latest**: `data/research/stock_data_latest/sp500_stock_data_latest.pkl`
- **CSV**: Also saves CSV version for easy inspection

### SEC Fundamentals
- **Versioned**: `data/research/sec_data/sec_fundamentals_{years}_download_date_{date}.pkl`
- **Latest**: `data/research/stock_data_latest/sec_fundamentals_latest.pkl`

### S&P 500 Index
- **Versioned**: `data/research/sp500_index/sp500_index_{years}_download_date_{date}.pkl`
- **Latest**: `data/research/stock_data_latest/sp500_index_data_latest.pkl`

## Metadata

Each download function returns metadata containing:

```python
{
    'timestamp': '20250131_143022',
    'filepath': 'path/to/versioned/file.pkl',
    'latest_filepath': 'path/to/latest/file.pkl',
    'backup_filepath': 'path/to/backup/file_v1.pkl',  # or None
    'successful_tickers': [...],
    'failed_tickers': [...],
    'date_range': {
        'start': '2014-01-01',
        'end': '2025-01-31'
    },
    'data_info': {
        'total_records': 123456,
        'unique_symbols': 285,
        ...
    }
}
```

## Dependencies

- pandas
- numpy
- yfinance (for Yahoo Finance data)
- requests (for SEC API)
- tqdm (for progress bars)

## Notes

- **SEC API**: Requires proper User-Agent header (configured in `config.py`)
- **Rate Limiting**: Built-in rate limiting to avoid API throttling
- **Caching**: SEC data uses local cache in `./sec_cache/` directory
- **Error Handling**: Failed tickers are logged but don't stop the download process

