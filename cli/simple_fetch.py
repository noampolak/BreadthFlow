#!/usr/bin/env python3
"""
Simple data fetching without Spark - demonstrates container-native approach.
"""

import sys
import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import logging

# Add the jobs directory to Python path
sys.path.insert(0, '/opt/bitnami/spark/jobs')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_data_simple(symbols, start_date, end_date):
    """Fetch data using yfinance without Spark."""
    
    logger.info(f"🚀 Starting simple data fetch for {len(symbols)} symbols")
    logger.info(f"📅 Period: {start_date} to {end_date}")
    
    results = []
    failed_symbols = []
    
    for symbol in symbols:
        try:
            logger.info(f"📊 Fetching data for {symbol}...")
            
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty:
                # Add symbol column
                data['symbol'] = symbol
                data.reset_index(inplace=True)
                
                results.append(data)
                logger.info(f"✅ {symbol}: {len(data)} records")
            else:
                logger.warning(f"⚠️ {symbol}: No data available")
                failed_symbols.append(symbol)
                
        except Exception as e:
            logger.error(f"❌ {symbol}: {str(e)}")
            failed_symbols.append(symbol)
    
    if results:
        # Combine all data
        combined_data = pd.concat(results, ignore_index=True)
        logger.info(f"✅ Total records fetched: {len(combined_data)}")
        logger.info(f"📊 Sample data:")
        logger.info(combined_data.head())
        
        return {
            "success": True,
            "total_records": len(combined_data),
            "symbols_fetched": len(symbols) - len(failed_symbols),
            "failed_symbols": failed_symbols,
            "data": combined_data
        }
    else:
        return {
            "success": False,
            "message": "No data could be fetched",
            "failed_symbols": failed_symbols
        }

def main():
    """Main function to demonstrate simple data fetching."""
    
    # Get demo symbols
    from features.common.symbols import get_demo_symbols
    symbols = get_demo_symbols("small")
    
    # Fetch data
    result = fetch_data_simple(
        symbols=symbols,
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    if result["success"]:
        logger.info("🎉 Simple data fetch completed successfully!")
        logger.info(f"📊 Fetched {result['total_records']} records for {result['symbols_fetched']} symbols")
    else:
        logger.error(f"❌ Simple data fetch failed: {result['message']}")

if __name__ == "__main__":
    main()
