#!/usr/bin/env python3
"""
Fetch CIK (Central Index Key) mappings for S&P 500 tickers from SEC
"""
import requests
import json
import time
import pandas as pd

SEC_BASE = "https://www.sec.gov"
HEADERS = {
    "User-Agent": "Noam Polak noampolak@gmail.com",  # Required by SEC
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}

def get_sec_ticker_cik_map():
    """
    Fetch the complete ticker->CIK mapping from SEC
    Returns: dict of {ticker: cik}
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    
    print("📥 Fetching SEC ticker-CIK mapping...")
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch CIK map: {response.status_code}")
    
    data = response.json()
    
    # Convert to {ticker: CIK} dict
    ticker_cik_map = {}
    for entry in data.values():
        ticker = entry['ticker']
        cik = str(entry['cik_str']).zfill(10)  # Pad to 10 digits
        ticker_cik_map[ticker] = cik
    
    print(f"✅ Found {len(ticker_cik_map)} ticker-CIK mappings")
    return ticker_cik_map

def filter_sp500_ciks(tickers, full_cik_map):
    """
    Filter CIK map to only include S&P 500 tickers
    """
    sp500_cik_map = {}
    missing = []
    
    for ticker in tickers:
        if ticker in full_cik_map:
            sp500_cik_map[ticker] = full_cik_map[ticker]
        else:
            missing.append(ticker)
    
    print(f"\n📊 S&P 500 CIK Mapping:")
    print(f"   Found: {len(sp500_cik_map)}")
    print(f"   Missing: {len(missing)}")
    
    if missing:
        print(f"\n⚠️  Missing CIK for {len(missing)} tickers:")
        for t in missing[:10]:
            print(f"      {t}")
        if len(missing) > 10:
            print(f"      ... and {len(missing)-10} more")
    
    return sp500_cik_map, missing

def save_cik_mapping(cik_map, filename="data/research/sp500_cik_map.json"):
    """Save CIK mapping to file"""
    with open(filename, 'w') as f:
        json.dump(cik_map, f, indent=2)
    print(f"\n💾 Saved to: {filename}")

if __name__ == "__main__":
    # Example: Get mappings for a few tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    full_map = get_sec_ticker_cik_map()
    
    print(f"\n🔍 Test lookups:")
    for ticker in test_tickers:
        cik = full_map.get(ticker, "NOT FOUND")
        print(f"   {ticker}: {cik}")

