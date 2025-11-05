"""
SEC quarterly fundamental data fetching.

Downloads quarterly fundamental data from SEC EDGAR with automatic versioning.
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import hashlib
import os
import re
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from .config import (
    SEC_DATA_DIR,
    LATEST_DIR,
    SEC_DATA_PREFIX,
    SEC_USER_AGENT,
    SEC_BASE_URL,
    SEC_CACHE_DIR,
    SEC_RATE_LIMIT
)
from .versioning import save_with_versioning, copy_to_latest

# SEC API configuration
SEC_BASE = SEC_BASE_URL
HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}
QFPS = {"Q1", "Q2", "Q3", "Q4"}
START_DATE = pd.Timestamp("2014-09-30")
DATE_BUFFER_DAYS = 14
USE_FRAMES_FALLBACK = True
SLEEP_SEC = SEC_RATE_LIMIT

# Create cache directory
os.makedirs(SEC_CACHE_DIR, exist_ok=True)

# GAAP tag sets
TAG_REVENUE = ["RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues"]
TAG_NET_INCOME = ["NetIncomeLoss"]
TAG_EPS_DILUTED = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"]
TAG_SHARES = ["CommonStockSharesOutstanding",
              "WeightedAverageNumberOfDilutedSharesOutstanding",
              "WeightedAverageNumberOfSharesOutstandingDiluted"]
TAG_EQUITY = ["StockholdersEquity",
              "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
              "CommonStockholdersEquity"]
TAG_CFO = ["NetCashProvidedByUsedInOperatingActivities",
           "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
TAG_DPS = ["CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"]

# Session for requests
_session = requests.Session()
_session.headers.update(HEADERS)


def _cache_path(url: str) -> str:
    """Generate cache file path from URL."""
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(SEC_CACHE_DIR, f"{h}.json")


def fetch_json(url, params=None, use_cache=True, sleep=SLEEP_SEC):
    """Fetch JSON from SEC API with caching."""
    path = _cache_path(url + ("?" + json.dumps(params, sort_keys=True) if params else ""))
    if use_cache and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    for i in range(3):
        r = _session.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if use_cache:
                with open(path, "w") as f:
                    json.dump(data, f)
            return data
        if r.status_code == 404 and "/api/xbrl/frames/" in url:
            return {}
        time.sleep(0.4 * (i + 1))
    r.raise_for_status()


def get_sec_ticker_cik_map():
    """
    Fetch the complete ticker->CIK mapping from SEC
    Returns: dict of {ticker: cik}
    """
    # Use a fresh session for this request (not the cached _session)
    # This matches the original notebook implementation exactly
    import requests
    
    SEC_BASE = "https://www.sec.gov"
    HEADERS = {
        "User-Agent": SEC_USER_AGENT,  # Required by SEC
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov"
    }
    
    url = "https://www.sec.gov/files/company_tickers.json"
    
    print("📥 Fetching SEC ticker-CIK mapping...")
    
    # Use direct requests.get (not the session) - matches original notebook
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


def get_company_facts(cik: str):
    """Fetch company facts from SEC API."""
    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    return fetch_json(url, use_cache=True, sleep=0)


def _gaap_root(facts_dict):
    """Return the dict that holds us-gaap facts inside the CompanyFacts payload."""
    return (facts_dict or {}).get("facts", {}).get("us-gaap", {})


def _extract_quarterly_series_robust(facts: dict, tag_names, unit_priority=None):
    """Extract quarterly rows for the first tag that yields data in companyfacts."""
    gaap = _gaap_root(facts)
    for tag in tag_names:
        node = gaap.get(tag)
        if not node:
            continue
        units_dict = node.get("units", {})
        unit_list = unit_priority or tuple(units_dict.keys())
        for unit in unit_list:
            vals = units_dict.get(unit, [])
            if not vals:
                continue
            rows = []
            for v in vals:
                fp = v.get("fp")
                frame = v.get("frame", "")
                looks_quarterly = (fp in QFPS) or (re.search(r"Q[1-4]", str(frame)) is not None)
                if not looks_quarterly:
                    continue
                end = v.get("end")
                if not end:
                    continue
                end_dt = pd.to_datetime(end, errors="coerce")
                if pd.isna(end_dt) or end_dt < (START_DATE - pd.Timedelta(days=DATE_BUFFER_DAYS)):
                    continue
                rows.append({
                    "end": end_dt.normalize(),
                    "filed": pd.to_datetime(v.get("filed"), errors="coerce"),
                    "fy": v.get("fy"),
                    "fp": fp,
                    "frame": frame,
                    "val": v.get("val")
                })
            if rows:
                df = pd.DataFrame(rows).dropna(subset=["end"])
                df = (df.sort_values(["end", "filed"])
                        .groupby("end", as_index=False)
                        .tail(1)
                        .sort_values("end"))
                return df
    return pd.DataFrame(columns=["end", "filed", "fy", "fp", "frame", "val"])


def _missing_frames(start_date: pd.Timestamp, have_ends: pd.Series):
    """Find missing quarters for frames fallback."""
    want = []
    start = (start_date - pd.Timedelta(days=DATE_BUFFER_DAYS)).to_period("Q")
    end = pd.Timestamp.today().to_period("Q")
    have_periods = have_ends.dt.to_period("Q") if not have_ends.empty else pd.Series([], dtype="period[Q-DEC]")
    p = start
    while p <= end:
        if have_periods.empty or not (have_periods == p).any():
            want.append((p.year, p.quarter))
        p += 1
    return want


def _frames_quarter_series(cik: str, tag: str, unit: str, quarters: list, kind: str):
    """Fetch quarterly data from SEC frames API."""
    rows = []
    for year, q in quarters:
        if kind == "instant":
            variants = [f"FY{year}Q{q}I", f"CY{year}Q{q}I"]
        else:
            variants = [f"FY{year}Q{q}", f"FY{year}Q{q}YTD", f"CY{year}Q{q}", f"CY{year}Q{q}YTD"]
        got = False
        for frame_id in variants:
            url = f"{SEC_BASE}/api/xbrl/frames/us-gaap/{tag}/{unit}/{frame_id}"
            data = fetch_json(url, use_cache=True)
            if not data or "data" not in data:
                continue
            for row in data["data"]:
                if str(row.get("cik")).zfill(10) != str(cik):
                    continue
                end_dt = pd.to_datetime(row.get("end"), errors="coerce")
                if pd.isna(end_dt) or end_dt < (START_DATE - pd.Timedelta(days=DATE_BUFFER_DAYS)):
                    continue
                rows.append({
                    "end": end_dt.normalize(),
                    "filed": pd.to_datetime(row.get("filed"), errors="coerce"),
                    "fy": row.get("fy"),
                    "fp": row.get("fp"),
                    "frame": row.get("frame"),
                    "val": row.get("val")
                })
                got = True
            if got:
                break
        time.sleep(SLEEP_SEC)
    if not rows:
        return pd.DataFrame(columns=["end", "filed", "fy", "fp", "frame", "val"])
    df = (pd.DataFrame(rows)
            .sort_values(["end", "filed"])
            .groupby("end", as_index=False).tail(1)
            .sort_values("end"))
    return df


def _pull_quarterly_facts(facts: dict, cik: str):
    """Build quarterly facts (companyfacts + frames)."""
    def get_series(tags, prefer_units=None, unit_for_frames="USD", kind="duration"):
        s = _extract_quarterly_series_robust(facts, tags, unit_priority=prefer_units)
        if not USE_FRAMES_FALLBACK:
            return s
        have = s["end"] if not s.empty else pd.Series([], dtype="datetime64[ns]")
        missing = _missing_frames(START_DATE, have)
        if not missing:
            return s
        first_tag = tags[0]
        sf = _frames_quarter_series(cik, first_tag, unit_for_frames, missing, kind)
        return (pd.concat([s, sf], ignore_index=True)
                  .drop_duplicates(subset=["end"])
                  .sort_values("end"))

    rev = get_series(TAG_REVENUE, kind="duration")
    ni = get_series(TAG_NET_INCOME, kind="duration")
    eps = get_series(TAG_EPS_DILUTED, prefer_units=("USD/shares", "USD", "pure"), unit_for_frames="USD", kind="duration")
    cfo = get_series(TAG_CFO, kind="duration")
    dps = get_series(TAG_DPS, prefer_units=("USD/shares", "USD"), unit_for_frames="USD", kind="duration")
    sh = get_series(TAG_SHARES, prefer_units=("shares", "pure"), unit_for_frames="shares", kind="instant")
    eq = get_series(TAG_EQUITY, kind="instant")

    out = pd.DataFrame({"end": pd.to_datetime([])})
    for df_, name in [(rev, "revenue"), (ni, "net_income"), (eps, "eps_diluted"),
                      (sh, "shares_out"), (eq, "equity"),
                      (cfo, "operating_cash_flow"), (dps, "dividends_per_share")]:
        if not df_.empty:
            out = out.merge(df_[["end", "val"]].rename(columns={"val": name}),
                            on="end", how="outer")

    if out.empty:
        return out

    base = rev if not rev.empty else (ni if not ni.empty else (eps if not eps.empty else None))
    if base is not None and not base.empty:
        base_cols = [c for c in ["end", "fy", "fp", "frame"] if c in base.columns]
        out = out.merge(base[base_cols], on="end", how="left")
    else:
        out["fy"] = np.nan
        out["fp"] = np.nan
        out["frame"] = np.nan

    out = out[out["end"] >= (START_DATE - pd.Timedelta(days=DATE_BUFFER_DAYS))].copy()
    out = out.sort_values("end").reset_index(drop=True)
    out["year"] = out["end"].dt.year
    out["quarter"] = out["end"].dt.quarter
    return out


def safe_divide(numerator, denominator, default=np.nan):
    """Safely divide two arrays/series, returning default for division by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        if isinstance(result, pd.Series):
            result = result.replace([np.inf, -np.inf], default)
        else:
            result = np.where(np.isfinite(result), result, default)
    return result


def compute_quarterly_and_ttm(df_q: pd.DataFrame, price_df: pd.DataFrame, ticker: str):
    """Compute quarterly and TTM metrics from SEC data."""
    if df_q.empty:
        return df_q
    df = df_q.copy()
    
    px_cols = ["ticker", "period_end", "price_close"]
    extra_px_cols = [c for c in price_df.columns if c not in px_cols]
    merged = df.merge(
        price_df[["ticker", "period_end", "price_close"] + extra_px_cols]
            .rename(columns={"period_end": "end"}),
        on="end", how="left"
    )
    merged["ticker"] = ticker

    if "shares_out" in merged.columns and "net_income" in merged.columns and "eps_diluted" in merged.columns:
        computed_shares = safe_divide(merged["net_income"], merged["eps_diluted"])
        merged["shares_final"] = np.where(
            merged["shares_out"].isna() & merged["net_income"].notna() & merged["eps_diluted"].notna(),
            computed_shares,
            merged["shares_out"]
        )
    elif "shares_out" in merged.columns:
        merged["shares_final"] = merged["shares_out"]
    else:
        merged["shares_final"] = np.nan

    merged["market_cap"] = merged["price_close"] * merged["shares_final"]
    
    if "eps_diluted" in merged.columns:
        merged["pe_q"] = safe_divide(merged["price_close"], merged["eps_diluted"])
    else:
        merged["pe_q"] = np.nan
    
    if "equity" in merged.columns:
        merged["pb"] = safe_divide(merged["market_cap"], merged["equity"])
        merged["book_to_market"] = safe_divide(merged["equity"], merged["market_cap"])
    else:
        merged["pb"] = np.nan
        merged["book_to_market"] = np.nan

    merged = merged.sort_values("end")
    for col in ["revenue", "operating_cash_flow", "net_income", "eps_diluted", "dividends_per_share"]:
        if col in merged.columns:
            merged[f"{col}_ttm"] = merged[col].rolling(window=4, min_periods=4).sum()
        else:
            merged[f"{col}_ttm"] = np.nan

    if "eps_diluted_ttm" in merged.columns:
        merged["pe_ttm"] = safe_divide(merged["price_close"], merged["eps_diluted_ttm"])
    else:
        merged["pe_ttm"] = np.nan
    
    if "revenue_ttm" in merged.columns:
        merged["ps_ttm"] = safe_divide(merged["market_cap"], merged["revenue_ttm"])
    else:
        merged["ps_ttm"] = np.nan
    
    if "operating_cash_flow_ttm" in merged.columns:
        merged["pcf_ttm"] = safe_divide(merged["market_cap"], merged["operating_cash_flow_ttm"])
    else:
        merged["pcf_ttm"] = np.nan
    
    if "dividends_per_share_ttm" in merged.columns:
        merged["dividend_yield_ttm"] = safe_divide(merged["dividends_per_share_ttm"], merged["price_close"])
    else:
        merged["dividend_yield_ttm"] = np.nan

    desired_cols = [
        "ticker", "end", "fy", "fp", "frame", "year", "quarter", "price_close",
        "revenue", "net_income", "eps_diluted", "shares_out", "equity", "operating_cash_flow", "dividends_per_share",
        "market_cap", "pe_q", "pb", "book_to_market", "shares_final",
        "revenue_ttm", "operating_cash_flow_ttm", "eps_diluted_ttm", "dividends_per_share_ttm",
        "pe_ttm", "ps_ttm", "pcf_ttm", "dividend_yield_ttm"
    ] + extra_px_cols
    
    cols = [c for c in desired_cols if c in merged.columns]
    return merged[cols]


def fetch_quarterly_fundamentals(ticker: str, cik: str, price_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch quarterly fundamentals for a single ticker."""
    cik = str(int(cik)).zfill(10)
    facts = get_company_facts(cik)
    df_q = _pull_quarterly_facts(facts, cik)
    if df_q.empty:
        return df_q
    df_q = df_q[df_q["end"] >= START_DATE].copy()
    return compute_quarterly_and_ttm(df_q, price_df[price_df["ticker"] == ticker].copy(), ticker)


def fetch_sec_fundamentals(
    tickers,
    price_df,
    start_date='2014-01-01',
    end_date=None,
    data_dir=SEC_DATA_DIR,
    latest_dir=LATEST_DIR,
    verbose=True,
    resume=True
):
    """
    Fetch SEC quarterly fundamental data for multiple tickers.
    
    Features:
    - Incremental saving after each ticker
    - Resume functionality to continue from where it left off
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    price_df : pd.DataFrame
        Price data with columns ['ticker', 'period_end', 'price_close'] or
        ['symbol', 'date', 'close'] (will be converted)
    start_date : str
        Start date for data
    end_date : str, optional
        End date. If None, uses today
    data_dir : str
        Directory to save versioned files
    latest_dir : str
        Directory to save latest copy
    verbose : bool
        Whether to print progress
    resume : bool
        If True, check for existing partial results and continue from there
        
    Returns:
    --------
    tuple (pd.DataFrame, dict)
        (fundamentals_data, metadata)
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Extract years for filename
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    
    # Get download date (use today's date for consistency)
    download_date = datetime.now()
    download_date_str = download_date.strftime('%d_%m_%Y')
    
    # Generate filename for checkpoint/incremental save
    os.makedirs(data_dir, exist_ok=True)
    checkpoint_filename = f"{SEC_DATA_PREFIX}_{start_year}-{end_year}_download_date_{download_date_str}_in_progress.pkl"
    checkpoint_path = os.path.join(data_dir, checkpoint_filename)
    
    # Check for existing partial results if resume is enabled
    existing_fundamentals_df = None
    completed_tickers = set()
    successful_count = 0
    failed_count = 0
    
    if resume:
        # Try to load latest in-progress file or latest completed file
        latest_checkpoint = os.path.join(latest_dir, 'sec_fundamentals_latest_in_progress.pkl')
        
        # Check for in-progress checkpoint
        if os.path.exists(checkpoint_path):
            if verbose:
                print(f"📂 Found existing checkpoint: {os.path.basename(checkpoint_path)}")
            try:
                existing_fundamentals_df = pd.read_pickle(checkpoint_path)
                if 'symbol' in existing_fundamentals_df.columns:
                    completed_tickers = set(existing_fundamentals_df['symbol'].unique())
                    if verbose:
                        print(f"   ✅ Resuming: {len(completed_tickers)} tickers already fetched")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Could not load checkpoint: {str(e)}")
                existing_fundamentals_df = None
                completed_tickers = set()
        
        # Also check latest in-progress file
        elif os.path.exists(latest_checkpoint):
            if verbose:
                print(f"📂 Found latest checkpoint: {os.path.basename(latest_checkpoint)}")
            try:
                existing_fundamentals_df = pd.read_pickle(latest_checkpoint)
                if 'symbol' in existing_fundamentals_df.columns:
                    completed_tickers = set(existing_fundamentals_df['symbol'].unique())
                    if verbose:
                        print(f"   ✅ Resuming: {len(completed_tickers)} tickers already fetched")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Could not load checkpoint: {str(e)}")
    
    if verbose:
        print(f"🚀 Fetching SEC quarterly fundamental data...")
        print(f"📈 Adding fundamentals for {len(tickers)} stocks...")
        if completed_tickers:
            print(f"   🔄 Resuming: {len(completed_tickers)} already completed, {len(tickers) - len(completed_tickers)} remaining")
    
    # Convert price_df format if needed
    if 'symbol' in price_df.columns and 'date' in price_df.columns:
        price_df_formatted = price_df[['symbol', 'date', 'close']].rename(columns={
            'symbol': 'ticker',
            'date': 'period_end',
            'close': 'price_close'
        }).copy()
    else:
        price_df_formatted = price_df.copy()
    
    # Get CIK mappings
    if verbose:
        print("\n📥 Fetching CIK mappings from SEC...")
    
    try:
        full_cik_map = get_sec_ticker_cik_map()
    except Exception as e:
        error_msg = f"❌ Failed to fetch CIK mappings: {type(e).__name__}: {str(e)}"
        if verbose:
            print(error_msg)
            print("   🔍 Troubleshooting:")
            print("      - Check internet connection")
            print("      - Verify SEC_USER_AGENT is correctly formatted in config.py")
            print("      - SEC API might be temporarily unavailable")
        raise Exception(error_msg) from e
    
    if verbose:
        print(f"   ✅ Successfully fetched CIK mappings")
    
    cik_map = {}
    missing_ciks = []
    for symbol in tickers:
        if symbol in full_cik_map:
            cik_map[symbol] = full_cik_map[symbol]
        else:
            missing_ciks.append(symbol)
    
    if verbose:
        print(f"   ✅ Found CIKs for {len(cik_map)}/{len(tickers)} tickers")
        if missing_ciks:
            print(f"   ⚠️  Missing CIK for {len(missing_ciks)} tickers: {missing_ciks[:10]}{'...' if len(missing_ciks) > 10 else ''}")
            print(f"   💡 This is normal - some tickers may not have SEC filings")
    
    # Filter out already completed tickers if resuming
    tickers_to_fetch = [t for t in tickers if t not in completed_tickers]
    
    if not tickers_to_fetch:
        if verbose:
            print(f"\n✅ All tickers already fetched! Using existing data.")
        fundamentals_df = existing_fundamentals_df
        successful_count = len(completed_tickers)
    else:
        # Initialize with existing data if resuming
        fundamental_data_list = []
        if existing_fundamentals_df is not None and not existing_fundamentals_df.empty:
            fundamental_data_list.append(existing_fundamentals_df)
            successful_count = len(completed_tickers)
        else:
            successful_count = 0
        
        # Fetch remaining fundamentals
        start_time = datetime.now()
        
        for i, symbol in enumerate(tickers_to_fetch):
            # Print progress for every ticker (not just every 10)
            if verbose:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = i / elapsed if elapsed > 0 else 0
                remaining = len(tickers_to_fetch) - i - 1
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                
                print(f"\n  [{i+1}/{len(tickers_to_fetch)}] 📊 Fetching {symbol}...")
                print(f"      ✅ Completed: {successful_count} | ❌ Failed: {failed_count} | ⏱️  ETA: {eta_minutes:.1f} min")
            
            if symbol not in cik_map:
                if verbose:
                    print(f"      ⚠️  Skipping {symbol} - no CIK mapping found")
                failed_count += 1
                continue
            
            try:
                cik = cik_map[symbol]
                if verbose:
                    print(f"      🔍 Fetching fundamentals for CIK: {cik}")
                
                quarterly_df = fetch_quarterly_fundamentals(symbol, cik, price_df_formatted)
                
                if quarterly_df.empty:
                    if verbose:
                        print(f"      ⚠️  {symbol}: No quarterly data available")
                    failed_count += 1
                    # Save checkpoint even if failed (to track progress)
                    if fundamental_data_list:
                        temp_df = pd.concat(fundamental_data_list, ignore_index=True)
                        temp_df.to_pickle(checkpoint_path)
                        # Also update latest checkpoint
                        latest_checkpoint_path = os.path.join(latest_dir, 'sec_fundamentals_latest_in_progress.pkl')
                        os.makedirs(latest_dir, exist_ok=True)
                        temp_df.to_pickle(latest_checkpoint_path)
                    continue
                
                quarterly_df['symbol'] = symbol
                fundamental_data_list.append(quarterly_df)
                successful_count += 1
                
                if verbose:
                    print(f"      ✅ {symbol}: Retrieved {len(quarterly_df)} quarterly records")
                
                # Save incrementally after each successful fetch
                if fundamental_data_list:
                    temp_df = pd.concat(fundamental_data_list, ignore_index=True)
                    temp_df.to_pickle(checkpoint_path)
                    # Also update latest checkpoint
                    latest_checkpoint_path = os.path.join(latest_dir, 'sec_fundamentals_latest_in_progress.pkl')
                    os.makedirs(latest_dir, exist_ok=True)
                    temp_df.to_pickle(latest_checkpoint_path)
                    
                    if verbose:
                        print(f"      💾 Checkpoint saved: {successful_count} tickers completed, {len(temp_df):,} total records")
                
                time.sleep(SLEEP_SEC)
                
            except Exception as e:
                if verbose:
                    print(f"      ❌ {symbol}: {type(e).__name__}: {str(e)}")
                    import traceback
                    print(f"         Traceback: {traceback.format_exc().split(chr(10))[-2] if traceback.format_exc() else 'N/A'}")
                failed_count += 1
                # Save checkpoint even on error
                if fundamental_data_list:
                    temp_df = pd.concat(fundamental_data_list, ignore_index=True)
                    temp_df.to_pickle(checkpoint_path)
                    latest_checkpoint_path = os.path.join(latest_dir, 'sec_fundamentals_latest_in_progress.pkl')
                    os.makedirs(latest_dir, exist_ok=True)
                    temp_df.to_pickle(latest_checkpoint_path)
                continue
            
            # Print summary every 10 tickers
            if verbose and (i + 1) % 10 == 0:
                elapsed_total = (datetime.now() - start_time).total_seconds() / 60
                print(f"\n  📊 Progress Summary ({i+1}/{len(tickers_to_fetch)}):")
                print(f"      ✅ Successful: {successful_count} ({successful_count/(i+1)*100:.1f}%)")
                print(f"      ❌ Failed: {failed_count} ({failed_count/(i+1)*100:.1f}%)")
                print(f"      ⏱️  Elapsed: {elapsed_total:.1f} minutes")
                if elapsed_total > 0:
                    print(f"      📈 Rate: {(i+1)/elapsed_total:.1f} tickers/min")
                print(f"      💾 Checkpoint: {checkpoint_path}")
        
        if not fundamental_data_list:
            if existing_fundamentals_df is not None:
                fundamentals_df = existing_fundamentals_df
            else:
                print("❌ No SEC fundamental data retrieved")
                return None, None
        else:
            # Combine all fundamental data
            fundamentals_df = pd.concat(fundamental_data_list, ignore_index=True)
    
    if verbose:
        print(f"\n📊 Summary:")
        print(f"   ✅ Successful: {successful_count}")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   Total fundamental records: {len(fundamentals_df):,}")
    
    # Generate final filename (without _in_progress suffix)
    filename = f"{SEC_DATA_PREFIX}_{start_year}-{end_year}_download_date_{download_date_str}.pkl"
    filepath = os.path.join(data_dir, filename)
    
    # Save final versioned file
    if verbose:
        print(f"\n💾 Saving final data with versioning...")
    saved_path, backup_path = save_with_versioning(
        fundamentals_df, filepath, SEC_DATA_PREFIX, start_year, end_year, download_date_str, save_csv=True
    )
    
    # Copy to latest
    latest_path = copy_to_latest(saved_path, latest_dir, 'sec_fundamentals_latest')
    
    # Remove checkpoint files after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        if verbose:
            print(f"   🗑️  Removed checkpoint file")
    
    latest_checkpoint_path = os.path.join(latest_dir, 'sec_fundamentals_latest_in_progress.pkl')
    if os.path.exists(latest_checkpoint_path):
        os.remove(latest_checkpoint_path)
    
    # Create metadata
    metadata = {
        'timestamp': download_date.strftime('%Y%m%d_%H%M%S'),
        'filepath': saved_path,
        'latest_filepath': latest_path,
        'backup_filepath': backup_path,
        'successful_tickers': list(fundamentals_df['symbol'].unique()) if 'symbol' in fundamentals_df.columns else [],
        'failed_tickers': list(missing_ciks),
        'total_tickers': len(tickers),
        'date_range': {
            'start': start_date,
            'end': end_date
        },
        'data_info': {
            'total_records': len(fundamentals_df),
            'unique_symbols': fundamentals_df['symbol'].nunique() if 'symbol' in fundamentals_df.columns else 0
        }
    }
    
    if verbose:
        print(f"\n✅ SEC fundamentals saved successfully!")
        print(f"   📄 Main file: {os.path.basename(saved_path)}")
        print(f"   📋 Latest copy: {os.path.basename(latest_path)}")
    
    return fundamentals_df, metadata

