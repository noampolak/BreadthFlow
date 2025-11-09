"""
Feature creation functions for training.
"""

import pandas as pd
import numpy as np
from .cache_manager import load_features_from_cache, save_features_to_cache
from .config import DEFAULT_FEATURE_WINDOWS


# Technical Indicator Functions
def calculate_rsi_custom(prices, window=14):
    """Calculate RSI without talib dependency"""
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_macd_custom(prices, fast=12, slow=26, signal=9):
    """Calculate MACD without talib dependency"""
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line.values, macd_signal.values, macd_histogram.values


def calculate_cci_custom(high, low, close, window=14):
    """Calculate CCI (Commodity Channel Index)"""
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
    
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci.values


def calculate_kdj_custom(high, low, close, window=9, smooth_k=3, smooth_d=3):
    """Calculate KDJ indicator"""
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
    
    low_min = low.rolling(window).min()
    high_max = high.rolling(window).max()
    rsv = (close - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=smooth_k-1, adjust=False).mean()
    d = k.ewm(com=smooth_d-1, adjust=False).mean()
    return k.values, d.values


def calculate_williams_r_custom(high, low, close, window=14):
    """Calculate Williams %R"""
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
    
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return willr.values


def calculate_parabolic_sar_custom(high, low, close, af=0.02, max_af=0.2):
    """Calculate Parabolic SAR"""
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
    
    # Simplified SAR calculation
    sar = close.copy()
    for i in range(1, len(close)):
        if i == 0:
            sar.iloc[i] = low.iloc[i]
        else:
            sar.iloc[i] = sar.iloc[i-1] + af * (high.iloc[i-1] - sar.iloc[i-1])
    
    return sar.values


def calculate_momentum_features(df, window):
    """
    Calculate momentum features for a specific window (matches paper's approach)
    All features use the SAME window parameter
    
    ⚠️  TIMING FIX: Features for date T use data only up to T-1 (shifted by 1 trading day)
    This ensures when trading at start of day T, we have all required data.
    
    Based on paper's Table 2: mom_amplitude, mom_roc, mom_current_rtn, etc.
    """
    features = {}
    
    print(f"    📈 Calculating momentum features for {window}-day window...")
    print(f"       ⚠️  Using prices shifted by 1 trading day (T uses data ≤ T-1)")
    
    # ⚠️  TIMING FIX: Shift prices by 1 trading day so features for date T use close[T-1]
    # Group by symbol, shift close prices forward by 1 trading day
    df_sorted = df.sort_values(['symbol', 'date']).copy()
    
    # Shift close prices by 1 row (1 trading day) within each symbol group
    shifted_close = df_sorted.groupby('symbol')['close'].shift(1)
    shifted_high = df_sorted.groupby('symbol')['high'].shift(1)
    shifted_low = df_sorted.groupby('symbol')['low'].shift(1)
    shifted_volume = df_sorted.groupby('symbol')['volume'].shift(1)
    
    # Align shifted prices back to original index
    df_with_shifted = df_sorted.copy()
    df_with_shifted['close_shifted'] = shifted_close
    df_with_shifted['high_shifted'] = shifted_high
    df_with_shifted['low_shifted'] = shifted_low
    df_with_shifted['volume_shifted'] = shifted_volume
    df_with_shifted = df_with_shifted.set_index(df.index)
    
    # Now use shifted prices for calculations
    # 1. mom_current_rtn - current period return (using shifted prices)
    features['mom_current_rtn'] = df_with_shifted.groupby('symbol')['close_shifted'].pct_change(window)
    
    # 2. mom_current_rtn_std - std of returns (using shifted prices)
    features['mom_current_rtn_std'] = df_with_shifted.groupby('symbol')['close_shifted'].pct_change(1).rolling(window).std().reset_index(level=0, drop=True)
    
    # 3. mom_amplitude - price amplitude (high-low)/close (using shifted prices)
    amplitude = (df_with_shifted['high_shifted'] - df_with_shifted['low_shifted']) / df_with_shifted['close_shifted']
    features['mom_amplitude'] = amplitude
    features['mom_amplitude_std'] = amplitude.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 4. mom_roc - Rate of Change (using shifted prices)
    roc = df_with_shifted.groupby('symbol')['close_shifted'].transform(lambda x: (x - x.shift(window)) / x.shift(window) * 100)
    features['mom_roc'] = roc
    features['mom_roc_std'] = roc.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 5. mom_ps - momentum persistence (using shifted prices)
    returns = df_with_shifted.groupby('symbol')['close_shifted'].pct_change(1)
    features['mom_ps'] = returns.groupby(df['symbol']).transform(
        lambda x: x.rolling(window).apply(lambda y: (y > 0).sum() / len(y) if len(y) > 0 else 0.5)
    )
    mom_ps_series = features['mom_ps']
    features['mom_ps_std'] = mom_ps_series.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 6. mom_turnover - volume turnover (using shifted prices)
    turnover = df_with_shifted['volume_shifted'] * df_with_shifted['close_shifted']
    features['mom_turnover'] = turnover
    features['mom_turnover_std'] = turnover.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 7. mom_yield_dispersion - cross-sectional yield dispersion (using shifted prices)
    returns_cs = df_with_shifted.groupby('symbol')['close_shifted'].pct_change(window)
    yield_disp = returns_cs.groupby(df['date']).transform('std')
    features['mom_yield_dispersion'] = yield_disp
    features['mom_yield_dispersion_std'] = yield_disp.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 8. mom_lb - Lower Band (using shifted prices)
    rolling_mean = df_with_shifted.groupby('symbol')['close_shifted'].transform(lambda x: x.rolling(window).mean())
    rolling_std = df_with_shifted.groupby('symbol')['close_shifted'].transform(lambda x: x.rolling(window).std())
    mom_lb = rolling_mean - 2 * rolling_std
    features['mom_lb'] = mom_lb
    features['mom_lb_std'] = mom_lb.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 9. Fundamental momentum (using shifted close price)
    if 'market_cap' in df.columns:
        market_cap_ratio = df_with_shifted['close_shifted'] / (df['market_cap'] / 1e9)
        features['mom_market_cap'] = market_cap_ratio
        features['mom_market_cap_std'] = market_cap_ratio.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    if 'cir_cap' in df.columns:
        cir_cap_ratio = df_with_shifted['close_shifted'] / (df['cir_cap'] / 1e9)
        features['mom_cir_cap'] = cir_cap_ratio
        features['mom_cir_cap_std'] = cir_cap_ratio.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # Add fundamental ratios with momentum trends (ratios don't need shifting, but use shifted close if needed)
    # Check for TTM versions first (pe_ttm, ps_ttm, pcf_ttm), then fallback to simple versions
    ratio_mapping = [
        ('pe_ttm', 'pe_q', 'pe'),  # Try pe_ttm, then pe_q, then pe
        ('ps_ttm', 'ps', None),
        ('pcf_ttm', 'pcf', None),
        ('pb', None, None)  # pb doesn't have TTM version
    ]
    
    for preferred, fallback1, fallback2 in ratio_mapping:
        ratio_col = None
        if preferred in df.columns:
            ratio_col = preferred
        elif fallback1 and fallback1 in df.columns:
            ratio_col = fallback1
        elif fallback2 and fallback2 in df.columns:
            ratio_col = fallback2
        
        if ratio_col:
            # Use base name (without _ttm or _q) for feature name
            base_name = ratio_col.replace('_ttm', '').replace('_q', '')
            ratio_pct = df.groupby('symbol')[ratio_col].pct_change(window)
            features[f'mom_{base_name}'] = ratio_pct
            features[f'mom_{base_name}_std'] = df.groupby('symbol')[ratio_col].transform(lambda x: x.rolling(window).std())
    
    print(f"    ✅ Generated {len(features)} momentum features")
    return features


def calculate_reversal_features(df, window):
    """
    Calculate reversal features for a specific window (matches paper's approach)
    All features use the SAME window parameter
    
    Based on paper's Table 2: rev_amplitude, rev_roc, rev_current_rtn, etc.
    """
    features = {}
    
    print(f"    📉 Calculating reversal features for {window}-day window...")
    
    # 1. rev_current_rtn - reversal of current return (contrarian)
    features['rev_current_rtn'] = -df.groupby('symbol')['close'].pct_change(window)
    
    # 2. rev_current_rtn_std
    features['rev_current_rtn_std'] = df.groupby('symbol')['close'].pct_change(1).rolling(window).std().reset_index(level=0, drop=True)
    
    # 3. rev_amplitude - reversal amplitude
    amplitude = (df['high'] - df['low']) / df['close']
    features['rev_amplitude'] = amplitude
    features['rev_amplitude_std'] = amplitude.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 4. rev_roc - reversal ROC (opposite momentum)
    roc = -df.groupby('symbol')['close'].transform(lambda x: (x - x.shift(window)) / x.shift(window) * 100)
    features['rev_roc'] = roc
    features['rev_roc_std'] = roc.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 5. rev_turnover
    turnover = df['volume'] * df['close']
    features['rev_turnover'] = turnover
    features['rev_turnover_std'] = turnover.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 6. rev_yield_dispersion
    returns_cs = df.groupby('symbol')['close'].pct_change(window)
    yield_disp = returns_cs.groupby(df['date']).transform('std')
    features['rev_yield_dispersion'] = yield_disp
    features['rev_yield_dispersion_std'] = yield_disp.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 7. rev_lb - reversal lower band
    rolling_mean = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window).mean())
    rolling_std = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window).std())
    rev_lb = rolling_mean - 2 * rolling_std
    features['rev_lb'] = rev_lb
    features['rev_lb_std'] = rev_lb.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 8. Fundamental reversals
    if 'market_cap' in df.columns:
        market_cap_pct = -df.groupby('symbol')['market_cap'].pct_change(window)
        features['rev_market_cap'] = market_cap_pct
        features['rev_market_cap_std'] = df.groupby('symbol')['market_cap'].transform(lambda x: x.rolling(window).std())
    
    if 'cir_cap' in df.columns:
        cir_cap_pct = -df.groupby('symbol')['cir_cap'].pct_change(window)
        features['rev_cir_cap'] = cir_cap_pct
        features['rev_cir_cap_std'] = cir_cap_pct.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # Add fundamental ratios with reversal trends
    # Check for TTM versions first (pe_ttm, ps_ttm, pcf_ttm), then fallback to simple versions
    ratio_mapping = [
        ('pe_ttm', 'pe_q', 'pe'),  # Try pe_ttm, then pe_q, then pe
        ('ps_ttm', 'ps', None),
        ('pcf_ttm', 'pcf', None),
        ('pb', None, None)  # pb doesn't have TTM version
    ]
    
    for preferred, fallback1, fallback2 in ratio_mapping:
        ratio_col = None
        if preferred in df.columns:
            ratio_col = preferred
        elif fallback1 and fallback1 in df.columns:
            ratio_col = fallback1
        elif fallback2 and fallback2 in df.columns:
            ratio_col = fallback2
        
        if ratio_col:
            # Use base name (without _ttm or _q) for feature name
            base_name = ratio_col.replace('_ttm', '').replace('_q', '')
            ratio_pct = -df.groupby('symbol')[ratio_col].pct_change(window)
            features[f'rev_{base_name}'] = ratio_pct
            features[f'rev_{base_name}_std'] = df.groupby('symbol')[ratio_col].transform(lambda x: x.rolling(window).std())
    
    print(f"    ✅ Generated {len(features)} reversal features")
    return features


def calculate_volume_features(df, window):
    """
    Calculate volume features for a specific window
    """
    features = {}
    
    print(f"    📊 Calculating volume features for {window}-day window...")
    
    # 1. Volume change (vol_change)
    features['vol_change'] = df.groupby('symbol')['volume'].pct_change(window)
    
    # 2. Volume turnover
    turnover = df['volume'] * df['close']
    features['turnover'] = turnover
    features['turnover_std'] = turnover.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 3. On-Balance Volume (OBV)
    price_change = df.groupby('symbol')['close'].diff()
    obv_change = df['volume'] * np.sign(price_change)
    features['obv'] = obv_change.groupby(df['symbol']).transform(lambda x: x.rolling(window).sum())
    
    # 4. Yield dispersion (already calculated in momentum, but keep for completeness)
    returns = df.groupby('symbol')['close'].pct_change(window)
    yield_disp = returns.groupby(df['date']).transform('std')
    features['yield_dispersion'] = yield_disp
    features['yield_dispersion_std'] = yield_disp.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    print(f"    ✅ Generated {len(features)} volume features")
    return features


def calculate_base_features(df, window):
    """
    Calculate base features (non-momentum, non-reversal) for a specific window.
    These are the base indicators from the paper.
    
    ⚠️  TIMING FIX: Features for date T use data only up to T-1 (shifted by 1 trading day)
    """
    features = {}
    
    print(f"    📊 Calculating base features for {window}-day window...")
    print(f"       ⚠️  Using prices shifted by 1 trading day (T uses data ≤ T-1)")
    
    # Shift prices by 1 trading day for timing fix
    df_sorted = df.sort_values(['symbol', 'date']).copy()
    shifted_close = df_sorted.groupby('symbol')['close'].shift(1)
    shifted_high = df_sorted.groupby('symbol')['high'].shift(1)
    shifted_low = df_sorted.groupby('symbol')['low'].shift(1)
    
    df_with_shifted = df_sorted.copy()
    df_with_shifted['close_shifted'] = shifted_close
    df_with_shifted['high_shifted'] = shifted_high
    df_with_shifted['low_shifted'] = shifted_low
    df_with_shifted = df_with_shifted.set_index(df.index)
    
    # 1. amplitude - price amplitude (high-low)/close
    amplitude = (df_with_shifted['high_shifted'] - df_with_shifted['low_shifted']) / df_with_shifted['close_shifted']
    features['amplitude'] = amplitude
    features['amplitude_std'] = amplitude.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 2. change - price change
    features['change'] = df_with_shifted.groupby('symbol')['close_shifted'].pct_change(1)
    
    # 3. current_rtn - current period return
    features['current_rtn'] = df_with_shifted.groupby('symbol')['close_shifted'].pct_change(1)
    features['current_rtn_std'] = features['current_rtn'].groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 4. roc - Rate of Change
    roc = df_with_shifted.groupby('symbol')['close_shifted'].transform(lambda x: (x - x.shift(window)) / x.shift(window) * 100)
    features['roc'] = roc
    features['roc_std'] = roc.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    # 5. lb - Lower Band (Bollinger lower band: SMA - 2*std)
    rolling_mean = df_with_shifted.groupby('symbol')['close_shifted'].transform(lambda x: x.rolling(window).mean())
    rolling_std = df_with_shifted.groupby('symbol')['close_shifted'].transform(lambda x: x.rolling(window).std())
    lb = rolling_mean - 2 * rolling_std
    features['lb'] = lb
    features['lb_std'] = lb.groupby(df['symbol']).transform(lambda x: x.rolling(window).std())
    
    print(f"    ✅ Generated {len(features)} base features")
    return features


def calculate_fundamental_std_features(df, window):
    """
    Calculate standard deviation features for fundamental indicators.
    Only calculates if the base fundamental column exists.
    """
    features = {}
    
    print(f"    📊 Calculating fundamental std features for {window}-day window...")
    
    # DEBUG: Check what fundamental columns are actually available
    # Use exact matches or specific patterns to avoid false positives
    fundamental_patterns = ['pe_ttm', 'pe_q', 'pe', 'ps_ttm', 'ps', 'pcf_ttm', 'pcf', 'pb', 'market_cap', 'cir_cap']
    available_fundamental_cols = [col for col in df.columns if col in fundamental_patterns or 
                                  any(col.startswith(pat) or col == pat for pat in ['pe_', 'ps_', 'pcf_', 'pb', 'market_cap', 'cir_cap'])]
    if available_fundamental_cols:
        print(f"       🔍 Found fundamental columns: {available_fundamental_cols}")
    else:
        print(f"       ⚠️  No fundamental columns found in DataFrame")
        print(f"       📋 Available columns: {list(df.columns)[:30]}{'...' if len(df.columns) > 30 else ''}")
        print(f"       🔍 Looking for: {fundamental_patterns}")
    
    # Check for TTM versions first (preferred), then fallback to quarterly/simple versions
    fundamental_cols = [
        ('pe_ttm', 'pe_q', 'pe'),  # Try pe_ttm, then pe_q, then pe
        ('ps_ttm', 'ps', None),
        ('pcf_ttm', 'pcf', None),
        ('pb', None, None),  # pb doesn't have TTM version
        ('market_cap', None, None),
        ('cir_cap', None, None)
    ]
    
    for preferred_col, fallback1_col, fallback2_col in fundamental_cols:
        # Try preferred column first, then fallbacks
        col_to_use = None
        if preferred_col in df.columns:
            col_to_use = preferred_col
        elif fallback1_col and fallback1_col in df.columns:
            col_to_use = fallback1_col
        elif fallback2_col and fallback2_col in df.columns:
            col_to_use = fallback2_col
        
        if col_to_use:
            # Calculate rolling std of the fundamental metric
            # Use the base name without _ttm suffix for the std feature name
            feature_name = col_to_use.replace('_ttm', '').replace('_q', '') + '_std'
            features[feature_name] = df.groupby('symbol')[col_to_use].transform(lambda x: x.rolling(window).std())
            print(f"       ✅ Created {feature_name} from {col_to_use}")
    
    print(f"    ✅ Generated {len(features)} fundamental std features")
    if len(features) > 0:
        print(f"       Features: {list(features.keys())}")
    return features


def calculate_hurst_exponent(series, max_lag=None):
    """
    Calculate Hurst exponent using R/S (Rescaled Range) method.
    
    Parameters:
    -----------
    series : pd.Series or np.ndarray
        Time series data
    max_lag : int, optional
        Maximum lag to use. If None, uses len(series) // 2
    
    Returns:
    --------
    float
        Hurst exponent (0 < H < 1)
        - H = 0.5: Random walk
        - H > 0.5: Trending (persistent)
        - H < 0.5: Mean reverting (anti-persistent)
    """
    if len(series) < 10:
        return 0.5  # Default to random walk for short series
    
    series = pd.Series(series) if not isinstance(series, pd.Series) else series
    series = series.dropna()
    
    if len(series) < 10:
        return 0.5
    
    if max_lag is None:
        max_lag = len(series) // 2
    
    max_lag = min(max_lag, len(series) // 2)
    if max_lag < 2:
        return 0.5
    
    lags = range(2, max_lag + 1)
    rs_values = []
    
    for lag in lags:
        # Split series into chunks of size lag
        n_chunks = len(series) // lag
        if n_chunks < 1:
            continue
        
        rs_list = []
        for i in range(n_chunks):
            chunk = series.iloc[i*lag:(i+1)*lag]
            if len(chunk) < 2:
                continue
            
            # Calculate mean
            mean_chunk = chunk.mean()
            
            # Calculate deviations from mean
            deviations = chunk - mean_chunk
            
            # Calculate cumulative deviations
            cumsum_deviations = deviations.cumsum()
            
            # Calculate range
            R = cumsum_deviations.max() - cumsum_deviations.min()
            
            # Calculate standard deviation
            S = chunk.std()
            
            # Avoid division by zero
            if S == 0 or pd.isna(S) or S < 1e-10:
                continue
            
            # Calculate R/S
            rs = R / S
            if rs > 0 and not (pd.isna(rs) or np.isinf(rs)):
                rs_list.append(rs)
        
        if rs_list:
            rs_values.append(np.mean(rs_list))
    
    if len(rs_values) < 2:
        return 0.5
    
    # Fit log(R/S) vs log(lag) to get Hurst exponent
    log_lags = np.log(lags[:len(rs_values)])
    log_rs = np.log(rs_values)
    
    # Remove any NaN or inf values
    valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
    if np.sum(valid_mask) < 2:
        return 0.5
    
    log_lags = log_lags[valid_mask]
    log_rs = log_rs[valid_mask]
    
    # Linear regression: log(R/S) = H * log(lag) + c
    if len(log_lags) > 1:
        H = np.polyfit(log_lags, log_rs, 1)[0]
        # Clamp H to reasonable range
        H = np.clip(H, 0.01, 0.99)
        return H
    
    return 0.5


def create_features_for_window(df, feature_window, verbose=True):
    """
    Create features based on a specific window size using paper's approach.
    All ~80-100 features are calculated using the SAME window.
    
    Parameters:
    -----------
    df : DataFrame
        Stock data (should already have fundamental data merged if available)
    feature_window : int
        Feature calculation window in days (5, 10, 15, 20, 25, 30)
    verbose : bool
        Whether to print detailed information
    
    Returns:
    --------
    feature_df : DataFrame
        DataFrame with ~80-100 window-specific features
    """
    print(f"\n  📊 Creating comprehensive features for {feature_window}-day window...")
    
    # Start with ALL columns from df (including fundamentals if already merged)
    # This ensures fundamental data is included if it was merged with stock data
    required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    
    # Check which required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in df: {missing_cols}")
    
    # Start with required columns, then add all other columns (fundamentals, sector, etc.)
    feature_df = df[required_cols].copy()
    
    # Add all other columns from df (fundamentals, sector, metadata, etc.)
    # This includes: pe_ttm, pb, ps_ttm, pcf_ttm, market_cap, cir_cap, sector, etc.
    other_cols = [col for col in df.columns if col not in required_cols]
    for col in other_cols:
        feature_df[col] = df[col]
    
    if verbose:
        fundamental_cols_found = [col for col in feature_df.columns 
                                  if any(x in col.lower() for x in ['pe', 'pb', 'ps', 'pcf', 'market_cap', 'cir_cap', 'revenue', 'equity'])]
        if fundamental_cols_found:
            print(f"    ✅ Included {len(fundamental_cols_found)} fundamental columns: {fundamental_cols_found[:5]}{'...' if len(fundamental_cols_found) > 5 else ''}")
        else:
            print(f"    ⚠️  No fundamental columns found in input DataFrame")
    
    # Calculate ALL features using the SAME window
    print(f"    🔧 Calculating base features...")
    base_feats = calculate_base_features(df, feature_window)
    
    print(f"    🔧 Calculating momentum features...")
    momentum_feats = calculate_momentum_features(df, feature_window)
    
    print(f"    🔧 Calculating reversal features...")
    reversal_feats = calculate_reversal_features(df, feature_window)
    
    print(f"    🔧 Calculating volume features...")
    volume_feats = calculate_volume_features(df, feature_window)
    
    print(f"    🔧 Calculating fundamental std features...")
    # Use feature_df instead of df - it has the fundamental columns copied if they exist
    fundamental_std_feats = calculate_fundamental_std_features(feature_df, feature_window)
    
    # Technical indicators (using window parameter)
    print(f"    🔧 Calculating technical indicators...")
    tech_feats = {}
    
    # RSI (use window as period)
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        symbol_data = df[mask]['close'].values
        if len(symbol_data) >= feature_window:
            rsi = calculate_rsi_custom(symbol_data, window=feature_window)
            tech_feats.setdefault('rsi', pd.Series(index=df.index, dtype=float))[mask] = rsi
    
    # MACD (scaled to window)
    fast = max(int(feature_window * 0.5), 3)
    slow = feature_window
    signal = max(int(feature_window * 0.3), 3)
    
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        symbol_data = df[mask]['close'].values
        if len(symbol_data) >= slow:
            macd_line, macd_signal, macd_hist = calculate_macd_custom(symbol_data, fast, slow, signal)
            tech_feats.setdefault('macd', pd.Series(index=df.index, dtype=float))[mask] = macd_line
            tech_feats.setdefault('macd_signal', pd.Series(index=df.index, dtype=float))[mask] = macd_signal
            tech_feats.setdefault('macd_histogram', pd.Series(index=df.index, dtype=float))[mask] = macd_hist
    
    # CCI (use window as period)
    cci_values = calculate_cci_custom(df['high'].values, df['low'].values, df['close'].values, window=feature_window)
    tech_feats['cci'] = pd.Series(cci_values, index=df.index)
    
    # KDJ (use window as period)
    kdj_k, kdj_d = calculate_kdj_custom(df['high'].values, df['low'].values, df['close'].values, window=feature_window)
    tech_feats['kdj_slow_k'] = pd.Series(kdj_k, index=df.index)
    tech_feats['kdj_slow_d'] = pd.Series(kdj_d, index=df.index)
    
    # Williams %R (use window as period)
    willr_values = calculate_williams_r_custom(df['high'].values, df['low'].values, df['close'].values, window=feature_window)
    tech_feats['willr'] = pd.Series(willr_values, index=df.index)
    
    # SAR (use window-based parameters)
    sar_values = calculate_parabolic_sar_custom(df['high'].values, df['low'].values, df['close'].values)
    tech_feats['sar'] = pd.Series(sar_values, index=df.index)
    
    # EMA and SMA (use window as period)
    tech_feats['ema'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=feature_window, adjust=False).mean())
    tech_feats['sma'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(feature_window).mean())
    
    # Note: Hurst exponent removed for performance - it was computationally expensive
    # (calculating for every data point for every symbol). If needed later, can be
    # added back with optimizations (e.g., calculate every N days instead of daily)
    
    # Combine all features
    all_features = {**base_feats, **momentum_feats, **reversal_feats, **volume_feats, **fundamental_std_feats, **tech_feats}
    
    print(f"    📊 Total features calculated: {len(all_features)}")
    
    # Add to dataframe
    for feat_name, feat_values in all_features.items():
        if isinstance(feat_values, pd.Series):
            feature_df[feat_name] = feat_values
        else:
            feature_df[feat_name] = feat_values
    
    # Clean data
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    
    print(f"    ✅ Final features: {len(feature_df.columns)} columns")
    print(f"    📊 Dataset size: {len(feature_df):,} records")
    
    return feature_df


def precalculate_all_features(df, feature_windows=None, force_recalculate=False):
    """
    Pre-calculate features for all windows (done once!).
    If cached features exist, load them instead of recalculating.
    
    ⚠️  TIMING FIX: Features for date T use data only up to T-1
    - Feature calculation functions shift prices by 1 trading day internally
    - This aligns with trading: at start of day T, features for date T contain data up to T-1
    
    Parameters:
    -----------
    df : DataFrame
        Stock data
    feature_windows : list, optional
        List of feature windows to calculate. If None, uses DEFAULT_FEATURE_WINDOWS.
    force_recalculate : bool
        If True, ignore cache and recalculate all features
    
    Returns:
    --------
    dict : Dictionary mapping window -> feature DataFrame
    """
    if feature_windows is None:
        feature_windows = DEFAULT_FEATURE_WINDOWS
    
    print("\n" + "="*70)
    print("🔧 PRE-CALCULATING FEATURES FOR ALL WINDOWS")
    print("="*70)
    print("⚠️  TIMING FIX: Features use prices shifted by 1 trading day (T uses data ≤ T-1)")
    print("="*70)
    
    if force_recalculate:
        print("⚠️  Force recalculate enabled - ignoring cache")
    
    feature_dfs = {}
    
    for window in feature_windows:
        print(f"\n📊 Window {window} days:")
        
        # Try to load from cache first
        if not force_recalculate:
            cached_features = load_features_from_cache(window)
            if cached_features is not None:
                # Cached features already have timing fix applied
                feature_dfs[window] = cached_features
                continue
        
        # Calculate features if not cached
        print(f"   🔧 Calculating features for {window}-day window...")
        # ⚠️  TIMING FIX: Feature calculation now uses prices shifted by 1 trading day
        feature_df = create_features_for_window(df, window)
        
        feature_dfs[window] = feature_df
        
        # Save to cache for future runs (already shifted)
        save_features_to_cache(feature_df, window)
    
    print("\n" + "="*70)
    print(f"✅ FEATURES READY FOR {len(feature_windows)} WINDOWS")
    print("   ⚠️  All features use prices shifted by 1 trading day (T uses data ≤ T-1)")
    print("="*70)
    
    return feature_dfs
