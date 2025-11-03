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
    for ratio in ['pe', 'pb', 'ps', 'pcf']:
        if ratio in df.columns:
            ratio_pct = df.groupby('symbol')[ratio].pct_change(window)
            features[f'mom_{ratio}'] = ratio_pct
            features[f'mom_{ratio}_std'] = df.groupby('symbol')[ratio].transform(lambda x: x.rolling(window).std())
    
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
    for ratio in ['pe', 'pb', 'ps', 'pcf']:
        if ratio in df.columns:
            ratio_pct = -df.groupby('symbol')[ratio].pct_change(window)
            features[f'rev_{ratio}'] = ratio_pct
            features[f'rev_{ratio}_std'] = df.groupby('symbol')[ratio].transform(lambda x: x.rolling(window).std())
    
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


def create_features_for_window(df, feature_window):
    """
    Create features based on a specific window size using paper's approach.
    All ~80-100 features are calculated using the SAME window.
    
    Parameters:
    -----------
    df : DataFrame
        Stock data
    feature_window : int
        Feature calculation window in days (5, 10, 15, 20, 25, 30)
    
    Returns:
    --------
    feature_df : DataFrame
        DataFrame with ~80-100 window-specific features
    """
    print(f"\n  📊 Creating comprehensive features for {feature_window}-day window...")
    
    # Start with basic columns
    feature_df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # Add fundamental data if available
    fundamental_cols = ['pe', 'pb', 'ps', 'pcf', 'market_cap', 'cir_cap']
    for col in fundamental_cols:
        if col in df.columns:
            feature_df[col] = df[col]
    
    # Add sector (categorical)
    if 'sector' in df.columns:
        feature_df['sector'] = df['sector']
    
    # Calculate ALL features using the SAME window
    print(f"    🔧 Calculating momentum features...")
    momentum_feats = calculate_momentum_features(df, feature_window)
    
    print(f"    🔧 Calculating reversal features...")
    reversal_feats = calculate_reversal_features(df, feature_window)
    
    print(f"    🔧 Calculating volume features...")
    volume_feats = calculate_volume_features(df, feature_window)
    
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
    
    # Combine all features
    all_features = {**momentum_feats, **reversal_feats, **volume_feats, **tech_feats}
    
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
