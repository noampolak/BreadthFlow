"""
Cache management functions for features, targets, and training logs.
"""

import os
import json
import pandas as pd
import shutil
from .config import FEATURES_CACHE_DIR, LOG_FILE


def save_features_to_cache(feature_df, window):
    """Save pre-calculated features to disk"""
    os.makedirs(FEATURES_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(FEATURES_CACHE_DIR, f'features_{window}d.pkl')
    print(f"   💾 Saving features to cache: {cache_file}")
    feature_df.to_pickle(cache_file)
    print(f"   ✅ Cached {len(feature_df.columns)} columns, {len(feature_df):,} records")


def load_features_from_cache(window):
    """Load pre-calculated features from disk if available"""
    cache_file = os.path.join(FEATURES_CACHE_DIR, f'features_{window}d.pkl')
    if os.path.exists(cache_file):
        print(f"   ♻️  Loading cached features from: {cache_file}")
        feature_df = pd.read_pickle(cache_file)
        print(f"   ✅ Loaded {len(feature_df.columns)} columns, {len(feature_df):,} records")
        return feature_df
    return None


def save_target_to_cache(target_series, window):
    """Save pre-calculated target to disk"""
    os.makedirs(FEATURES_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(FEATURES_CACHE_DIR, f'target_{window}d.pkl')
    print(f"   💾 Saving target to cache: {cache_file}")
    target_series.to_pickle(cache_file)


def load_target_from_cache(window):
    """Load pre-calculated target from disk if available"""
    cache_file = os.path.join(FEATURES_CACHE_DIR, f'target_{window}d.pkl')
    if os.path.exists(cache_file):
        print(f"   ♻️  Loading cached target from: {cache_file}")
        target_series = pd.read_pickle(cache_file)
        print(f"   ✅ Loaded target with {len(target_series):,} samples")
        return target_series
    return None


def load_training_log():
    """Load training log to track which models are already trained"""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_training_log(log):
    """Save training log"""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)


def clear_feature_cache():
    """Clear all cached features and targets"""
    if os.path.exists(FEATURES_CACHE_DIR):
        shutil.rmtree(FEATURES_CACHE_DIR)
        os.makedirs(FEATURES_CACHE_DIR)
        print("🗑️  Feature cache cleared!")
    else:
        print("ℹ️  No cache to clear")

