"""
Model training functions for individual models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb

# Try to import training_feature_utils for NON_FEATURE_COLS
try:
    from training_feature_utils import NON_FEATURE_COLS
except ImportError:
    # Fallback if not available
    NON_FEATURE_COLS = [
        'symbol', 'date', 'quarter_end', 'fiscal_year', 
        'fiscal_period', 'frame', 'year'
    ]

from .config import CLASS_LABELS


def train_single_model_optimized(X, y, model_name, feature_window, target_window, train_split=0.6):
    """
    Train a single model with TIME-BASED split (not random shuffle).
    
    CRITICAL: For financial time series, we must split chronologically:
    - Train on EARLY data (e.g., 2015-2022)
    - Test on LATE data (e.g., 2023-2024)
    
    This prevents data leakage and simulates real trading conditions.
    
    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target
    model_name : str
        Model type ('RandomForest', 'XGBoost', 'GradientBoosting', 'SVM', 'LSTM')
    feature_window : int
        Feature window size
    target_window : int
        Target window size
    train_split : float, default=0.6
        Fraction of data to use for training (e.g., 0.6 = 60% train, 40% test)
        Must be between 0 and 1.
    
    Returns:
    --------
    model : trained model
    metrics : dict with evaluation metrics
    """
    
    # ============================================================
    # STEP 1: TIME-BASED SPLIT (Chronological)
    # ============================================================
    
    # Ensure data is sorted chronologically
    if isinstance(X, pd.DataFrame):
        X = X.sort_index()
    if isinstance(y, pd.Series):
        y = y.sort_index()
    
    # Calculate split index
    split_idx = int(len(X) * train_split)
    
    # Split chronologically - FIRST train_split% for training, LAST (1-train_split)% for testing
    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()
    
    print(f"     🔍 {model_name} - Time-based split ({train_split*100:.0f}%/{100-train_split*100:.0f}%):")
    print(f"        Train: {len(X_train):,} samples (earliest {len(X_train)/len(X)*100:.1f}%)")
    print(f"        Test:  {len(X_test):,} samples (latest {len(X_test)/len(X)*100:.1f}%)")
    
    # Show class distribution to ensure balance
    train_dist = y_train.value_counts().sort_index()
    test_dist = y_test.value_counts().sort_index()
    print(f"        Train classes: {dict(train_dist)}")
    print(f"        Test classes:  {dict(test_dist)}")
    
    # ============================================================
    # STEP 2: 5-LAYER DATA CLEANING
    # ============================================================
    
    # Layer 1: Remove all-NaN columns
    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
        print(f"        🧹 Removing {len(all_nan_cols)} all-NaN columns")
        X_train = X_train.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols)
    
    # Layer 2: Remove high-NaN columns (>80%)
    high_nan_cols = X_train.columns[X_train.isna().mean() > 0.8].tolist()
    if high_nan_cols:
        print(f"        🧹 Removing {len(high_nan_cols)} high-NaN columns (>80%)")
        X_train = X_train.drop(columns=high_nan_cols)
        X_test = X_test.drop(columns=high_nan_cols)
    
    # Layer 3: Ensure all columns are numeric
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < len(X_train.columns):
        print(f"        🧹 Keeping only {len(numeric_cols)} numeric columns")
        X_train = X_train[numeric_cols]
        X_test = X_test[numeric_cols]
    
    # Layer 4: Smart median filling
    for col in X_train.columns:
        if X_train[col].isna().any():
            median_val = X_train[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            X_train[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)
    
    # Layer 5: Outlier clipping (1st to 99th percentile)
    for col in X_train.columns:
        if X_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            q01 = X_train[col].quantile(0.01)
            q99 = X_train[col].quantile(0.99)
            X_train[col] = X_train[col].clip(q01, q99)
            X_test[col] = X_test[col].clip(q01, q99)
    
    print(f"        ✅ Clean data: {X_train.shape[1]} features")
    
    # ============================================================
    # STEP 3: TRAIN MODEL
    # ============================================================
    
    if model_name == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    
    elif model_name == 'XGBoost':
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
    
    elif model_name == 'SVM':
        # SVM with memory optimization
        from sklearn.kernel_approximation import RBFSampler
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        
        max_svm_samples = 200000
        if len(X_train) > max_svm_samples:
            X_train_svm = X_train.iloc[-max_svm_samples:]
            y_train_svm = y_train.iloc[-max_svm_samples:]
            print(f"        ℹ️  SVM: Using last {len(X_train_svm):,} of {len(X_train):,} samples")
        else:
            X_train_svm = X_train
            y_train_svm = y_train
        
        model = Pipeline([
            ('rbf_features', RBFSampler(
                gamma=0.1,
                n_components=400,
                random_state=42
            )),
            ('sgd_classifier', SGDClassifier(
                loss='log_loss',
                alpha=0.00001,
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])
        model.fit(X_train_svm, y_train_svm)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train model (except SVM which was already trained)
    if model_name != 'SVM':
        model.fit(X_train, y_train)
    
    # ============================================================
    # STEP 4: EVALUATE MODEL
    # ============================================================
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    report = classification_report(
        y_test, y_pred,
        target_names=list(CLASS_LABELS.values()),
        output_dict=True,
        zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': report,
        'feature_window': feature_window,
        'target_window': target_window,
        'n_features': X_train.shape[1],
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    return model, metrics

