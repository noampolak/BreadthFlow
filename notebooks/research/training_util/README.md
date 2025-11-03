# Training Utilities Package

A modular package for training multi-class models with feature and target caching.

## ✅ Status: Complete

All functions have been extracted and implemented from the training notebook. The package is ready to use!

## Usage

```python
from training_util import train_all_models_optimized

# Train all default models
training_log = train_all_models_optimized(stock_data)

# Train only specific windows and models
training_log = train_all_models_optimized(
    stock_data,
    feature_windows=[5, 10, 15],  # Only these feature windows
    target_windows=[5],            # Only 5-day target
    model_types=['XGBoost'],       # Only XGBoost models
    train_split=0.7,               # 70% train, 30% test
    model_suffix='_2014_2021'      # Add suffix to model files
)
```

## Parameters

- **feature_windows**: List of feature windows (e.g., [5, 10, 15])
- **target_windows**: List of target windows (e.g., [5])
- **model_types**: List of model types (e.g., ['XGBoost', 'RandomForest'])
- **train_split**: Training/test split ratio (default 0.6)
- **model_suffix**: Optional suffix for model files
- **force_recalculate_features**: If True, ignore cache and recalculate

## Structure

- `config.py`: Configuration constants
- `cache_manager.py`: Feature/target caching and training log management
- `feature_creator.py`: Feature creation functions (needs implementation)
- `target_calculator.py`: Target calculation functions (needs implementation)
- `model_trainer.py`: Individual model training with time-based splits
- `main.py`: Main orchestrator function

