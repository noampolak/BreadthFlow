"""
Hyperparameter Optimizer using Optuna

Implements hyperparameter optimization for machine learning models
using Optuna for efficient parameter search and optimization.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Union
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Optional import for optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Hyperparameter optimization will be limited.")
import joblib

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.
    
    Features:
    - Bayesian optimization with Optuna
    - Support for multiple ML frameworks
    - Time series aware cross-validation
    - Multi-objective optimization
    - Pruning and early stopping
    """
    
    def __init__(
        self, 
        n_trials: int = 100,
        timeout: int = 3600,
        direction: str = "maximize",
        pruner: str = "median"
    ):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            direction: Optimization direction (maximize/minimize)
            pruner: Pruning strategy (median, percentile, etc.)
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        self.pruner = pruner
        self.logger = logging.getLogger(__name__)
        
        # Initialize pruner
        if pruner == "median":
            self.pruner = optuna.pruners.MedianPruner()
        elif pruner == "percentile":
            self.pruner = optuna.pruners.PercentilePruner(25.0)
        else:
            self.pruner = optuna.pruners.NopPruner()
        
        self.logger.info(f"HyperparameterOptimizer initialized with {n_trials} trials")
    
    def optimize_sklearn_model(
        self,
        model_class,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
        cv_folds: int = 5,
        scoring: str = "accuracy",
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a scikit-learn model.
        
        Args:
            model_class: Model class to optimize
            X: Feature matrix
            y: Target vector
            param_space: Parameter space definition
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with optimization results
        """
        try:
            self.logger.info(f"Starting hyperparameter optimization for {model_class.__name__}")
            
            def objective(trial):
                # Sample parameters
                params = {}
                for param_name, param_config in param_space.items():
                    if param_config["type"] == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config["choices"]
                        )
                    elif param_config["type"] == "int":
                        params[param_name] = trial.suggest_int(
                            param_name, 
                            param_config["low"], 
                            param_config["high"],
                            step=param_config.get("step", 1)
                        )
                    elif param_config["type"] == "float":
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config["low"],
                            param_config["high"],
                            step=param_config.get("step", None),
                            log=param_config.get("log", False)
                        )
                
                # Create model with sampled parameters
                model = model_class(**params)
                
                # Use time series split for financial data
                if hasattr(model, 'fit') and hasattr(model, 'predict'):
                    tscv = TimeSeriesSplit(n_splits=cv_folds)
                    scores = cross_val_score(
                        model, X, y, 
                        cv=tscv, 
                        scoring=scoring, 
                        n_jobs=n_jobs
                    )
                    return scores.mean()
                else:
                    raise ValueError("Model must implement fit and predict methods")
            
            # Create study
            study = optuna.create_study(
                direction=self.direction,
                pruner=self.pruner
            )
            
            # Optimize
            study.optimize(
                objective, 
                n_trials=self.n_trials, 
                timeout=self.timeout
            )
            
            # Get results
            best_params = study.best_params
            best_score = study.best_value
            
            self.logger.info(f"Optimization completed. Best score: {best_score:.4f}")
            
            return {
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": len(study.trials),
                "study": study
            }
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            raise
    
    def optimize_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        scoring: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with optimization results
        """
        try:
            import xgboost as xgb
            
            param_space = {
                "n_estimators": {"type": "int", "low": 50, "high": 1000, "step": 50},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                "reg_alpha": {"type": "float", "low": 0, "high": 10, "log": True},
                "reg_lambda": {"type": "float", "low": 0, "high": 10, "log": True}
            }
            
            return self.optimize_sklearn_model(
                xgb.XGBClassifier,
                X, y, param_space, cv_folds, scoring
            )
            
        except ImportError:
            self.logger.error("XGBoost not available")
            raise
        except Exception as e:
            self.logger.error(f"Error optimizing XGBoost: {str(e)}")
            raise
    
    def optimize_lightgbm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        scoring: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with optimization results
        """
        try:
            import lightgbm as lgb
            
            param_space = {
                "n_estimators": {"type": "int", "low": 50, "high": 1000, "step": 50},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                "reg_alpha": {"type": "float", "low": 0, "high": 10, "log": True},
                "reg_lambda": {"type": "float", "low": 0, "high": 10, "log": True},
                "num_leaves": {"type": "int", "low": 10, "high": 100}
            }
            
            return self.optimize_sklearn_model(
                lgb.LGBMClassifier,
                X, y, param_space, cv_folds, scoring
            )
            
        except ImportError:
            self.logger.error("LightGBM not available")
            raise
        except Exception as e:
            self.logger.error(f"Error optimizing LightGBM: {str(e)}")
            raise
    
    def optimize_random_forest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        scoring: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Optimize Random Forest hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with optimization results
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            param_space = {
                "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
                "max_depth": {"type": "int", "low": 3, "high": 20},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
                "bootstrap": {"type": "categorical", "choices": [True, False]}
            }
            
            return self.optimize_sklearn_model(
                RandomForestClassifier,
                X, y, param_space, cv_folds, scoring
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing Random Forest: {str(e)}")
            raise
    
    def optimize_svm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        scoring: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Optimize SVM hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with optimization results
        """
        try:
            from sklearn.svm import SVC
            
            param_space = {
                "C": {"type": "float", "low": 0.1, "high": 100, "log": True},
                "gamma": {"type": "float", "low": 0.001, "high": 1, "log": True},
                "kernel": {"type": "categorical", "choices": ["rbf", "poly", "sigmoid"]},
                "degree": {"type": "int", "low": 2, "high": 5}
            }
            
            return self.optimize_sklearn_model(
                SVC,
                X, y, param_space, cv_folds, scoring
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing SVM: {str(e)}")
            raise
    
    def multi_objective_optimization(
        self,
        model_class,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
        objectives: List[str] = ["accuracy", "precision", "recall"],
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform multi-objective optimization.
        
        Args:
            model_class: Model class to optimize
            X: Feature matrix
            y: Target vector
            param_space: Parameter space definition
            objectives: List of objectives to optimize
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with optimization results
        """
        try:
            self.logger.info(f"Starting multi-objective optimization for {model_class.__name__}")
            
            def objective(trial):
                # Sample parameters
                params = {}
                for param_name, param_config in param_space.items():
                    if param_config["type"] == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config["choices"]
                        )
                    elif param_config["type"] == "int":
                        params[param_name] = trial.suggest_int(
                            param_name, 
                            param_config["low"], 
                            param_config["high"],
                            step=param_config.get("step", 1)
                        )
                    elif param_config["type"] == "float":
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config["low"],
                            param_config["high"],
                            step=param_config.get("step", None),
                            log=param_config.get("log", False)
                        )
                
                # Create model with sampled parameters
                model = model_class(**params)
                
                # Calculate multiple objectives
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                scores = {}
                
                for obj in objectives:
                    if obj == "accuracy":
                        scores[obj] = cross_val_score(model, X, y, cv=tscv, scoring="accuracy").mean()
                    elif obj == "precision":
                        scores[obj] = cross_val_score(model, X, y, cv=tscv, scoring="precision_macro").mean()
                    elif obj == "recall":
                        scores[obj] = cross_val_score(model, X, y, cv=tscv, scoring="recall_macro").mean()
                    elif obj == "f1":
                        scores[obj] = cross_val_score(model, X, y, cv=tscv, scoring="f1_macro").mean()
                
                return tuple(scores[obj] for obj in objectives)
            
            # Create study for multi-objective optimization
            study = optuna.create_study(
                directions=["maximize"] * len(objectives),
                pruner=self.pruner
            )
            
            # Optimize
            study.optimize(
                objective, 
                n_trials=self.n_trials, 
                timeout=self.timeout
            )
            
            # Get Pareto front
            pareto_front = study.best_trials
            
            self.logger.info(f"Multi-objective optimization completed. Found {len(pareto_front)} Pareto optimal solutions")
            
            return {
                "pareto_front": pareto_front,
                "objectives": objectives,
                "n_trials": len(study.trials),
                "study": study
            }
            
        except Exception as e:
            self.logger.error(f"Error in multi-objective optimization: {str(e)}")
            raise
    
    def save_study(self, study, filepath: str) -> None:
        """
        Save optimization study to file.
        
        Args:
            study: Optuna study object
            filepath: Path to save the study
        """
        try:
            joblib.dump(study, filepath)
            self.logger.info(f"Study saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving study: {str(e)}")
            raise
    
    def load_study(self, filepath: str):
        """
        Load optimization study from file.
        
        Args:
            filepath: Path to the study file
            
        Returns:
            Loaded Optuna study
        """
        try:
            study = joblib.load(filepath)
            self.logger.info(f"Study loaded from {filepath}")
            return study
        except Exception as e:
            self.logger.error(f"Error loading study: {str(e)}")
            raise
