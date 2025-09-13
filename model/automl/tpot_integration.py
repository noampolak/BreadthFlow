"""
TPOT Integration

Integration with TPOT (Tree-based Pipeline Optimization Tool)
for automated machine learning pipeline optimization.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import TPOT
try:
    from tpot import TPOTClassifier, TPOTRegressor
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False
    TPOTClassifier = None
    TPOTRegressor = None

logger = logging.getLogger(__name__)


class TPOTIntegration:
    """
    Integration with TPOT for automated machine learning pipeline optimization.
    
    Features:
    - Automated pipeline optimization
    - Genetic programming for pipeline search
    - Cross-validation and evaluation
    - Pipeline export and import
    """
    
    def __init__(
        self, 
        generations: int = 5, 
        population_size: int = 20,
        cv: int = 5,
        random_state: int = 42
    ):
        """
        Initialize TPOT integration.
        
        Args:
            generations: Number of generations for optimization
            population_size: Population size for genetic algorithm
            cv: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.generations = generations
        self.population_size = population_size
        self.cv = cv
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        if not TPOT_AVAILABLE:
            self.logger.warning("TPOT not available. Install with: pip install tpot")
        
        self.logger.info(f"TPOTIntegration initialized (available: {TPOT_AVAILABLE})")
    
    def train_classification_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: str = "accuracy",
        max_time_mins: int = None,
        max_eval_time_mins: int = 5
    ) -> Dict[str, Any]:
        """
        Train a classification pipeline using TPOT.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            scoring: Scoring metric
            max_time_mins: Maximum time in minutes
            max_eval_time_mins: Maximum evaluation time per pipeline
            
        Returns:
            Dictionary with training results
        """
        try:
            if not TPOT_AVAILABLE:
                return {"error": "TPOT not available"}
            
            self.logger.info("Starting TPOT classification pipeline training")
            start_time = datetime.now()
            
            # Initialize TPOT classifier
            tpot = TPOTClassifier(
                generations=self.generations,
                population_size=self.population_size,
                cv=self.cv,
                random_state=self.random_state,
                scoring=scoring,
                max_time_mins=max_time_mins,
                max_eval_time_mins=max_eval_time_mins,
                verbosity=2,
                n_jobs=1  # Single job for stability
            )
            
            # Train the pipeline
            tpot.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = tpot.predict(X_train)
            y_test_pred = tpot.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            train_metrics = {
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "train_precision": precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                "train_recall": recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
                "train_f1": f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
            }
            
            test_metrics = {
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "test_precision": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                "test_recall": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
                "test_f1": f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            }
            
            # Get pipeline information
            pipeline_info = {
                "model": tpot,
                "best_pipeline": str(tpot.fitted_pipeline_),
                "cv_scores": tpot.cv_results_,
                "training_time": (datetime.now() - start_time).total_seconds(),
                "generations_completed": tpot.generations_completed,
                "best_score": tpot.best_score_
            }
            
            self.logger.info(f"TPOT classification training completed in {pipeline_info['training_time']:.2f}s")
            
            return {
                "success": True,
                "model": tpot,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "pipeline_info": pipeline_info
            }
            
        except Exception as e:
            self.logger.error(f"Error in TPOT classification: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def train_regression_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: str = "neg_mean_squared_error",
        max_time_mins: int = None,
        max_eval_time_mins: int = 5
    ) -> Dict[str, Any]:
        """
        Train a regression pipeline using TPOT.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            scoring: Scoring metric
            max_time_mins: Maximum time in minutes
            max_eval_time_mins: Maximum evaluation time per pipeline
            
        Returns:
            Dictionary with training results
        """
        try:
            if not TPOT_AVAILABLE:
                return {"error": "TPOT not available"}
            
            self.logger.info("Starting TPOT regression pipeline training")
            start_time = datetime.now()
            
            # Initialize TPOT regressor
            tpot = TPOTRegressor(
                generations=self.generations,
                population_size=self.population_size,
                cv=self.cv,
                random_state=self.random_state,
                scoring=scoring,
                max_time_mins=max_time_mins,
                max_eval_time_mins=max_eval_time_mins,
                verbosity=2,
                n_jobs=1  # Single job for stability
            )
            
            # Train the pipeline
            tpot.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = tpot.predict(X_train)
            y_test_pred = tpot.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            train_metrics = {
                "train_mse": mean_squared_error(y_train, y_train_pred),
                "train_mae": mean_absolute_error(y_train, y_train_pred),
                "train_r2": r2_score(y_train, y_train_pred)
            }
            
            test_metrics = {
                "test_mse": mean_squared_error(y_test, y_test_pred),
                "test_mae": mean_absolute_error(y_test, y_test_pred),
                "test_r2": r2_score(y_test, y_test_pred)
            }
            
            # Get pipeline information
            pipeline_info = {
                "model": tpot,
                "best_pipeline": str(tpot.fitted_pipeline_),
                "cv_scores": tpot.cv_results_,
                "training_time": (datetime.now() - start_time).total_seconds(),
                "generations_completed": tpot.generations_completed,
                "best_score": tpot.best_score_
            }
            
            self.logger.info(f"TPOT regression training completed in {pipeline_info['training_time']:.2f}s")
            
            return {
                "success": True,
                "model": tpot,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "pipeline_info": pipeline_info
            }
            
        except Exception as e:
            self.logger.error(f"Error in TPOT regression: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def export_pipeline(self, model, filepath: str) -> bool:
        """
        Export the best pipeline to a Python file.
        
        Args:
            model: Trained TPOT model
            filepath: Path to save the pipeline
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not TPOT_AVAILABLE:
                return False
            
            model.export(filepath)
            self.logger.info(f"Pipeline exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting pipeline: {str(e)}")
            return False
    
    def get_pipeline_code(self, model) -> str:
        """
        Get the Python code for the best pipeline.
        
        Args:
            model: Trained TPOT model
            
        Returns:
            Python code string
        """
        try:
            if not TPOT_AVAILABLE:
                return ""
            
            return str(model.fitted_pipeline_)
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline code: {str(e)}")
            return ""
    
    def get_evaluation_history(self, model) -> Dict[str, Any]:
        """
        Get the evaluation history of the optimization process.
        
        Args:
            model: Trained TPOT model
            
        Returns:
            Dictionary with evaluation history
        """
        try:
            if not TPOT_AVAILABLE:
                return {"error": "TPOT not available"}
            
            history = {
                "cv_results": model.cv_results_,
                "generations_completed": model.generations_completed,
                "best_score": model.best_score_,
                "best_pipeline": str(model.fitted_pipeline_)
            }
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error getting evaluation history: {str(e)}")
            return {"error": str(e)}
    
    def is_available(self) -> bool:
        """
        Check if TPOT is available.
        
        Returns:
            True if available, False otherwise
        """
        return TPOT_AVAILABLE
