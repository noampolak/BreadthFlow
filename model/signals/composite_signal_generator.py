"""
Composite Signal Generator

Orchestrates multiple signal generation strategies and combines their outputs
into a unified signal generation system.
"""

from typing import Any, Dict, List, Optional

# Optional pandas import
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
    DataFrame = pd.DataFrame
    Series = pd.Series
except ImportError:
    PANDAS_AVAILABLE = False
    # Create dummy types for type hints when pandas is not available
    DataFrame = Any
    Series = Any
# Import logging components directly to avoid PySpark dependency
import logging
import time
from datetime import datetime

import numpy as np

from .signal_config import SignalConfig
from .signal_generator_interface import SignalGeneratorInterface
from .strategies.base_signal_strategy import BaseSignalStrategy
from .strategies.fundamental_analysis_strategy import FundamentalAnalysisStrategy
from .strategies.technical_analysis_strategy import TechnicalAnalysisStrategy

logger = logging.getLogger(__name__)


class CompositeSignalGenerator(SignalGeneratorInterface):
    """Composite signal generator that combines multiple strategies"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = "composite_signal_generator"
        self.strategies: Dict[str, BaseSignalStrategy] = {}
        # Simplified logging to avoid PySpark dependency
        self.logger = logging.getLogger("composite_signal_generator")
        self.error_handler = None

        # Performance tracking
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_data_points": 0,
            "average_generation_time": 0.0,
        }

        # Default configuration
        self.default_config = {
            "strategy_weights": {"technical_analysis": 0.6, "fundamental_analysis": 0.4},
            "combination_method": "weighted_average",  # 'weighted_average', 'voting', 'ensemble'
            "consensus_threshold": 0.7,
            "enable_strategy_filtering": True,
            "min_strategies_required": 1,
            "signal_aggregation": "mean",  # 'mean', 'median', 'max', 'min'
            "confidence_aggregation": "weighted_mean",  # 'weighted_mean', 'min', 'max'
        }

        # Update config with defaults
        self.config.update(self.default_config)

        # Register default strategies
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default signal generation strategies"""
        # Technical Analysis Strategy
        tech_config = self.config.get("technical_analysis", {})
        tech_strategy = TechnicalAnalysisStrategy("technical_analysis", tech_config)
        self.register_strategy("technical_analysis", tech_strategy)

        # Fundamental Analysis Strategy
        fund_config = self.config.get("fundamental_analysis", {})
        fund_strategy = FundamentalAnalysisStrategy("fundamental_analysis", fund_config)
        self.register_strategy("fundamental_analysis", fund_strategy)

    def register_strategy(self, name: str, strategy: BaseSignalStrategy):
        """Register a signal generation strategy"""
        self.strategies[name] = strategy
        logger.info(f"✅ Registered signal strategy: {name}")

    def get_name(self) -> str:
        """Get the name of this signal generator"""
        return self.name

    def get_supported_strategies(self) -> List[str]:
        """Get list of supported strategy names"""
        return list(self.strategies.keys())

    def get_config(self) -> Dict[str, Any]:
        """Get signal generator configuration"""
        return self.config

    def generate_signals(self, config: SignalConfig, data: Dict[str, Any]):
        """Generate composite signals using multiple strategies"""

        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for signal generation but not available")

        # Simplified performance logging
        start_time = time.time()
        try:
            # Validate configuration
            if not self._validate_config(config):
                raise ValueError("Invalid configuration for composite signal generator")

            # Validate data
            if not self._validate_data(data, config):
                raise ValueError("Invalid data for composite signal generation")

            # Generate signals from each strategy
            strategy_results = self._generate_strategy_signals(config, data)

            if not strategy_results:
                logger.warning("No strategy results generated")
                if PANDAS_AVAILABLE:
                    return DataFrame()
                else:
                    return {}

            # Combine strategy results
            composite_signals = self._combine_strategy_signals(strategy_results, config)

            # Apply consensus filtering if enabled
            if self.config.get("enable_strategy_filtering", True):
                composite_signals = self._apply_consensus_filtering(composite_signals, strategy_results)

            # Calculate final metrics
            composite_signals = self._calculate_final_metrics(composite_signals, strategy_results)

            # Update statistics
            self._update_generation_stats(composite_signals, True)

            # Log success
            logger.info(
                f"Composite signal generation successful: {len(strategy_results)} strategies, {len(composite_signals)} signals"
            )

            return composite_signals

        except Exception as e:
            logger.error(f"Error in composite signal generation: {e}")
            if PANDAS_AVAILABLE:
                self._update_generation_stats(DataFrame(), False)
            else:
                self._update_generation_stats({}, False)
            raise

    def _validate_config(self, config) -> bool:
        """Validate composite signal generator configuration"""
        # Handle both dict and SignalConfig objects
        if isinstance(config, dict):
            # For dict configs, just check if it's not empty
            return bool(config)
        else:
            # For SignalConfig objects, use their validate method
            return config.validate()

        # Check if strategy is supported (only for SignalConfig objects)
        if not isinstance(config, dict) and hasattr(config, "strategy_name"):
            if config.strategy_name not in self.get_supported_strategies():
                logger.error(f"Strategy {config.strategy_name} not found in composite generator")
                return False

        return True

    def _validate_data(self, data: Dict[str, DataFrame], config) -> bool:
        """Validate data for composite signal generation"""
        if not data:
            return False

        # Handle both dict and SignalConfig objects
        if isinstance(config, dict):
            # For dict configs, just check if we have some data
            return bool(data)
        else:
            # Check if all required data is available
            if hasattr(config, "required_resources"):
                for resource in config.required_resources:
                    if resource not in data:
                        logger.error(f"Required resource {resource} not found in data")
                        return False

                    if data[resource].empty:
                        logger.error(f"Resource {resource} data is empty")
                        return False

        return True

    def _generate_strategy_signals(self, config: SignalConfig, data: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        """Generate signals from each registered strategy"""
        strategy_results = {}

        for strategy_name, strategy in self.strategies.items():
            try:
                logger.info(f"Generating signals from strategy: {strategy_name}")

                # Generate signals using the strategy
                signals = strategy.generate_signals(data, config)

                if not signals.empty:
                    strategy_results[strategy_name] = signals
                    logger.info(f"✅ Generated {len(signals)} signals from {strategy_name}")
                else:
                    logger.warning(f"⚠️ No signals generated from {strategy_name}")

            except Exception as e:
                logger.error(f"Error generating signals from {strategy_name}: {e}")
                continue

        return strategy_results

    def _combine_strategy_signals(self, strategy_results: Dict[str, DataFrame], config: SignalConfig) -> DataFrame:
        """Combine signals from multiple strategies"""

        if not strategy_results:
            return DataFrame()

        # Get combination method
        combination_method = self.config.get("combination_method", "weighted_average")

        if combination_method == "weighted_average":
            return self._weighted_average_combination(strategy_results, config)
        elif combination_method == "voting":
            return self._voting_combination(strategy_results, config)
        elif combination_method == "ensemble":
            return self._ensemble_combination(strategy_results, config)
        else:
            logger.warning(f"Unknown combination method: {combination_method}, using weighted average")
            return self._weighted_average_combination(strategy_results, config)

    def _weighted_average_combination(self, strategy_results: Dict[str, DataFrame], config: SignalConfig) -> DataFrame:
        """Combine signals using weighted average"""

        # Get strategy weights
        strategy_weights = self.config.get("strategy_weights", {})

        # Initialize composite signals with the first strategy's data
        first_strategy = list(strategy_results.keys())[0]
        composite_signals = strategy_results[first_strategy].copy()

        # Initialize weighted columns
        composite_signals["weighted_signal_strength"] = 0.0
        composite_signals["weighted_confidence"] = 0.0
        composite_signals["strategy_count"] = 0

        total_weight = 0

        for strategy_name, signals in strategy_results.items():
            weight = strategy_weights.get(strategy_name, 1.0)
            total_weight += weight

            if "signal_strength" in signals.columns:
                composite_signals["weighted_signal_strength"] += signals["signal_strength"] * weight

            if "confidence" in signals.columns:
                composite_signals["weighted_confidence"] += signals["confidence"] * weight

            composite_signals["strategy_count"] += 1

        # Normalize by total weight
        if total_weight > 0:
            composite_signals["signal_strength"] = composite_signals["weighted_signal_strength"] / total_weight
            composite_signals["confidence"] = composite_signals["weighted_confidence"] / total_weight

        # Determine signal type
        composite_signals["signal_type"] = composite_signals["signal_strength"].apply(
            lambda x: "buy" if x > 0.3 else ("sell" if x < -0.3 else "hold")
        )

        return composite_signals

    def _voting_combination(self, strategy_results: Dict[str, DataFrame], config: SignalConfig) -> DataFrame:
        """Combine signals using voting mechanism"""

        # Initialize composite signals
        first_strategy = list(strategy_results.keys())[0]
        composite_signals = strategy_results[first_strategy].copy()

        # Count votes for each signal type
        buy_votes = Series(0, index=composite_signals.index)
        sell_votes = Series(0, index=composite_signals.index)
        hold_votes = Series(0, index=composite_signals.index)

        for strategy_name, signals in strategy_results.items():
            if "signal_type" in signals.columns:
                buy_votes += (signals["signal_type"] == "buy").astype(int)
                sell_votes += (signals["signal_type"] == "sell").astype(int)
                hold_votes += (signals["signal_type"] == "hold").astype(int)

        # Determine consensus signal type
        total_strategies = len(strategy_results)
        consensus_threshold = self.config.get("consensus_threshold", 0.7)
        min_votes = total_strategies * consensus_threshold

        composite_signals["signal_type"] = np.where(
            buy_votes >= min_votes, "buy", np.where(sell_votes >= min_votes, "sell", "hold")
        )

        # Calculate signal strength based on vote distribution
        composite_signals["signal_strength"] = (buy_votes - sell_votes) / total_strategies

        # Calculate confidence based on consensus
        max_votes = pd.concat([buy_votes, sell_votes, hold_votes], axis=1).max(axis=1)
        composite_signals["confidence"] = max_votes / total_strategies

        return composite_signals

    def _ensemble_combination(self, strategy_results: Dict[str, DataFrame], config: SignalConfig) -> DataFrame:
        """Combine signals using ensemble methods"""

        # Initialize composite signals
        first_strategy = list(strategy_results.keys())[0]
        composite_signals = strategy_results[first_strategy].copy()

        # Collect signal strengths from all strategies
        signal_strengths = []
        confidences = []

        for strategy_name, signals in strategy_results.items():
            if "signal_strength" in signals.columns:
                signal_strengths.append(signals["signal_strength"])

            if "confidence" in signals.columns:
                confidences.append(signals["confidence"])

        # Combine using aggregation method
        signal_aggregation = self.config.get("signal_aggregation", "mean")
        confidence_aggregation = self.config.get("confidence_aggregation", "weighted_mean")

        if signal_strengths:
            signal_df = pd.concat(signal_strengths, axis=1)
            if signal_aggregation == "mean":
                composite_signals["signal_strength"] = signal_df.mean(axis=1)
            elif signal_aggregation == "median":
                composite_signals["signal_strength"] = signal_df.median(axis=1)
            elif signal_aggregation == "max":
                composite_signals["signal_strength"] = signal_df.max(axis=1)
            elif signal_aggregation == "min":
                composite_signals["signal_strength"] = signal_df.min(axis=1)

        if confidences:
            confidence_df = pd.concat(confidences, axis=1)
            if confidence_aggregation == "weighted_mean":
                # Weight by signal strength
                weights = abs(signal_df) if signal_strengths else None
                if weights is not None:
                    composite_signals["confidence"] = (confidence_df * weights).sum(axis=1) / weights.sum(axis=1)
                else:
                    composite_signals["confidence"] = confidence_df.mean(axis=1)
            else:
                composite_signals["confidence"] = confidence_df.mean(axis=1)

        # Determine signal type
        composite_signals["signal_type"] = composite_signals["signal_strength"].apply(
            lambda x: "buy" if x > 0.3 else ("sell" if x < -0.3 else "hold")
        )

        return composite_signals

    def _apply_consensus_filtering(self, composite_signals: DataFrame, strategy_results: Dict[str, DataFrame]) -> DataFrame:
        """Apply consensus filtering to remove conflicting signals"""

        min_strategies = self.config.get("min_strategies_required", 1)
        consensus_threshold = self.config.get("consensus_threshold", 0.7)

        # Count strategies that agree on signal direction
        agreement_count = Series(0, index=composite_signals.index)
        total_strategies = len(strategy_results)

        for strategy_name, signals in strategy_results.items():
            if "signal_type" in signals.columns and "signal_type" in composite_signals.columns:
                agreement = (signals["signal_type"] == composite_signals["signal_type"]).astype(int)
                agreement_count += agreement

        # Filter signals based on consensus
        consensus_ratio = agreement_count / total_strategies
        consensus_mask = (consensus_ratio >= consensus_threshold) & (agreement_count >= min_strategies)

        filtered_signals = composite_signals[consensus_mask].copy()

        if len(filtered_signals) < len(composite_signals):
            logger.info(f"Consensus filtering removed {len(composite_signals) - len(filtered_signals)} signals")

        return filtered_signals

    def _calculate_final_metrics(self, composite_signals: DataFrame, strategy_results: Dict[str, DataFrame]) -> DataFrame:
        """Calculate final metrics for composite signals"""

        if composite_signals.empty:
            return composite_signals

        # Add metadata
        composite_signals["composite_generator"] = self.name
        composite_signals["generation_timestamp"] = datetime.now()
        composite_signals["strategies_used"] = len(strategy_results)

        # Calculate signal quality score
        if "signal_strength" in composite_signals.columns and "confidence" in composite_signals.columns:
            composite_signals["signal_quality"] = abs(composite_signals["signal_strength"]) * composite_signals["confidence"]

        return composite_signals

    def _update_generation_stats(self, signals: DataFrame, success: bool):
        """Update generation statistics"""
        self.generation_stats["total_generations"] += 1

        if success:
            self.generation_stats["successful_generations"] += 1
            self.generation_stats["total_data_points"] += len(signals)
        else:
            self.generation_stats["failed_generations"] += 1

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this signal generator"""
        metrics = self.generation_stats.copy()

        if metrics["total_generations"] > 0:
            metrics["success_rate"] = metrics["successful_generations"] / metrics["total_generations"]
        else:
            metrics["success_rate"] = 0.0

        # Add strategy-specific metrics
        strategy_metrics = {}
        for strategy_name, strategy in self.strategies.items():
            strategy_metrics[strategy_name] = strategy.get_performance_metrics()

        metrics["strategy_metrics"] = strategy_metrics

        return metrics

    def get_signal_quality_metrics(self, signals: DataFrame) -> Dict[str, Any]:
        """Calculate signal quality metrics"""
        if signals.empty:
            return {
                "total_signals": 0,
                "signal_strength_avg": 0.0,
                "confidence_avg": 0.0,
                "signal_distribution": {},
                "quality_score_avg": 0.0,
            }

        metrics = {
            "total_signals": len(signals),
            "signal_strength_avg": signals["signal_strength"].mean() if "signal_strength" in signals.columns else 0.0,
            "confidence_avg": signals["confidence"].mean() if "confidence" in signals.columns else 0.0,
            "quality_score_avg": signals["signal_quality"].mean() if "signal_quality" in signals.columns else 0.0,
        }

        # Calculate signal distribution
        if "signal_type" in signals.columns:
            metrics["signal_distribution"] = signals["signal_type"].value_counts().to_dict()

        # Calculate strategy agreement
        if "strategy_count" in signals.columns:
            metrics["avg_strategy_agreement"] = signals["strategy_count"].mean()

        return metrics
