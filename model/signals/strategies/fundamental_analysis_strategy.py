"""
Fundamental Analysis Strategy

Implements fundamental analysis-based signal generation strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import time
from .base_signal_strategy import BaseSignalStrategy
from ..signal_config import SignalConfig
from ..components.fundamental_indicators import FundamentalIndicators

logger = logging.getLogger(__name__)


class FundamentalAnalysisStrategy(BaseSignalStrategy):
    """Fundamental analysis-based signal generation strategy"""

    def __init__(self, name: str = "fundamental_analysis", config: Dict[str, Any] = None):
        super().__init__(name, config)

        # Set required data and timeframes
        self.required_data = ["stock_price", "revenue", "market_cap"]
        self.supported_timeframes = ["1day", "1week", "1month", "quarterly"]

        # Initialize fundamental indicators component
        self.fundamental_indicators = FundamentalIndicators()

        # Default strategy configuration
        self.default_config = {
            "indicators": ["pe_ratio", "pb_ratio", "roe", "revenue_growth"],
            "valuation_indicators": ["pe_ratio", "pb_ratio", "ps_ratio"],
            "growth_indicators": ["revenue_growth", "earnings_growth"],
            "profitability_indicators": ["roe", "roa", "gross_margin"],
            "positive_indicators": ["roe", "roa", "gross_margin", "revenue_growth"],
            "negative_indicators": ["pe_ratio", "pb_ratio", "debt_to_equity"],
            "valuation_weights": {"pe_ratio": 0.3, "pb_ratio": 0.3, "ps_ratio": 0.2, "ev_ebitda": 0.2},
            "growth_weights": {"revenue_growth": 0.5, "earnings_growth": 0.5},
            "profitability_weights": {"roe": 0.4, "roa": 0.3, "gross_margin": 0.3},
            "signal_threshold": 0.5,
            "confidence_threshold": 0.7,
        }

        # Update config with defaults
        self.config.update(self.default_config)

    def generate_signals(self, data: Dict[str, pd.DataFrame], config: SignalConfig) -> pd.DataFrame:
        """Generate fundamental analysis signals"""

        start_time = time.time()
        success = False

        try:
            # Validate data
            if not self.validate_data(data):
                raise ValueError("Invalid data for fundamental analysis strategy")

            # Preprocess data
            processed_data = self.preprocess_data(data, config)

            if not processed_data:
                logger.error("No valid data after preprocessing")
                return pd.DataFrame()

            # Get stock price data as base
            stock_data = processed_data.get("stock_price")
            if stock_data is None or stock_data.empty:
                logger.error("No stock price data available")
                return pd.DataFrame()

            # Merge fundamental data with stock data
            signals = self._merge_fundamental_data(stock_data, processed_data)

            # Generate fundamental indicators
            indicators = self.config.get("indicators", ["pe_ratio", "pb_ratio", "roe", "revenue_growth"])
            signals = self.fundamental_indicators.generate_fundamental_signals(signals, indicators, self.config)

            # Generate valuation signals
            signals = self._generate_valuation_signals(signals)

            # Generate growth signals
            signals = self._generate_growth_signals(signals)

            # Generate profitability signals
            signals = self._generate_profitability_signals(signals)

            # Calculate composite fundamental score
            signals = self._calculate_fundamental_score(signals)

            # Calculate signal strength and confidence
            signals = self._calculate_signal_metrics(signals)

            # Postprocess signals
            signals = self.postprocess_signals(signals, config)

            success = True
            return signals

        except Exception as e:
            logger.error(f"Error generating fundamental analysis signals: {e}")
            return pd.DataFrame()

        finally:
            # Update performance stats
            generation_time = time.time() - start_time
            self.update_performance_stats(success, generation_time)

    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate that required data is available"""
        required_resources = ["stock_price", "revenue", "market_cap"]

        for resource in required_resources:
            if resource not in data:
                logger.error(f"Fundamental analysis strategy requires {resource} data")
                return False

            if data[resource].empty:
                logger.error(f"{resource} data is empty")
                return False

        # Check stock price data has required columns
        stock_data = data["stock_price"]
        required_columns = ["close", "date"]
        missing_columns = [col for col in required_columns if col not in stock_data.columns]

        if missing_columns:
            logger.error(f"Missing required columns in stock data: {missing_columns}")
            return False

        return True

    def _merge_fundamental_data(self, stock_data: pd.DataFrame, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge fundamental data with stock price data"""
        signals = stock_data.copy()

        # Merge revenue data
        if "revenue" in processed_data:
            revenue_data = processed_data["revenue"]
            if not revenue_data.empty and "date" in revenue_data.columns:
                # Forward fill revenue data to match stock data frequency
                revenue_data = revenue_data.set_index("date").reindex(signals["date"], method="ffill").reset_index()
                signals = pd.merge(signals, revenue_data, on="date", how="left", suffixes=("", "_revenue"))

        # Merge market cap data
        if "market_cap" in processed_data:
            market_cap_data = processed_data["market_cap"]
            if not market_cap_data.empty and "date" in market_cap_data.columns:
                # Forward fill market cap data
                market_cap_data = market_cap_data.set_index("date").reindex(signals["date"], method="ffill").reset_index()
                signals = pd.merge(signals, market_cap_data, on="date", how="left", suffixes=("", "_market_cap"))

        return signals

    def _generate_valuation_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate valuation-based signals"""
        signals = data.copy()

        valuation_indicators = self.config.get("valuation_indicators", ["pe_ratio", "pb_ratio", "ps_ratio"])
        thresholds = self.config.get("valuation_thresholds", {})

        # Generate valuation signals using fundamental indicators
        signals = self.fundamental_indicators.get_valuation_signals(signals, valuation_indicators, thresholds)

        # Calculate composite valuation score
        valuation_weights = self.config.get("valuation_weights", {})
        positive_indicators = [f"{ind}_undervalued" for ind in valuation_indicators]
        negative_indicators = [f"{ind}_overvalued" for ind in valuation_indicators]

        signals["valuation_score"] = self.fundamental_indicators.get_fundamental_score(
            signals, positive_indicators, negative_indicators, valuation_weights
        )

        return signals

    def _generate_growth_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate growth-based signals"""
        signals = data.copy()

        growth_indicators = self.config.get("growth_indicators", ["revenue_growth", "earnings_growth"])
        growth_weights = self.config.get("growth_weights", {})

        # Calculate growth score
        signals["growth_score"] = self.fundamental_indicators.get_fundamental_score(
            signals, growth_indicators, [], growth_weights
        )

        # Generate growth signals
        growth_threshold = self.config.get("growth_threshold", 0.1)  # 10% growth threshold
        signals["growth_buy"] = (signals["growth_score"] > growth_threshold).astype(int)
        signals["growth_sell"] = (signals["growth_score"] < -growth_threshold).astype(int)
        signals["growth_neutral"] = (
            (signals["growth_score"] >= -growth_threshold) & (signals["growth_score"] <= growth_threshold)
        ).astype(int)

        return signals

    def _generate_profitability_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate profitability-based signals"""
        signals = data.copy()

        profitability_indicators = self.config.get("profitability_indicators", ["roe", "roa", "gross_margin"])
        profitability_weights = self.config.get("profitability_weights", {})

        # Calculate profitability score
        signals["profitability_score"] = self.fundamental_indicators.get_fundamental_score(
            signals, profitability_indicators, [], profitability_weights
        )

        # Generate profitability signals
        profitability_threshold = self.config.get("profitability_threshold", 0.15)  # 15% ROE threshold
        signals["profitability_buy"] = (signals["profitability_score"] > profitability_threshold).astype(int)
        signals["profitability_sell"] = (signals["profitability_score"] < 0.05).astype(int)  # 5% minimum
        signals["profitability_neutral"] = (
            (signals["profitability_score"] >= 0.05) & (signals["profitability_score"] <= profitability_threshold)
        ).astype(int)

        return signals

    def _calculate_fundamental_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite fundamental score"""
        signals = data.copy()

        # Get component scores
        component_scores = []
        component_weights = {"valuation_score": 0.3, "growth_score": 0.4, "profitability_score": 0.3}

        for score_col, weight in component_weights.items():
            if score_col in signals.columns:
                # Normalize score to 0-1 range
                score_data = signals[score_col]
                if score_data.dtype in ["float64", "int64"]:
                    normalized_score = (score_data - score_data.min()) / (score_data.max() - score_data.min())
                    component_scores.append(normalized_score * weight)

        if component_scores:
            # Calculate weighted average
            fundamental_score = pd.concat(component_scores, axis=1).sum(axis=1)
        else:
            # Fallback to simple average of available indicators
            positive_indicators = self.config.get("positive_indicators", [])
            negative_indicators = self.config.get("negative_indicators", [])

            fundamental_score = self.fundamental_indicators.get_fundamental_score(
                signals, positive_indicators, negative_indicators
            )

        signals["fundamental_score"] = fundamental_score

        return signals

    def _calculate_signal_metrics(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate signal strength and confidence metrics"""

        # Calculate signal strength based on fundamental score
        if "fundamental_score" in signals.columns:
            # Normalize fundamental score to -1 to 1 range
            fundamental_score = signals["fundamental_score"]
            signal_strength = (
                2 * (fundamental_score - fundamental_score.min()) / (fundamental_score.max() - fundamental_score.min()) - 1
            )
            signals["signal_strength"] = signal_strength
        else:
            signals["signal_strength"] = 0

        # Calculate signal confidence based on multiple factors
        confidence_factors = []

        # Valuation confidence (more extreme valuations = higher confidence)
        if "valuation_score" in signals.columns:
            valuation_confidence = abs(signals["valuation_score"] - 0.5) * 2
            confidence_factors.append(valuation_confidence)

        # Growth confidence (stronger growth = higher confidence)
        if "growth_score" in signals.columns:
            growth_confidence = abs(signals["growth_score"])
            confidence_factors.append(growth_confidence)

        # Profitability confidence (higher profitability = higher confidence)
        if "profitability_score" in signals.columns:
            profitability_confidence = signals["profitability_score"]
            confidence_factors.append(profitability_confidence)

        # Calculate average confidence
        if confidence_factors:
            confidence = pd.concat(confidence_factors, axis=1).mean(axis=1)
        else:
            confidence = pd.Series(0.5, index=signals.index)

        signals["confidence"] = confidence

        # Determine signal type
        signals["signal_type"] = np.where(
            signals["signal_strength"] > 0.3, "buy", np.where(signals["signal_strength"] < -0.3, "sell", "hold")
        )

        return signals

    def get_strategy_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get individual strategy signals"""
        signals = {}

        # Valuation strategy
        if "valuation_score" in data.columns:
            signals["valuation_strategy"] = np.where(
                data["valuation_score"] > 0.7, 1, np.where(data["valuation_score"] < 0.3, -1, 0)
            )

        # Growth strategy
        if "growth_score" in data.columns:
            signals["growth_strategy"] = np.where(data["growth_score"] > 0.1, 1, np.where(data["growth_score"] < -0.1, -1, 0))

        # Profitability strategy
        if "profitability_score" in data.columns:
            signals["profitability_strategy"] = np.where(
                data["profitability_score"] > 0.15, 1, np.where(data["profitability_score"] < 0.05, -1, 0)
            )

        # Composite fundamental strategy
        if "fundamental_score" in data.columns:
            signals["fundamental_strategy"] = np.where(
                data["fundamental_score"] > 0.6, 1, np.where(data["fundamental_score"] < 0.4, -1, 0)
            )

        return signals

    def get_signal_summary(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of generated signals"""
        if signals.empty:
            return {
                "total_signals": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "hold_signals": 0,
                "average_strength": 0.0,
                "average_confidence": 0.0,
                "valuation_score_avg": 0.0,
                "growth_score_avg": 0.0,
                "profitability_score_avg": 0.0,
            }

        summary = {
            "total_signals": len(signals),
            "buy_signals": len(signals[signals["signal_type"] == "buy"]),
            "sell_signals": len(signals[signals["signal_type"] == "sell"]),
            "hold_signals": len(signals[signals["signal_type"] == "hold"]),
            "average_strength": signals["signal_strength"].mean() if "signal_strength" in signals.columns else 0.0,
            "average_confidence": signals["confidence"].mean() if "confidence" in signals.columns else 0.0,
            "valuation_score_avg": signals["valuation_score"].mean() if "valuation_score" in signals.columns else 0.0,
            "growth_score_avg": signals["growth_score"].mean() if "growth_score" in signals.columns else 0.0,
            "profitability_score_avg": signals["profitability_score"].mean()
            if "profitability_score" in signals.columns
            else 0.0,
        }

        # Calculate signal distribution
        if "signal_type" in signals.columns:
            summary["signal_distribution"] = signals["signal_type"].value_counts().to_dict()

        return summary
