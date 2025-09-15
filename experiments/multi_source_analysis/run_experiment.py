"""
Multi-Source Analysis Experiment Runner

This script demonstrates how to use the generic feature engineering modules
to create a comprehensive multi-source analysis experiment.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from features import TechnicalIndicators, FinancialFundamentals, MarketMicrostructure, TimeFeatures, FeatureUtils


class MultiSourceExperiment:
    """Multi-source analysis experiment using generic feature modules."""

    def __init__(self, config_path: str):
        """Initialize experiment with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.experiment_name = self.config["experiment_name"]
        self.results_dir = self.config["output"]["results_directory"]
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize feature calculators
        self.technical_calc = TechnicalIndicators()
        self.financial_calc = FinancialFundamentals()
        self.microstructure_calc = MarketMicrostructure()
        self.time_calc = TimeFeatures()
        self.utils = FeatureUtils()

    def fetch_market_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Create sample market data for demonstration."""
        try:
            # Generate realistic price data
            dates = pd.date_range(start=start_date, end=end_date, freq="D")

            # Create realistic OHLCV data
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol

            # Start with a base price
            base_price = 100 + hash(symbol) % 500

            # Generate price movements
            returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% volatility
            prices = base_price * np.exp(np.cumsum(returns))

            # Create OHLCV data
            data = {
                "Open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                "Close": prices,
                "Volume": np.random.randint(1000000, 10000000, len(dates)),
            }

            df = pd.DataFrame(data, index=dates)

            # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
            df["High"] = df[["Open", "Close"]].max(axis=1) + np.abs(np.random.normal(0, 0.005, len(dates)))
            df["Low"] = df[["Open", "Close"]].min(axis=1) - np.abs(np.random.normal(0, 0.005, len(dates)))

            return df

        except Exception as e:
            print(f"Error creating sample data for {symbol}: {e}")
            return None

    def fetch_financial_data(self, symbol: str) -> pd.DataFrame:
        """Fetch financial fundamental data."""
        try:
            # Use existing data pipeline service for fundamentals
            response = requests.post(
                "http://localhost:8001/data/fundamentals",
                json={
                    "symbol": symbol,
                    "start_date": self.config["data"]["start_date"],
                    "end_date": self.config["data"]["end_date"],
                },
            )

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data["data"])
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
                return df
            else:
                print(f"Error fetching fundamentals for {symbol}: {response.text}")
                return None

        except Exception as e:
            print(f"Error fetching fundamentals for {symbol}: {e}")
            return None

    def generate_features(self, market_data: pd.DataFrame, financial_data: pd.DataFrame = None) -> pd.DataFrame:
        """Generate all features using generic modules."""
        features = pd.DataFrame(index=market_data.index)

        # Technical indicators
        if self.config["features"]["technical"]["enabled"]:
            print("Generating technical indicators...")
            technical_features = self.technical_calc.get_all_indicators(market_data)
            features = pd.concat([features, technical_features], axis=1)

        # Financial fundamentals
        if self.config["features"]["financial"]["enabled"] and financial_data is not None:
            print("Generating financial fundamentals...")
            financial_features = self.financial_calc.get_all_fundamentals(financial_data)
            features = pd.concat([features, financial_features], axis=1)

        # Market microstructure
        if self.config["features"]["market_microstructure"]["enabled"]:
            print("Generating market microstructure features...")
            microstructure_features = self.microstructure_calc.get_all_microstructure(market_data)
            features = pd.concat([features, microstructure_features], axis=1)

        # Time features
        if self.config["features"]["time_features"]["enabled"]:
            print("Generating time features...")
            time_features = self.time_calc.get_all_time_features(market_data.index)
            features = pd.concat([features, time_features], axis=1)

        return features

    def engineer_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply additional feature engineering using generic utilities."""
        print("Applying feature engineering...")

        # Handle missing values
        if "missing_values" in self.config["feature_engineering"]:
            missing_config = self.config["feature_engineering"]["missing_values"]
            features = self.utils.handle_missing_values(
                features, method=missing_config["method"], limit=missing_config.get("limit")
            )

        # Remove outliers
        if "outliers" in self.config["feature_engineering"]:
            outlier_config = self.config["feature_engineering"]["outliers"]
            features = self.utils.remove_outliers(
                features, method=outlier_config["method"], threshold=outlier_config["threshold"]
            )

        # Create lag features
        if self.config["feature_engineering"]["lag_features"]["enabled"]:
            lag_config = self.config["feature_engineering"]["lag_features"]
            for col in lag_config["columns"]:
                if col in features.columns:
                    lag_features = self.utils.create_lag_features(
                        features[col], lags=lag_config["lags"], name_prefix=f"{col}_lag"
                    )
                    features = pd.concat([features, lag_features], axis=1)

        # Create rolling features
        if self.config["feature_engineering"]["rolling_features"]["enabled"]:
            rolling_config = self.config["feature_engineering"]["rolling_features"]
            for col in rolling_config["columns"]:
                if col in features.columns:
                    rolling_features = self.utils.create_rolling_features(
                        features[col],
                        windows=rolling_config["windows"],
                        functions=rolling_config["functions"],
                        name_prefix=f"{col}_rolling",
                    )
                    features = pd.concat([features, rolling_features], axis=1)

        # Scale features
        if "scaling" in self.config["feature_engineering"]:
            scaling_config = self.config["feature_engineering"]["scaling"]
            features = self.utils.scale_features(features, method=scaling_config["method"])

        return features

    def create_target(self, market_data: pd.DataFrame) -> pd.Series:
        """Create target variable based on configuration."""
        target_config = self.config["target"]

        if target_config["type"] == "classification":
            # Price change classification
            price_change = market_data["Close"].pct_change()
            target = (price_change > target_config["threshold"]).astype(int)
        else:
            # Regression target
            target = market_data["Close"].pct_change()

        return target

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train models using the existing model training service."""
        print("Training models...")

        # Prepare training data
        train_data = {
            "features": X.to_dict("records"),
            "target": y.tolist(),
            "feature_names": X.columns.tolist(),
            "algorithms": self.config["models"]["algorithms"],
        }

        try:
            # Use existing model training service
            response = requests.post("http://localhost:8003/train-models", json=train_data)

            if response.status_code == 200:
                results = response.json()
                print(f"Training completed successfully!")
                return results
            else:
                print(f"Error training models: {response.text}")
                return None

        except Exception as e:
            print(f"Error training models: {e}")
            return None

    def run_experiment(self):
        """Run the complete multi-source analysis experiment."""
        print(f"Starting experiment: {self.experiment_name}")
        print("=" * 50)

        # Fetch data for all symbols
        all_features = []
        all_targets = []

        for symbol in self.config["data"]["symbols"]:
            print(f"\nProcessing {symbol}...")

            # Fetch market data
            market_data = self.fetch_market_data(symbol, self.config["data"]["start_date"], self.config["data"]["end_date"])

            if market_data is None:
                print(f"Skipping {symbol} - no market data available")
                continue

            # Fetch financial data
            financial_data = self.fetch_financial_data(symbol)

            # Generate features
            features = self.generate_features(market_data, financial_data)

            # Engineer additional features
            features = self.engineer_features(features)

            # Create target
            target = self.create_target(market_data)

            # Align features and target
            common_index = features.index.intersection(target.index)
            features = features.loc[common_index]
            target = target.loc[common_index]

            # Add symbol column
            features["symbol"] = symbol

            all_features.append(features)
            all_targets.append(target)

        if not all_features:
            print("No data available for any symbols!")
            return

        # Combine all data
        print("\nCombining data from all symbols...")
        X = pd.concat(all_features, axis=0)
        y = pd.concat(all_targets, axis=0)

        # Remove rows with missing target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"Final dataset shape: {X.shape}")
        print(f"Target distribution: {y.value_counts()}")

        # Train models
        model_results = self.train_models(X, y)

        if model_results:
            # Save results
            results_file = os.path.join(self.results_dir, f"{self.experiment_name}_results.json")
            with open(results_file, "w") as f:
                json.dump(model_results, f, indent=2, default=str)

            print(f"\nExperiment completed successfully!")
            print(f"Results saved to: {results_file}")
        else:
            print("\nExperiment failed - no model results")


def main():
    """Main function to run the experiment."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return

    experiment = MultiSourceExperiment(config_path)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
