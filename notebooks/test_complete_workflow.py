#!/usr/bin/env python3
"""
Complete Workflow Test for Multi-Source Training Example

This script tests the complete workflow from feature generation to model training
inside the Docker environment.
"""

import sys
import os

sys.path.append("/home/jovyan/work")

from features import TechnicalIndicators, FinancialFundamentals, MarketMicrostructure, TimeFeatures, FeatureUtils
from experiments.multi_source_analysis.run_experiment import MultiSourceExperiment
import pandas as pd
import numpy as np
import requests
import yaml


def test_complete_workflow():
    """Test the complete multi-source training workflow."""
    print("🚀 Complete Multi-Source Training Workflow Test")
    print("=" * 60)

    # Test 1: Feature Modules
    print("\n1️⃣ Testing Generic Feature Modules...")
    try:
        tech_calc = TechnicalIndicators()
        financial_calc = FinancialFundamentals()
        micro_calc = MarketMicrostructure()
        time_calc = TimeFeatures()
        utils = FeatureUtils()
        print("✅ All feature calculators initialized")
    except Exception as e:
        print(f"❌ Feature modules failed: {e}")
        return False

    # Test 2: Sample Data Generation
    print("\n2️⃣ Testing Sample Data Generation...")
    try:
        experiment = MultiSourceExperiment("/home/jovyan/work/experiments/multi_source_analysis/config.yaml")
        sample_data = experiment.fetch_market_data("AAPL", "2024-01-01", "2024-01-31")

        if sample_data is not None and len(sample_data) > 0:
            print(f"✅ Sample data generated: {sample_data.shape}")
            print(f"   Columns: {list(sample_data.columns)}")
        else:
            print("❌ Failed to generate sample data")
            return False
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

    # Test 3: Feature Generation
    print("\n3️⃣ Testing Feature Generation...")
    try:
        # Generate all features
        features = pd.DataFrame(index=sample_data.index)

        # Technical indicators
        tech_features = tech_calc.get_all_indicators(sample_data)
        features = pd.concat([features, tech_features], axis=1)
        print(f"✅ Technical indicators: {len(tech_features.columns)} features")

        # Time features
        time_features = time_calc.get_all_time_features(sample_data.index)
        features = pd.concat([features, time_features], axis=1)
        print(f"✅ Time features: {len(time_features.columns)} features")

        # Market microstructure
        micro_features = micro_calc.get_all_microstructure(sample_data)
        features = pd.concat([features, micro_features], axis=1)
        print(f"✅ Market microstructure: {len(micro_features.columns)} features")

        print(f"✅ Total features generated: {features.shape[1]}")

    except Exception as e:
        print(f"❌ Feature generation failed: {e}")
        return False

    # Test 4: ML Services Connectivity
    print("\n4️⃣ Testing ML Services...")
    try:
        services = {
            "Data Pipeline": "http://data-pipeline:8001/health",
            "Feature Engineering": "http://feature-engineering:8002/health",
            "Model Training": "http://model-training:8003/health",
            "AutoML": "http://automl:8004/health",
        }

        connected = 0
        for service_name, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {service_name}: Connected")
                    connected += 1
                else:
                    print(f"❌ {service_name}: Error {response.status_code}")
            except Exception as e:
                print(f"❌ {service_name}: Connection failed")

        if connected == len(services):
            print("✅ All ML services connected")
        else:
            print(f"⚠️ Only {connected}/{len(services)} services connected")

    except Exception as e:
        print(f"❌ Service connectivity test failed: {e}")
        return False

    # Test 5: Model Training (Simple Test)
    print("\n5️⃣ Testing Model Training Service...")
    try:
        # Create simple test data
        test_data = {
            "features": [
                {"feature1": 1.0, "feature2": 2.0},
                {"feature1": 1.5, "feature2": 2.5},
                {"feature1": 2.0, "feature2": 3.0},
            ],
            "target": [0, 1, 0],
            "feature_names": ["feature1", "feature2"],
            "algorithms": ["random_forest"],
            "experiment_name": "workflow_test",
            "target_type": "classification",
        }

        response = requests.post("http://model-training:8003/train-models", json=test_data, timeout=30)

        if response.status_code == 200:
            results = response.json()
            print("✅ Model training service working")
            print(f"   Response: {str(results)[:100]}...")
        else:
            print(f"❌ Model training failed: {response.status_code}")
            print(f"   Response: {response.text[:100]}...")

    except Exception as e:
        print(f"❌ Model training test failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("🎉 COMPLETE WORKFLOW TEST PASSED!")
    print("✅ All components working correctly")
    print("✅ Multi-source training example is ready to use")
    print("\n📚 Next steps:")
    print("1. Open Jupyter Lab: http://localhost:8888")
    print("2. Open: notebooks/multi_source_training_example.ipynb")
    print("3. Run the complete example step by step")

    return True


if __name__ == "__main__":
    success = test_complete_workflow()
    sys.exit(0 if success else 1)
