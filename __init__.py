"""
Breadth/Thrust Signals POC

A real-time quantitative trading signal system using market breadth indicators.
Built with PySpark, Kafka, Delta Lake, and modern big data technologies.
"""

__version__ = "1.0.0"
__author__ = "Noam (Engineering), Yossi (Research)"
__description__ = "Real-time market breadth analysis and trading signals"

# Import main components
from . import backtests, cli, features, ingestion, model, streaming
