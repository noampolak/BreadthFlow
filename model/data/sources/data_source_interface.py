"""
Data Source Interface

Abstract interface for data sources in the BreadthFlow system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Optional pandas import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

class DataSourceInterface(ABC):
    """Abstract interface for data sources"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this data source"""
        pass
    
    @abstractmethod
    def get_supported_resources(self) -> List[str]:
        """Get list of supported resource names"""
        pass
    
    @abstractmethod
    def fetch_data(self, resource_name: str, symbols: List[str], 
                   start_date: datetime, end_date: datetime, 
                   **kwargs):
        """Fetch data for specified resource and symbols"""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get data source configuration"""
        pass
    
    def validate_resource_support(self, resource_name: str) -> bool:
        """Check if this data source supports the specified resource"""
        return resource_name in self.get_supported_resources()
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limiting information for this data source"""
        return {
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'requests_per_day': 10000
        }
    
    def get_data_quality_info(self) -> Dict[str, Any]:
        """Get data quality information for this data source"""
        return {
            'completeness': 0.95,
            'accuracy': 0.98,
            'latency': 'low',
            'update_frequency': 'real_time'
        }
