"""
Signal Generator Interface

Abstract interface for signal generators in the BreadthFlow system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# Optional pandas import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
from datetime import datetime
from signal_config import SignalConfig

class SignalGeneratorInterface(ABC):
    """Abstract interface for signal generators"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this signal generator"""
        pass
    
    @abstractmethod
    def get_supported_strategies(self) -> List[str]:
        """Get list of supported strategy names"""
        pass
    
    @abstractmethod
    def generate_signals(self, config: SignalConfig, data: Dict[str, Any]):
        """Generate signals based on configuration and data"""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get signal generator configuration"""
        pass
    
    def validate_config(self, config: SignalConfig) -> bool:
        """Validate signal configuration"""
        if not config.validate():
            return False
        
        # Check if strategy is supported
        if config.strategy_name not in self.get_supported_strategies():
            return False
        
        return True
    
    def validate_data(self, data: Dict[str, Any], required_resources: List[str]) -> bool:
        """Validate that required data is available"""
        if not PANDAS_AVAILABLE:
            return False
            
        for resource in required_resources:
            if resource not in data:
                return False
            
            if data[resource].empty:
                return False
        
        return True
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this signal generator"""
        return {
            'total_signals_generated': 0,
            'average_generation_time': 0.0,
            'success_rate': 1.0,
            'last_generation': None
        }
    
    def get_signal_quality_metrics(self, signals) -> Dict[str, Any]:
        """Calculate signal quality metrics"""
        if not PANDAS_AVAILABLE:
            return {
                'total_signals': 0,
                'signal_strength_avg': 0.0,
                'confidence_avg': 0.0,
                'signal_distribution': {},
                'error': 'pandas not available'
            }
        
        if signals.empty:
            return {
                'total_signals': 0,
                'signal_strength_avg': 0.0,
                'confidence_avg': 0.0,
                'signal_distribution': {}
            }
        
        metrics = {
            'total_signals': len(signals),
            'signal_strength_avg': 0.0,
            'confidence_avg': 0.0,
            'signal_distribution': {}
        }
        
        # Calculate average signal strength
        if 'signal_strength' in signals.columns:
            metrics['signal_strength_avg'] = signals['signal_strength'].mean()
        
        # Calculate average confidence
        if 'confidence' in signals.columns:
            metrics['confidence_avg'] = signals['confidence'].mean()
        
        # Calculate signal distribution
        if 'signal_type' in signals.columns:
            metrics['signal_distribution'] = signals['signal_type'].value_counts().to_dict()
        
        return metrics
