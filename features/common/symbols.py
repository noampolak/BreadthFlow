"""
Symbol Management for Breadth/Thrust Signals POC

Provides utilities for loading and managing symbol lists for market breadth analysis.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SymbolManager:
    """
    Manages symbol lists for market breadth analysis.
    
    Provides access to predefined symbol sets and utilities for
    loading, validating, and using symbol lists.
    """
    
    def __init__(self, symbols_file: str = "data/symbols.json"):
        """
        Initialize SymbolManager.
        
        Args:
            symbols_file: Path to the symbols JSON file
        """
        self.symbols_file = Path(symbols_file)
        self.symbols_data = self._load_symbols_file()
    
    def _load_symbols_file(self) -> Dict[str, Any]:
        """
        Load symbols data from JSON file.
        
        Returns:
            Dictionary containing symbol lists and metadata
        """
        try:
            if not self.symbols_file.exists():
                logger.warning(f"Symbols file not found: {self.symbols_file}")
                return self._get_default_symbols()
            
            with open(self.symbols_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data.get('symbols', {}))} symbol lists")
            return data
            
        except Exception as e:
            logger.error(f"Error loading symbols file: {e}")
            return self._get_default_symbols()
    
    def _get_default_symbols(self) -> Dict[str, Any]:
        """
        Get default symbols if file loading fails.
        
        Returns:
            Default symbol data structure
        """
        return {
            "description": "Default symbols for Breadth/Thrust Signals POC",
            "symbols": {
                "demo_small": {
                    "name": "Demo Small Set",
                    "description": "Small set for quick demos and testing",
                    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
                },
                "demo_medium": {
                    "name": "Demo Medium Set", 
                    "description": "Medium set for standard demos",
                    "symbols": [
                        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
                        "ADBE", "ORCL", "INTC", "AMD", "QCOM", "AVGO", "MU", "PANW"
                    ]
                }
            },
            "recommendations": {
                "quick_testing": "demo_small",
                "standard_demo": "demo_medium"
            }
        }
    
    def get_symbol_list(self, list_name: str) -> List[str]:
        """
        Get a specific symbol list by name.
        
        Args:
            list_name: Name of the symbol list to retrieve
            
        Returns:
            List of symbols
            
        Raises:
            ValueError: If symbol list not found
        """
        symbols = self.symbols_data.get("symbols", {})
        
        if list_name not in symbols:
            available_lists = list(symbols.keys())
            raise ValueError(
                f"Symbol list '{list_name}' not found. "
                f"Available lists: {available_lists}"
            )
        
        return symbols[list_name]["symbols"]
    
    def get_symbol_list_info(self, list_name: str) -> Dict[str, Any]:
        """
        Get information about a specific symbol list.
        
        Args:
            list_name: Name of the symbol list
            
        Returns:
            Dictionary with list information
        """
        symbols = self.symbols_data.get("symbols", {})
        
        if list_name not in symbols:
            raise ValueError(f"Symbol list '{list_name}' not found")
        
        return symbols[list_name]
    
    def list_available_symbol_lists(self) -> List[str]:
        """
        Get list of available symbol list names.
        
        Returns:
            List of available symbol list names
        """
        return list(self.symbols_data.get("symbols", {}).keys())
    
    def get_recommended_list(self, use_case: str) -> str:
        """
        Get recommended symbol list for a specific use case.
        
        Args:
            use_case: Use case (e.g., 'quick_testing', 'standard_demo')
            
        Returns:
            Recommended symbol list name
        """
        recommendations = self.symbols_data.get("recommendations", {})
        
        if use_case not in recommendations:
            available_cases = list(recommendations.keys())
            raise ValueError(
                f"Use case '{use_case}' not found. "
                f"Available cases: {available_cases}"
            )
        
        return recommendations[use_case]
    
    def get_symbols_for_use_case(self, use_case: str) -> List[str]:
        """
        Get symbols for a specific use case.
        
        Args:
            use_case: Use case (e.g., 'quick_testing', 'standard_demo')
            
        Returns:
            List of symbols for the use case
        """
        list_name = self.get_recommended_list(use_case)
        return self.get_symbol_list(list_name)
    
    def validate_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Validate a list of symbols.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Dictionary with validation results
        """
        if not symbols:
            return {
                "valid": False,
                "error": "Empty symbol list",
                "valid_symbols": [],
                "invalid_symbols": []
            }
        
        # Basic validation - check for common issues
        valid_symbols = []
        invalid_symbols = []
        
        for symbol in symbols:
            if not symbol or not isinstance(symbol, str):
                invalid_symbols.append(symbol)
                continue
            
            # Remove whitespace and convert to uppercase
            clean_symbol = symbol.strip().upper()
            
            if not clean_symbol:
                invalid_symbols.append(symbol)
                continue
            
            # Basic format validation
            if len(clean_symbol) > 10 or not clean_symbol.replace('-', '').replace('.', '').isalnum():
                invalid_symbols.append(symbol)
                continue
            
            valid_symbols.append(clean_symbol)
        
        return {
            "valid": len(invalid_symbols) == 0,
            "valid_symbols": valid_symbols,
            "invalid_symbols": invalid_symbols,
            "total_symbols": len(symbols),
            "valid_count": len(valid_symbols),
            "invalid_count": len(invalid_symbols)
        }
    
    def get_symbol_list_summary(self) -> Dict[str, Any]:
        """
        Get summary of all available symbol lists.
        
        Returns:
            Dictionary with summary information
        """
        symbols = self.symbols_data.get("symbols", {})
        recommendations = self.symbols_data.get("recommendations", {})
        
        summary = {
            "total_lists": len(symbols),
            "lists": {},
            "recommendations": recommendations,
            "metadata": {
                "description": self.symbols_data.get("description", ""),
                "last_updated": self.symbols_data.get("last_updated", "")
            }
        }
        
        for list_name, list_data in symbols.items():
            summary["lists"][list_name] = {
                "name": list_data.get("name", list_name),
                "description": list_data.get("description", ""),
                "symbol_count": len(list_data.get("symbols", [])),
                "symbols": list_data.get("symbols", [])
            }
        
        return summary
    
    def create_custom_list(self, name: str, symbols: List[str], description: str = "") -> Dict[str, Any]:
        """
        Create a custom symbol list.
        
        Args:
            name: Name for the custom list
            symbols: List of symbols
            description: Description of the list
            
        Returns:
            Dictionary with the custom list data
        """
        validation = self.validate_symbols(symbols)
        
        if not validation["valid"]:
            raise ValueError(f"Invalid symbols: {validation['invalid_symbols']}")
        
        custom_list = {
            "name": name,
            "description": description,
            "symbols": validation["valid_symbols"]
        }
        
        logger.info(f"Created custom symbol list '{name}' with {len(validation['valid_symbols'])} symbols")
        return custom_list


def get_symbol_manager() -> SymbolManager:
    """
    Factory function to create SymbolManager instance.
    
    Returns:
        Configured SymbolManager instance
    """
    return SymbolManager()


def get_demo_symbols(size: str = "medium") -> List[str]:
    """
    Get demo symbols based on size.
    
    Args:
        size: Size of demo set ('small', 'medium', 'large')
        
    Returns:
        List of demo symbols
    """
    manager = get_symbol_manager()
    
    size_mapping = {
        "small": "demo_small",
        "medium": "demo_medium", 
        "large": "demo_large"
    }
    
    if size not in size_mapping:
        raise ValueError(f"Invalid size '{size}'. Use: {list(size_mapping.keys())}")
    
    return manager.get_symbol_list(size_mapping[size])


def get_sp500_symbols() -> List[str]:
    """
    Get S&P 500 symbols.
    
    Returns:
        List of S&P 500 symbols
    """
    manager = get_symbol_manager()
    return manager.get_symbol_list("sp500")


def get_tech_symbols() -> List[str]:
    """
    Get technology sector symbols.
    
    Returns:
        List of technology symbols
    """
    manager = get_symbol_manager()
    return manager.get_symbol_list("tech_leaders")


# Example usage and testing
if __name__ == "__main__":
    # Create symbol manager
    manager = get_symbol_manager()
    
    # List available symbol lists
    print("Available symbol lists:")
    for list_name in manager.list_available_symbol_lists():
        info = manager.get_symbol_list_info(list_name)
        print(f"  - {list_name}: {info['name']} ({len(info['symbols'])} symbols)")
    
    # Get demo symbols
    demo_symbols = get_demo_symbols("medium")
    print(f"\nDemo symbols (medium): {demo_symbols}")
    
    # Get S&P 500 symbols
    sp500_symbols = get_sp500_symbols()
    print(f"\nS&P 500 symbols (first 10): {sp500_symbols[:10]}")
    
    # Validate symbols
    test_symbols = ["AAPL", "MSFT", "INVALID", "", "GOOGL"]
    validation = manager.validate_symbols(test_symbols)
    print(f"\nValidation results: {validation}")
    
    # Get summary
    summary = manager.get_symbol_list_summary()
    print(f"\nTotal symbol lists: {summary['total_lists']}")
