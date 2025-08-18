# Symbol Lists for Breadth/Thrust Signals POC

## 📊 Overview

The Breadth/Thrust Signals system uses predefined symbol lists to calculate market breadth indicators. These lists are organized by market segments, sectors, and use cases to provide flexibility in analysis.

## 🎯 Available Symbol Lists

### **Demo Lists** (For Testing & Quick Analysis)

#### `demo_small` (8 symbols)
- **Purpose**: Quick testing and development
- **Symbols**: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, NFLX
- **Use Case**: Fast prototyping and initial testing

#### `demo_medium` (20 symbols)
- **Purpose**: Standard demonstrations and analysis
- **Symbols**: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, NFLX, ADBE, ORCL, INTC, AMD, QCOM, AVGO, MU, PANW, WDAY, ZS, CRWD, TEAM
- **Use Case**: Standard demos and moderate analysis

#### `demo_large` (40 symbols)
- **Purpose**: Comprehensive demonstrations
- **Symbols**: Includes tech leaders, financials, healthcare, and consumer stocks
- **Use Case**: Full system demonstrations and comprehensive analysis

### **Major Indices**

#### `sp500` (100 symbols)
- **Purpose**: S&P 500 constituents for broad market analysis
- **Description**: Major S&P 500 stocks for market breadth analysis
- **Use Case**: Production analysis of large-cap US market

#### `nasdaq100` (90 symbols)
- **Purpose**: NASDAQ 100 constituents for tech-focused analysis
- **Description**: Major NASDAQ stocks for technology-focused breadth analysis
- **Use Case**: Technology sector and growth stock analysis

#### `dow30` (30 symbols)
- **Purpose**: Dow Jones Industrial Average constituents
- **Description**: 30 major blue-chip stocks
- **Use Case**: Blue-chip and value stock analysis

### **Sector Lists**

#### `tech_leaders` (30 symbols)
- **Purpose**: Technology sector analysis
- **Description**: Major technology companies for tech sector breadth
- **Use Case**: Technology sector breadth analysis

#### `financials` (20 symbols)
- **Purpose**: Financial sector analysis
- **Description**: Major financial institutions
- **Use Case**: Financial sector breadth analysis

#### `healthcare` (20 symbols)
- **Purpose**: Healthcare sector analysis
- **Description**: Major healthcare and biotech companies
- **Use Case**: Healthcare sector breadth analysis

#### `sectors` (20 symbols)
- **Purpose**: Sector ETF analysis
- **Description**: Major sector ETFs for sector rotation analysis
- **Use Case**: Sector rotation and ETF analysis

## 🚀 How to Use Symbol Lists

### **CLI Commands**

#### List Available Symbol Lists
```bash
poetry run bf data symbols
```

#### Fetch Data Using Symbol Lists
```bash
# Quick demo
poetry run bf data fetch --symbol-list demo_small

# S&P 500 analysis
poetry run bf data fetch --symbol-list sp500

# Technology focus
poetry run bf data fetch --symbol-list tech_leaders

# Custom symbols (fallback)
poetry run bf data fetch --symbols AAPL,MSFT,GOOGL
```

#### Generate Signals Using Symbol Lists
```bash
# Generate signals for demo symbols
poetry run bf signals generate --symbol-list demo_medium

# Generate signals for S&P 500
poetry run bf signals generate --symbol-list sp500

# Generate signals for tech sector
poetry run bf signals generate --symbol-list tech_leaders
```

#### Run Backtests Using Symbol Lists
```bash
# Backtest with demo symbols
poetry run bf backtest run --symbol-list demo_small --from-date 2024-01-01 --to-date 2024-12-31

# Backtest with S&P 500
poetry run bf backtest run --symbol-list sp500 --from-date 2024-01-01 --to-date 2024-12-31

# Backtest with tech leaders
poetry run bf backtest run --symbol-list tech_leaders --from-date 2024-01-01 --to-date 2024-12-31
```

### **Programmatic Usage**

#### Python API
```python
from features.common.symbols import get_symbol_manager, get_demo_symbols, get_sp500_symbols

# Get symbol manager
manager = get_symbol_manager()

# Get specific symbol list
sp500_symbols = manager.get_symbol_list("sp500")
tech_symbols = manager.get_symbol_list("tech_leaders")

# Get demo symbols
demo_small = get_demo_symbols("small")
demo_medium = get_demo_symbols("medium")
demo_large = get_demo_symbols("large")

# Get S&P 500 symbols
sp500 = get_sp500_symbols()

# Get tech symbols
tech = get_tech_symbols()
```

#### Symbol List Information
```python
# Get list information
info = manager.get_symbol_list_info("sp500")
print(f"Name: {info['name']}")
print(f"Description: {info['description']}")
print(f"Symbol count: {info['symbol_count']}")

# Get all available lists
available_lists = manager.list_available_symbol_lists()
print(f"Available lists: {available_lists}")

# Get summary
summary = manager.get_symbol_list_summary()
print(f"Total lists: {summary['total_lists']}")
```

## 📈 Use Case Recommendations

### **Quick Testing & Development**
- **List**: `demo_small`
- **Symbols**: 8 major tech stocks
- **Use**: Fast prototyping, unit testing, development

### **Standard Demonstrations**
- **List**: `demo_medium`
- **Symbols**: 20 diverse stocks
- **Use**: System demos, moderate analysis, presentations

### **Comprehensive Analysis**
- **List**: `demo_large`
- **Symbols**: 40 stocks across sectors
- **Use**: Full system demonstrations, comprehensive testing

### **Production S&P 500 Analysis**
- **List**: `sp500`
- **Symbols**: 100 S&P 500 constituents
- **Use**: Production trading, large-cap market analysis

### **Technology Sector Focus**
- **List**: `tech_leaders`
- **Symbols**: 30 major tech companies
- **Use**: Technology sector analysis, growth stock focus

### **Sector Rotation Analysis**
- **List**: `sectors`
- **Symbols**: 20 sector ETFs
- **Use**: Sector rotation strategies, macro analysis

## 🔧 Customization

### **Creating Custom Symbol Lists**
```python
from features.common.symbols import get_symbol_manager

manager = get_symbol_manager()

# Create custom list
custom_list = manager.create_custom_list(
    name="my_custom_list",
    symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    description="My custom technology portfolio"
)

print(f"Created list with {len(custom_list['symbols'])} symbols")
```

### **Validating Symbols**
```python
# Validate symbol list
validation = manager.validate_symbols(["AAPL", "MSFT", "INVALID", "", "GOOGL"])

if validation["valid"]:
    print("All symbols are valid")
else:
    print(f"Invalid symbols: {validation['invalid_symbols']}")
    print(f"Valid symbols: {validation['valid_symbols']}")
```

### **Modifying Symbol Lists**
Edit the `data/symbols.json` file to:
- Add new symbol lists
- Modify existing lists
- Update recommendations
- Add new use cases

## 📊 Symbol List Structure

### **JSON Format**
```json
{
  "description": "Symbol lists for Breadth/Thrust Signals POC",
  "last_updated": "2024-12-19",
  "symbols": {
    "list_name": {
      "name": "Display Name",
      "description": "Description of the list",
      "symbols": ["SYMBOL1", "SYMBOL2", "SYMBOL3"]
    }
  },
  "recommendations": {
    "use_case": "list_name"
  }
}
```

### **Symbol Validation Rules**
- Symbols must be valid stock tickers
- Maximum length: 10 characters
- Allowed characters: letters, numbers, hyphens, dots
- Case-insensitive (automatically converted to uppercase)

## 🎯 Best Practices

### **Performance Considerations**
- **Small lists** (≤10 symbols): Fast processing, good for testing
- **Medium lists** (10-50 symbols): Balanced performance and coverage
- **Large lists** (≥50 symbols): Comprehensive analysis, slower processing

### **Data Availability**
- All symbols should have data available on Yahoo Finance
- Consider market hours and data freshness
- Handle missing data gracefully

### **Analysis Scope**
- **Sector analysis**: Use sector-specific lists
- **Market breadth**: Use broad market lists (sp500, nasdaq100)
- **Quick testing**: Use demo lists
- **Production**: Use comprehensive lists

## 🔍 Troubleshooting

### **Common Issues**

#### Symbol Not Found
```bash
# Check if symbol exists
poetry run bf data fetch --symbols INVALID_SYMBOL
# Error: Symbol not found

# Use valid symbols
poetry run bf data fetch --symbols AAPL,MSFT,GOOGL
```

#### Symbol List Not Found
```bash
# List available symbol lists
poetry run bf data symbols

# Use correct list name
poetry run bf data fetch --symbol-list demo_small
```

#### Performance Issues
```bash
# Use smaller lists for faster processing
poetry run bf data fetch --symbol-list demo_small

# Reduce parallel workers
poetry run bf data fetch --symbol-list sp500 --parallel 5
```

### **Validation Errors**
```python
# Check symbol validation
validation = manager.validate_symbols(["AAPL", "INVALID", ""])
print(f"Valid: {validation['valid']}")
print(f"Invalid symbols: {validation['invalid_symbols']}")
```

## 📚 Next Steps

1. **Start with demo lists** for initial testing
2. **Graduate to larger lists** for comprehensive analysis
3. **Create custom lists** for specific strategies
4. **Monitor performance** and adjust list sizes accordingly
5. **Update lists regularly** to reflect market changes

---

**Happy Trading! 📈**
