"""
Financial Fundamentals Module

Generic, reusable financial fundamental analysis indicators that can be used
across multiple experiments and investment strategies.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional


class FinancialFundamentals:
    """Generic financial fundamentals calculator."""

    @staticmethod
    def calculate_valuation_ratios(
        market_cap: Union[pd.Series, np.ndarray],
        earnings: Union[pd.Series, np.ndarray],
        book_value: Union[pd.Series, np.ndarray],
        sales: Union[pd.Series, np.ndarray],
    ) -> Dict[str, pd.Series]:
        """Calculate key valuation ratios."""
        if isinstance(market_cap, np.ndarray):
            market_cap = pd.Series(market_cap)
        if isinstance(earnings, np.ndarray):
            earnings = pd.Series(earnings)
        if isinstance(book_value, np.ndarray):
            book_value = pd.Series(book_value)
        if isinstance(sales, np.ndarray):
            sales = pd.Series(sales)

        # P/E Ratio
        pe_ratio = market_cap / earnings
        pe_ratio = pe_ratio.replace([np.inf, -np.inf], np.nan)

        # P/B Ratio
        pb_ratio = market_cap / book_value
        pb_ratio = pb_ratio.replace([np.inf, -np.inf], np.nan)

        # P/S Ratio
        ps_ratio = market_cap / sales
        ps_ratio = ps_ratio.replace([np.inf, -np.inf], np.nan)

        return {"pe_ratio": pe_ratio, "pb_ratio": pb_ratio, "ps_ratio": ps_ratio}

    @staticmethod
    def calculate_profitability_ratios(
        revenue: Union[pd.Series, np.ndarray],
        net_income: Union[pd.Series, np.ndarray],
        total_assets: Union[pd.Series, np.ndarray],
        equity: Union[pd.Series, np.ndarray],
    ) -> Dict[str, pd.Series]:
        """Calculate profitability ratios."""
        if isinstance(revenue, np.ndarray):
            revenue = pd.Series(revenue)
        if isinstance(net_income, np.ndarray):
            net_income = pd.Series(net_income)
        if isinstance(total_assets, np.ndarray):
            total_assets = pd.Series(total_assets)
        if isinstance(equity, np.ndarray):
            equity = pd.Series(equity)

        # Net Profit Margin
        net_margin = net_income / revenue
        net_margin = net_margin.replace([np.inf, -np.inf], np.nan)

        # ROA (Return on Assets)
        roa = net_income / total_assets
        roa = roa.replace([np.inf, -np.inf], np.nan)

        # ROE (Return on Equity)
        roe = net_income / equity
        roe = roe.replace([np.inf, -np.inf], np.nan)

        return {"net_margin": net_margin, "roa": roa, "roe": roe}

    @staticmethod
    def calculate_leverage_ratios(
        total_debt: Union[pd.Series, np.ndarray],
        total_assets: Union[pd.Series, np.ndarray],
        equity: Union[pd.Series, np.ndarray],
    ) -> Dict[str, pd.Series]:
        """Calculate leverage and debt ratios."""
        if isinstance(total_debt, np.ndarray):
            total_debt = pd.Series(total_debt)
        if isinstance(total_assets, np.ndarray):
            total_assets = pd.Series(total_assets)
        if isinstance(equity, np.ndarray):
            equity = pd.Series(equity)

        # Debt-to-Assets Ratio
        debt_to_assets = total_debt / total_assets
        debt_to_assets = debt_to_assets.replace([np.inf, -np.inf], np.nan)

        # Debt-to-Equity Ratio
        debt_to_equity = total_debt / equity
        debt_to_equity = debt_to_equity.replace([np.inf, -np.inf], np.nan)

        # Equity Ratio
        equity_ratio = equity / total_assets
        equity_ratio = equity_ratio.replace([np.inf, -np.inf], np.nan)

        return {"debt_to_assets": debt_to_assets, "debt_to_equity": debt_to_equity, "equity_ratio": equity_ratio}

    @staticmethod
    def calculate_growth_metrics(
        current_values: Union[pd.Series, np.ndarray], previous_values: Union[pd.Series, np.ndarray], periods: int = 1
    ) -> pd.Series:
        """Calculate growth metrics (YoY, QoQ, etc.)."""
        if isinstance(current_values, np.ndarray):
            current_values = pd.Series(current_values)
        if isinstance(previous_values, np.ndarray):
            previous_values = pd.Series(previous_values)

        growth_rate = (current_values - previous_values) / previous_values
        growth_rate = growth_rate.replace([np.inf, -np.inf], np.nan)

        return growth_rate

    @staticmethod
    def calculate_efficiency_ratios(
        revenue: Union[pd.Series, np.ndarray],
        total_assets: Union[pd.Series, np.ndarray],
        inventory: Union[pd.Series, np.ndarray],
        accounts_receivable: Union[pd.Series, np.ndarray],
    ) -> Dict[str, pd.Series]:
        """Calculate efficiency ratios."""
        if isinstance(revenue, np.ndarray):
            revenue = pd.Series(revenue)
        if isinstance(total_assets, np.ndarray):
            total_assets = pd.Series(total_assets)
        if isinstance(inventory, np.ndarray):
            inventory = pd.Series(inventory)
        if isinstance(accounts_receivable, np.ndarray):
            accounts_receivable = pd.Series(accounts_receivable)

        # Asset Turnover
        asset_turnover = revenue / total_assets
        asset_turnover = asset_turnover.replace([np.inf, -np.inf], np.nan)

        # Inventory Turnover
        inventory_turnover = revenue / inventory
        inventory_turnover = inventory_turnover.replace([np.inf, -np.inf], np.nan)

        # Receivables Turnover
        receivables_turnover = revenue / accounts_receivable
        receivables_turnover = receivables_turnover.replace([np.inf, -np.inf], np.nan)

        return {
            "asset_turnover": asset_turnover,
            "inventory_turnover": inventory_turnover,
            "receivables_turnover": receivables_turnover,
        }

    @staticmethod
    def get_all_fundamentals(fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all financial fundamentals for a dataset."""
        results = pd.DataFrame(index=fundamental_data.index)

        # Valuation ratios
        if all(col in fundamental_data.columns for col in ["market_cap", "earnings", "book_value", "sales"]):
            valuation = FinancialFundamentals.calculate_valuation_ratios(
                fundamental_data["market_cap"],
                fundamental_data["earnings"],
                fundamental_data["book_value"],
                fundamental_data["sales"],
            )
            results["pe_ratio"] = valuation["pe_ratio"]
            results["pb_ratio"] = valuation["pb_ratio"]
            results["ps_ratio"] = valuation["ps_ratio"]

        # Profitability ratios
        if all(col in fundamental_data.columns for col in ["revenue", "net_income", "total_assets", "equity"]):
            profitability = FinancialFundamentals.calculate_profitability_ratios(
                fundamental_data["revenue"],
                fundamental_data["net_income"],
                fundamental_data["total_assets"],
                fundamental_data["equity"],
            )
            results["net_margin"] = profitability["net_margin"]
            results["roa"] = profitability["roa"]
            results["roe"] = profitability["roe"]

        # Leverage ratios
        if all(col in fundamental_data.columns for col in ["total_debt", "total_assets", "equity"]):
            leverage = FinancialFundamentals.calculate_leverage_ratios(
                fundamental_data["total_debt"], fundamental_data["total_assets"], fundamental_data["equity"]
            )
            results["debt_to_assets"] = leverage["debt_to_assets"]
            results["debt_to_equity"] = leverage["debt_to_equity"]
            results["equity_ratio"] = leverage["equity_ratio"]

        # Growth metrics
        if "revenue" in fundamental_data.columns:
            revenue_growth = FinancialFundamentals.calculate_growth_metrics(
                fundamental_data["revenue"], fundamental_data["revenue"].shift(4)  # YoY growth
            )
            results["revenue_growth"] = revenue_growth

        if "earnings" in fundamental_data.columns:
            earnings_growth = FinancialFundamentals.calculate_growth_metrics(
                fundamental_data["earnings"], fundamental_data["earnings"].shift(4)  # YoY growth
            )
            results["earnings_growth"] = earnings_growth

        return results
