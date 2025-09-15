from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class TradingSignal(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    signal_type: str = Field(..., description="Signal type (BUY, SELL, HOLD)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    strength: str = Field(..., description="Signal strength (STRONG, MEDIUM, WEAK)")
    date: str = Field(..., description="Signal date")
    timeframe: str = Field(..., description="Timeframe (1min, 5min, 15min, 1hour, 1day)")
    create_time: str = Field(..., description="Signal creation timestamp")

    class Config:
        from_attributes = True


class SignalStats(BaseModel):
    total_signals: int = Field(..., description="Total number of signals")
    buy_signals: int = Field(..., description="Number of buy signals")
    sell_signals: int = Field(..., description="Number of sell signals")
    hold_signals: int = Field(..., description="Number of hold signals")
    avg_confidence: float = Field(..., description="Average confidence score")
    strong_signals: int = Field(..., description="Number of strong signals")


class SignalExportResponse(BaseModel):
    data: str = Field(..., description="Exported data")
    format: str = Field(..., description="Export format (csv or json)")
    filename: str = Field(..., description="Suggested filename")
    content_type: str = Field(..., description="MIME content type")
