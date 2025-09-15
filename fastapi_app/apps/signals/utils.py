from sqlalchemy.orm import Session
from typing import List, Tuple, Dict, Any
import json
import csv
import io
from datetime import datetime, timedelta
import os
import glob

from apps.signals.schemas import TradingSignal, SignalStats, SignalExportResponse


class SignalService:
    def __init__(self, db: Session):
        self.db = db

    def get_latest_signals(self) -> Tuple[List[TradingSignal], SignalStats]:
        """Get latest trading signals from the last 4 days"""
        try:
            # Look for signal files in MinIO storage (simulated)
            signals = self._read_signal_files()

            # Calculate statistics
            stats = self._calculate_signal_stats(signals)

            return signals, stats
        except Exception as e:
            print(f"Error fetching signals: {e}")
            # Return empty data if no signals found
            return [], SignalStats(
                total_signals=0, buy_signals=0, sell_signals=0, hold_signals=0, avg_confidence=0.0, strong_signals=0
            )

    def _read_signal_files(self) -> List[TradingSignal]:
        """Read signal files from storage"""
        signals = []

        try:
            # Look for signal files in the expected storage location
            # This would typically be MinIO storage or a file system
            signal_files = glob.glob("/app/data/signals/*.json")

            for file_path in signal_files:
                try:
                    with open(file_path, "r") as f:
                        signal_data = json.load(f)
                        if isinstance(signal_data, list):
                            for item in signal_data:
                                signals.append(TradingSignal(**item))
                        else:
                            signals.append(TradingSignal(**signal_data))
                except Exception as e:
                    print(f"Error reading signal file {file_path}: {e}")
                    continue

        except Exception as e:
            print(f"Error reading signal files: {e}")
            # Return empty list if no signals found
            pass

        return signals

    def _calculate_signal_stats(self, signals: List[TradingSignal]) -> SignalStats:
        """Calculate signal statistics"""
        if not signals:
            return SignalStats(
                total_signals=0, buy_signals=0, sell_signals=0, hold_signals=0, avg_confidence=0.0, strong_signals=0
            )

        total_signals = len(signals)
        buy_signals = sum(1 for s in signals if s.signal_type.upper() == "BUY")
        sell_signals = sum(1 for s in signals if s.signal_type.upper() == "SELL")
        hold_signals = sum(1 for s in signals if s.signal_type.upper() == "HOLD")
        avg_confidence = sum(s.confidence for s in signals) / total_signals * 100
        strong_signals = sum(1 for s in signals if s.strength.upper() == "STRONG")

        return SignalStats(
            total_signals=total_signals,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            hold_signals=hold_signals,
            avg_confidence=avg_confidence,
            strong_signals=strong_signals,
        )

    def export_signals(self, format: str) -> SignalExportResponse:
        """Export signals in the specified format"""
        signals, _ = self.get_latest_signals()

        if format.lower() == "csv":
            return self._export_csv(signals)
        elif format.lower() == "json":
            return self._export_json(signals)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_csv(self, signals: List[TradingSignal]) -> SignalExportResponse:
        """Export signals as CSV"""
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(["Symbol", "Signal Type", "Confidence", "Strength", "Date", "Timeframe", "Create Time"])

        # Write data
        for signal in signals:
            writer.writerow(
                [
                    signal.symbol,
                    signal.signal_type,
                    f"{signal.confidence:.3f}",
                    signal.strength,
                    signal.date,
                    signal.timeframe,
                    signal.create_time,
                ]
            )

        csv_data = output.getvalue()
        output.close()

        return SignalExportResponse(
            data=csv_data,
            format="csv",
            filename=f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            content_type="text/csv",
        )

    def _export_json(self, signals: List[TradingSignal]) -> SignalExportResponse:
        """Export signals as JSON"""
        json_data = json.dumps([signal.dict() for signal in signals], indent=2)

        return SignalExportResponse(
            data=json_data,
            format="json",
            filename=f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            content_type="application/json",
        )
