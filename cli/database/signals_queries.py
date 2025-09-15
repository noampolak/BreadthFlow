"""
Signals-related database queries for BreadthFlow dashboard
"""

from .connection import get_db_connection


class SignalsQueries:
    def __init__(self):
        self.db = get_db_connection()

    def get_signals_data(self, run_id):
        """Get trading signals data for a specific run"""
        try:
            if not self.db or not self.db.connection:
                print("No database connection available")
                return {"total_signals": 0, "buy_signals": 0, "sell_signals": 0, "avg_confidence": 0}

            signals_query = """
                SELECT 
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN signal_type = 'BUY' THEN 1 END) as buy_signals,
                    COUNT(CASE WHEN signal_type = 'SELL' THEN 1 END) as sell_signals,
                    ROUND(AVG(confidence), 2) as avg_confidence
                FROM signals_metadata 
                WHERE run_id = :run_id
            """
            signals_result = self.db.execute_query(signals_query, {"run_id": run_id})

            if signals_result:
                row = signals_result.fetchone()
                return {
                    "total_signals": row[0],
                    "buy_signals": row[1],
                    "sell_signals": row[2],
                    "avg_confidence": row[3] if row[3] else 0,
                }
            else:
                return {"total_signals": 0, "buy_signals": 0, "sell_signals": 0, "avg_confidence": 0}

        except Exception as e:
            print(f"Error getting signals data: {e}")
            return {"total_signals": 0, "buy_signals": 0, "sell_signals": 0, "avg_confidence": 0}

    def get_backtest_data(self, run_id):
        """Get backtest results for a specific run"""
        try:
            if not self.db or not self.db.connection:
                print("No database connection available")
                return {"total_return": 0, "sharpe_ratio": 0, "max_drawdown": 0, "total_trades": 0}

            backtest_query = """
                SELECT 
                    total_return, sharpe_ratio, max_drawdown, total_trades
                FROM backtest_results 
                WHERE run_id = :run_id
            """
            backtest_result = self.db.execute_query(backtest_query, {"run_id": run_id})

            if backtest_result:
                row = backtest_result.fetchone()
                return {
                    "total_return": float(row[0]) if row[0] else 0,
                    "sharpe_ratio": float(row[1]) if row[1] else 0,
                    "max_drawdown": float(row[2]) if row[2] else 0,
                    "total_trades": int(row[3]) if row[3] else 0,
                }
            else:
                return {"total_return": 0, "sharpe_ratio": 0, "max_drawdown": 0, "total_trades": 0}

        except Exception as e:
            print(f"Error getting backtest data: {e}")
            return {"total_return": 0, "sharpe_ratio": 0, "max_drawdown": 0, "total_trades": 0}

    def export_signals_by_run(self, run_id):
        """Export signals data for a specific run"""
        try:
            if not self.db or not self.db.connection:
                print("No database connection available")
                return []

            export_query = """
                SELECT 
                    symbol, signal_type, confidence, timestamp, price, volume
                FROM signals_metadata 
                WHERE run_id = :run_id
                ORDER BY timestamp
            """
            export_result = self.db.execute_query(export_query, {"run_id": run_id})

            if export_result:
                signals = []
                for row in export_result:
                    signals.append(
                        {
                            "symbol": row[0],
                            "signal_type": row[1],
                            "confidence": float(row[2]) if row[2] else 0,
                            "timestamp": row[3].isoformat() if row[3] else None,
                            "price": float(row[4]) if row[4] else 0,
                            "volume": int(row[5]) if row[5] else 0,
                        }
                    )
                return signals
            else:
                return []

        except Exception as e:
            print(f"Error exporting signals: {e}")
            return []

    def get_latest_signals(self):
        """Get latest trading signals from all timeframes"""
        try:
            if not self.db or not self.db.connection:
                print("No database connection available")
                return []

            signals_query = """
                SELECT 
                    symbol, signal_type, confidence, timestamp, price, volume,
                    timeframe, run_id
                FROM signals_metadata 
                ORDER BY timestamp DESC 
                LIMIT 50
            """
            signals_result = self.db.execute_query(signals_query)

            signals = []
            if signals_result:
                for row in signals_result:
                    signals.append(
                        {
                            "symbol": row[0],
                            "signal_type": row[1],
                            "confidence": float(row[2]) if row[2] else 0,
                            "timestamp": row[3].isoformat() if row[3] else None,
                            "price": float(row[4]) if row[4] else 0,
                            "volume": int(row[5]) if row[5] else 0,
                            "timeframe": row[6],
                            "run_id": row[7],
                        }
                    )

            return signals

        except Exception as e:
            print(f"Error getting latest signals: {e}")
            return []

    def export_signals(self, format_type, run_id=None):
        """Export signals data in the specified format"""
        try:
            if not self.db or not self.db.connection:
                print("No database connection available")
                return "No database connection available", "text/plain", 503

            if run_id:
                # Export signals for specific run
                export_query = """
                    SELECT 
                        symbol, signal_type, confidence, timestamp, price, volume, timeframe
                    FROM signals_metadata 
                    WHERE run_id = :run_id
                    ORDER BY timestamp
                """
                export_result = self.db.execute_query(export_query, {"run_id": run_id})
            else:
                # Export all recent signals
                export_query = """
                    SELECT 
                        symbol, signal_type, confidence, timestamp, price, volume, timeframe
                    FROM signals_metadata 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                """
                export_result = self.db.execute_query(export_query)

            if not export_result:
                return "No signals found", "text/plain", 404

            signals = []
            for row in export_result:
                signals.append(
                    {
                        "symbol": row[0],
                        "signal_type": row[1],
                        "confidence": float(row[2]) if row[2] else 0,
                        "timestamp": row[3].isoformat() if row[3] else None,
                        "price": float(row[4]) if row[4] else 0,
                        "volume": int(row[5]) if row[5] else 0,
                        "timeframe": row[6],
                    }
                )

            if format_type == "csv":
                # Generate CSV format
                csv_content = "Symbol,Signal Type,Confidence,Timestamp,Price,Volume,Timeframe\n"
                for signal in signals:
                    csv_content += f"{signal['symbol']},{signal['signal_type']},{signal['confidence']},{signal['timestamp']},{signal['price']},{signal['volume']},{signal['timeframe']}\n"
                return csv_content, "text/csv", 200
            else:
                # Generate JSON format
                import json

                return json.dumps(signals, indent=2), "application/json", 200

        except Exception as e:
            print(f"Error exporting signals: {e}")
            return f"Error exporting signals: {str(e)}", "text/plain", 500
