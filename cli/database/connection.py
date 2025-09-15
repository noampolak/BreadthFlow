"""
Database connection management for BreadthFlow dashboard
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


class DatabaseConnection:
    def __init__(self):
        self.engine = None
        self.connection = None

    def connect(self, connection_string=None):
        """Establish database connection"""
        if connection_string is None:
            # Default connection string from environment or config
            connection_string = os.getenv("DATABASE_URL", "postgresql://breadthflow:breadthflow@postgres:5432/breadthflow")

        try:
            self.engine = create_engine(connection_string)
            self.connection = self.engine.connect()
            return True
        except SQLAlchemyError as e:
            print(f"Database connection error: {e}")
            return False

    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()

    def execute_query(self, query, params=None):
        """Execute a SQL query"""
        try:
            if not self.connection:
                print("No database connection available")
                return None

            if params:
                result = self.connection.execute(text(query), params)
            else:
                result = self.connection.execute(text(query))
            return result
        except SQLAlchemyError as e:
            print(f"Query execution error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error executing query: {e}")
            return None

    def test_connection(self):
        """Test if database connection is working"""
        try:
            if not self.connection:
                return False
            result = self.execute_query("SELECT 1")
            return result is not None
        except Exception:
            return False


# Global database connection instance
_db_connection = None


def get_db_connection():
    """Get or create database connection"""
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
        if not _db_connection.connect():
            return None
    return _db_connection


def init_database():
    """Initialize database connection"""
    return get_db_connection()


def close_database():
    """Close database connection"""
    global _db_connection
    if _db_connection:
        _db_connection.disconnect()
        _db_connection = None
