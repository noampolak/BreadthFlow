#!/usr/bin/env python3
"""
Timeframe Database Initialization Script

This script initializes the database with timeframe-enhanced schema
and configuration data for multi-timeframe support.
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
import logging

# Add the jobs directory to Python path
sys.path.insert(0, '/opt/bitnami/spark/jobs')

from model.timeframe_config import get_timeframe_config_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeframeDatabaseInitializer:
    """Initialize database with timeframe-enhanced schema and data."""
    
    def __init__(self, database_url: str = None):
        """
        Initialize the database initializer.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url or os.getenv(
            'DATABASE_URL', 
            'postgresql://pipeline:pipeline123@postgres:5432/breadthflow'
        )
        self.schema_file = '/opt/bitnami/spark/jobs/sql/timeframe_schema.sql'
        
    def connect(self):
        """Create database connection."""
        try:
            conn = psycopg2.connect(self.database_url)
            conn.set_session(autocommit=False)
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def execute_schema_file(self, conn):
        """Execute the timeframe schema SQL file."""
        try:
            # Check if schema file exists
            if not os.path.exists(self.schema_file):
                # If running in different location, try relative path
                self.schema_file = 'sql/timeframe_schema.sql'
                if not os.path.exists(self.schema_file):
                    logger.error(f"Schema file not found: {self.schema_file}")
                    return False
            
            logger.info(f"Executing schema file: {self.schema_file}")
            
            with open(self.schema_file, 'r') as f:
                sql_content = f.read()
            
            # Split by semicolons and execute each statement
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            
            cursor = conn.cursor()
            for i, statement in enumerate(statements):
                try:
                    logger.debug(f"Executing statement {i+1}/{len(statements)}")
                    cursor.execute(statement)
                except Exception as e:
                    logger.warning(f"Statement {i+1} failed (might be expected): {e}")
                    logger.debug(f"Failed statement: {statement[:100]}...")
                    conn.rollback()
                    cursor = conn.cursor()  # Get a new cursor after rollback
                    continue
            
            conn.commit()
            cursor.close()
            logger.info("Schema file executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute schema file: {e}")
            conn.rollback()
            return False
    
    def populate_timeframe_configs(self, conn):
        """Populate timeframe_configs table with default configurations."""
        try:
            logger.info("Populating timeframe configurations...")
            
            # Get configurations from config manager
            config_manager = get_timeframe_config_manager()
            configs = config_manager.get_all_configs()
            
            cursor = conn.cursor()
            
            for timeframe, config in configs.items():
                # Convert config to JSONB
                config_json = json.dumps(config.to_dict())
                
                # Insert or update configuration
                cursor.execute("""
                    INSERT INTO timeframe_configs 
                    (timeframe, display_name, description, config_json, is_active)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (timeframe) 
                    DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        description = EXCLUDED.description,
                        config_json = EXCLUDED.config_json,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    timeframe,
                    config.display_name,
                    config.description,
                    config_json,
                    True
                ))
            
            conn.commit()
            cursor.close()
            
            logger.info(f"Populated {len(configs)} timeframe configurations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to populate timeframe configs: {e}")
            conn.rollback()
            return False
    
    def verify_installation(self, conn):
        """Verify that the timeframe schema was installed correctly."""
        try:
            logger.info("Verifying timeframe schema installation...")
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Check tables exist
            tables_to_check = [
                'timeframe_configs',
                'timeframe_data_summary', 
                'signals_metadata',
                'backtest_results'
            ]
            
            for table in tables_to_check:
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM information_schema.tables 
                    WHERE table_name = %s
                """, (table,))
                
                result = cursor.fetchone()
                if result['count'] == 0:
                    logger.error(f"Table {table} not found")
                    return False
                else:
                    logger.info(f"✓ Table {table} exists")
            
            # Check views exist
            views_to_check = [
                'timeframe_performance_stats',
                'signals_summary_by_timeframe',
                'data_availability_by_timeframe'
            ]
            
            for view in views_to_check:
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM information_schema.views 
                    WHERE table_name = %s
                """, (view,))
                
                result = cursor.fetchone()
                if result['count'] == 0:
                    logger.error(f"View {view} not found")
                    return False
                else:
                    logger.info(f"✓ View {view} exists")
            
            # Check functions exist
            functions_to_check = [
                'update_timeframe_data_summary',
                'log_signal_generation',
                'log_backtest_results'
            ]
            
            for function in functions_to_check:
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM information_schema.routines 
                    WHERE routine_name = %s AND routine_type = 'FUNCTION'
                """, (function,))
                
                result = cursor.fetchone()
                if result['count'] == 0:
                    logger.error(f"Function {function} not found")
                    return False
                else:
                    logger.info(f"✓ Function {function} exists")
            
            # Check timeframe configurations were loaded
            cursor.execute("SELECT COUNT(*) as count FROM timeframe_configs")
            result = cursor.fetchone()
            
            if result['count'] == 0:
                logger.error("No timeframe configurations found")
                return False
            else:
                logger.info(f"✓ {result['count']} timeframe configurations loaded")
            
            # Check if pipeline_runs table has timeframe column
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM information_schema.columns 
                WHERE table_name = 'pipeline_runs' AND column_name = 'timeframe'
            """)
            
            result = cursor.fetchone()
            if result['count'] == 0:
                logger.error("pipeline_runs table missing timeframe column")
                return False
            else:
                logger.info("✓ pipeline_runs table enhanced with timeframe support")
            
            cursor.close()
            logger.info("✅ Timeframe schema verification completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify installation: {e}")
            return False
    
    def get_installation_status(self):
        """Get the status of timeframe schema installation."""
        try:
            conn = self.connect()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Check if timeframe_configs table exists
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM information_schema.tables 
                WHERE table_name = 'timeframe_configs'
            """)
            
            result = cursor.fetchone()
            table_exists = result['count'] > 0
            
            config_count = 0
            if table_exists:
                cursor.execute("SELECT COUNT(*) as count FROM timeframe_configs")
                result = cursor.fetchone()
                config_count = result['count']
            
            cursor.close()
            conn.close()
            
            return {
                'installed': table_exists,
                'config_count': config_count,
                'database_url': self.database_url
            }
            
        except Exception as e:
            logger.error(f"Failed to get installation status: {e}")
            return {
                'installed': False,
                'config_count': 0,
                'error': str(e)
            }
    
    def run_full_initialization(self):
        """Run the complete timeframe database initialization."""
        logger.info("🚀 Starting timeframe database initialization...")
        
        try:
            # Check current status
            status = self.get_installation_status()
            if status.get('installed') and status.get('config_count', 0) > 0:
                logger.info(f"✅ Timeframe schema already installed with {status['config_count']} configurations")
                return True
            
            # Connect to database
            conn = self.connect()
            logger.info("✅ Connected to database")
            
            # Execute schema
            if not self.execute_schema_file(conn):
                logger.error("❌ Failed to execute schema file")
                conn.close()
                return False
            
            # Populate configurations
            if not self.populate_timeframe_configs(conn):
                logger.error("❌ Failed to populate timeframe configurations")
                conn.close()
                return False
            
            # Verify installation
            if not self.verify_installation(conn):
                logger.error("❌ Installation verification failed")
                conn.close()
                return False
            
            conn.close()
            logger.info("🎉 Timeframe database initialization completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Initialize timeframe-enhanced database schema')
    parser.add_argument('--database-url', help='PostgreSQL connection string')
    parser.add_argument('--status', action='store_true', help='Check installation status only')
    parser.add_argument('--verify', action='store_true', help='Verify installation only')
    parser.add_argument('--force', action='store_true', help='Force reinstallation')
    
    args = parser.parse_args()
    
    # Initialize the database initializer
    initializer = TimeframeDatabaseInitializer(args.database_url)
    
    if args.status:
        # Check status only
        status = initializer.get_installation_status()
        print(f"Installation Status: {'✅ Installed' if status['installed'] else '❌ Not Installed'}")
        print(f"Configuration Count: {status.get('config_count', 0)}")
        print(f"Database URL: {status.get('database_url', 'N/A')}")
        if 'error' in status:
            print(f"Error: {status['error']}")
        return
    
    if args.verify:
        # Verify installation only
        try:
            conn = initializer.connect()
            success = initializer.verify_installation(conn)
            conn.close()
            print(f"Verification: {'✅ Passed' if success else '❌ Failed'}")
        except Exception as e:
            print(f"Verification Error: {e}")
        return
    
    # Check if already installed (unless forced)
    if not args.force:
        status = initializer.get_installation_status()
        if status.get('installed') and status.get('config_count', 0) > 0:
            print(f"✅ Already installed with {status['config_count']} configurations")
            print("Use --force to reinstall or --verify to check integrity")
            return
    
    # Run full initialization
    success = initializer.run_full_initialization()
    exit_code = 0 if success else 1
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
