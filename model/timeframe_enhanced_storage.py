#!/usr/bin/env python3
"""
Timeframe-Enhanced Storage Manager

This module extends the existing storage capabilities to support multiple timeframes
while maintaining backward compatibility with the current daily storage structure.

Key Features:
- Timeframe-aware storage paths in MinIO
- Backward compatibility with existing daily structure
- Enhanced metadata for timeframe tracking
- Automatic folder organization by timeframe
"""

import boto3
import pandas as pd
import io
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeframeEnhancedStorage:
    """
    Enhanced storage manager that organizes data by timeframe while maintaining
    backward compatibility with existing daily storage structure.
    """
    
    def __init__(self, s3_endpoint: str = "http://minio:9000", 
                 access_key: str = "minioadmin", 
                 secret_key: str = "minioadmin",
                 bucket_name: str = "breadthflow"):
        """
        Initialize the timeframe-enhanced storage manager.
        
        Args:
            s3_endpoint: MinIO S3 endpoint
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket_name: S3 bucket name
        """
        self.bucket_name = bucket_name
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=s3_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='us-east-1'
        )
        
        # Timeframe-aware storage paths
        self.storage_paths = {
            # OHLCV data paths
            'ohlcv': {
                '1day': 'ohlcv/daily/',     # Existing structure (unchanged)
                '1hour': 'ohlcv/hourly/',   # NEW
                '15min': 'ohlcv/minute/',   # NEW (15min and 5min in minute folder)
                '5min': 'ohlcv/minute/',    # NEW
                '1min': 'ohlcv/minute/',    # NEW
            },
            # Trading signals paths
            'trading_signals': {
                '1day': 'trading_signals/daily/',    # Existing structure (unchanged)
                '1hour': 'trading_signals/hourly/',  # NEW
                '15min': 'trading_signals/minute/',  # NEW
                '5min': 'trading_signals/minute/',   # NEW
                '1min': 'trading_signals/minute/',   # NEW
            },
            # Analytics paths
            'analytics': {
                '1day': 'analytics/daily/',    # Existing structure (unchanged)
                '1hour': 'analytics/hourly/',  # NEW
                '15min': 'analytics/minute/',  # NEW
                '5min': 'analytics/minute/',   # NEW
                '1min': 'analytics/minute/',   # NEW
            },
            # Metadata path
            'metadata': 'metadata/timeframes/'  # NEW
        }
        
        # Supported timeframes
        self.supported_timeframes = ['1min', '5min', '15min', '1hour', '1day']
        
        # File naming patterns for different timeframes
        self.file_patterns = {
            '1day': '{symbol}_{start_date}_{end_date}.parquet',
            '1hour': '{symbol}_{start_date}_{end_date}_1H.parquet',
            '15min': '{symbol}_{start_date}_{end_date}_15M.parquet',
            '5min': '{symbol}_{start_date}_{end_date}_5M.parquet',
            '1min': '{symbol}_{start_date}_{end_date}_1M.parquet'
        }
        
        logger.info(f"TimeframeEnhancedStorage initialized for bucket: {bucket_name}")
    
    def get_storage_path(self, data_type: str, timeframe: str = '1day') -> str:
        """
        Get the storage path for a specific data type and timeframe.
        
        Args:
            data_type: Type of data ('ohlcv', 'trading_signals', 'analytics')
            timeframe: Data timeframe (default: '1day' for backward compatibility)
            
        Returns:
            Storage path string
        """
        if timeframe not in self.supported_timeframes:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {self.supported_timeframes}")
        
        if data_type not in self.storage_paths:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        return self.storage_paths[data_type][timeframe]
    
    def get_file_name(self, symbol: str, start_date: str, end_date: str, 
                      timeframe: str = '1day', file_format: str = 'parquet') -> str:
        """
        Generate a timeframe-aware file name.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe
            file_format: File format ('parquet', 'json')
            
        Returns:
            Generated file name
        """
        if timeframe not in self.file_patterns:
            raise ValueError(f"No file pattern for timeframe: {timeframe}")
        
        base_name = self.file_patterns[timeframe].format(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Replace .parquet with the desired format if different
        if file_format != 'parquet':
            base_name = base_name.replace('.parquet', f'.{file_format}')
        
        return base_name
    
    def save_ohlcv_data(self, data: pd.DataFrame, symbol: str, 
                        start_date: str, end_date: str, 
                        timeframe: str = '1day',
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Save OHLCV data with timeframe awareness.
        
        Args:
            data: OHLCV DataFrame
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe (default: '1day' for backward compatibility)
            metadata: Additional metadata
            
        Returns:
            Result dictionary with storage information
        """
        try:
            # Get storage path for this timeframe
            base_path = self.get_storage_path('ohlcv', timeframe)
            
            # Create symbol-specific folder
            storage_path = f"{base_path}{symbol}/"
            
            # Generate file name
            file_name = self.get_file_name(symbol, start_date, end_date, timeframe)
            full_key = f"{storage_path}{file_name}"
            
            # Prepare metadata
            enhanced_metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'records_count': len(data),
                'storage_timestamp': datetime.now().isoformat(),
                'file_format': 'parquet',
                'data_type': 'ohlcv'
            }
            
            if metadata:
                enhanced_metadata.update(metadata)
            
            # Convert DataFrame to Parquet bytes
            parquet_buffer = io.BytesIO()
            data.to_parquet(parquet_buffer, index=False)
            parquet_buffer.seek(0)
            
            # Upload to MinIO
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=full_key,
                Body=parquet_buffer.getvalue(),
                Metadata={k: str(v) for k, v in enhanced_metadata.items()}
            )
            
            # Save metadata separately
            metadata_key = f"{self.storage_paths['metadata']}{symbol}_{timeframe}_{start_date}_{end_date}_metadata.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(enhanced_metadata, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Saved {len(data)} records for {symbol} ({timeframe}) to {full_key}")
            
            return {
                'success': True,
                'storage_path': full_key,
                'metadata_path': metadata_key,
                'records_saved': len(data),
                'timeframe': timeframe,
                'metadata': enhanced_metadata
            }
            
        except Exception as e:
            logger.error(f"Error saving OHLCV data for {symbol} ({timeframe}): {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'timeframe': timeframe
            }
    
    def save_signals_data(self, signals: List[Dict[str, Any]], 
                         timeframe: str = '1day',
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Save trading signals with timeframe awareness.
        
        Args:
            signals: List of signal dictionaries
            timeframe: Data timeframe (default: '1day' for backward compatibility)
            metadata: Additional metadata
            
        Returns:
            Result dictionary with storage information
        """
        try:
            # Get storage path for this timeframe
            base_path = self.get_storage_path('trading_signals', timeframe)
            
            # Generate timestamp-based file name for signals
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"signals_{timestamp}_{timeframe}.json"
            full_key = f"{base_path}{file_name}"
            
            # Prepare metadata
            enhanced_metadata = {
                'signals_count': len(signals),
                'timeframe': timeframe,
                'generation_timestamp': datetime.now().isoformat(),
                'file_format': 'json',
                'data_type': 'trading_signals'
            }
            
            if metadata:
                enhanced_metadata.update(metadata)
            
            # Prepare signals data with metadata
            signals_data = {
                'metadata': enhanced_metadata,
                'signals': signals
            }
            
            # Upload to MinIO
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=full_key,
                Body=json.dumps(signals_data, indent=2),
                ContentType='application/json',
                Metadata={k: str(v) for k, v in enhanced_metadata.items()}
            )
            
            logger.info(f"Saved {len(signals)} signals ({timeframe}) to {full_key}")
            
            return {
                'success': True,
                'storage_path': full_key,
                'signals_saved': len(signals),
                'timeframe': timeframe,
                'metadata': enhanced_metadata
            }
            
        except Exception as e:
            logger.error(f"Error saving signals data ({timeframe}): {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timeframe': timeframe
            }
    
    def list_data_by_timeframe(self, data_type: str, timeframe: str = '1day') -> List[Dict[str, Any]]:
        """
        List stored data files for a specific timeframe.
        
        Args:
            data_type: Type of data ('ohlcv', 'trading_signals', 'analytics')
            timeframe: Data timeframe
            
        Returns:
            List of file information dictionaries
        """
        try:
            base_path = self.get_storage_path(data_type, timeframe)
            
            # List objects with the timeframe prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=base_path
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    file_info = {
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'timeframe': timeframe,
                        'data_type': data_type
                    }
                    
                    # Try to get metadata
                    try:
                        metadata_response = self.s3_client.head_object(
                            Bucket=self.bucket_name,
                            Key=obj['Key']
                        )
                        if 'Metadata' in metadata_response:
                            file_info['metadata'] = metadata_response['Metadata']
                    except:
                        pass  # Metadata not available
                    
                    files.append(file_info)
            
            logger.info(f"Found {len(files)} {data_type} files for timeframe {timeframe}")
            return files
            
        except Exception as e:
            logger.error(f"Error listing {data_type} data for timeframe {timeframe}: {str(e)}")
            return []
    
    def get_timeframe_summary(self) -> Dict[str, Any]:
        """
        Get a summary of data across all timeframes.
        
        Returns:
            Summary dictionary with timeframe statistics
        """
        summary = {
            'timeframes': {},
            'total_files': 0,
            'total_size_mb': 0.0,
            'data_types': ['ohlcv', 'trading_signals', 'analytics']
        }
        
        for timeframe in self.supported_timeframes:
            timeframe_info = {
                'timeframe': timeframe,
                'data_types': {}
            }
            
            for data_type in ['ohlcv', 'trading_signals', 'analytics']:
                files = self.list_data_by_timeframe(data_type, timeframe)
                
                total_size = sum(f['size'] for f in files)
                timeframe_info['data_types'][data_type] = {
                    'file_count': len(files),
                    'total_size_bytes': total_size,
                    'total_size_mb': round(total_size / (1024 * 1024), 2)
                }
                
                summary['total_files'] += len(files)
                summary['total_size_mb'] += timeframe_info['data_types'][data_type]['total_size_mb']
            
            summary['timeframes'][timeframe] = timeframe_info
        
        summary['total_size_mb'] = round(summary['total_size_mb'], 2)
        
        logger.info(f"Generated timeframe summary: {summary['total_files']} files, {summary['total_size_mb']} MB")
        return summary
    
    def migrate_existing_data_structure(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Migrate existing daily data to the new timeframe-aware structure.
        
        This method helps transition from the old structure to the new one
        while maintaining backward compatibility.
        
        Args:
            dry_run: If True, only report what would be done without making changes
            
        Returns:
            Migration report
        """
        migration_report = {
            'dry_run': dry_run,
            'files_to_migrate': [],
            'migration_actions': [],
            'errors': []
        }
        
        try:
            # Check for files in the old structure (direct ohlcv/ path)
            old_ohlcv_response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='ohlcv/',
                Delimiter='/'  # Only get direct children, not nested
            )
            
            if 'Contents' in old_ohlcv_response:
                for obj in old_ohlcv_response['Contents']:
                    key = obj['Key']
                    
                    # Skip if already in a timeframe folder
                    if any(tf in key for tf in ['daily/', 'hourly/', 'minute/']):
                        continue
                    
                    # This is an old structure file that needs migration
                    old_path = key
                    new_path = key.replace('ohlcv/', 'ohlcv/daily/')
                    
                    migration_report['files_to_migrate'].append({
                        'old_path': old_path,
                        'new_path': new_path,
                        'size': obj['Size']
                    })
                    
                    if not dry_run:
                        # Perform the migration
                        try:
                            # Copy to new location
                            self.s3_client.copy_object(
                                Bucket=self.bucket_name,
                                CopySource={'Bucket': self.bucket_name, 'Key': old_path},
                                Key=new_path
                            )
                            
                            # Delete old file
                            self.s3_client.delete_object(
                                Bucket=self.bucket_name,
                                Key=old_path
                            )
                            
                            migration_report['migration_actions'].append(f"Migrated {old_path} -> {new_path}")
                            
                        except Exception as e:
                            migration_report['errors'].append(f"Failed to migrate {old_path}: {str(e)}")
            
            # Do the same for trading_signals and analytics
            for data_type in ['trading_signals', 'analytics']:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=f'{data_type}/',
                    Delimiter='/'
                )
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        key = obj['Key']
                        
                        # Skip if already in a timeframe folder
                        if any(tf in key for tf in ['daily/', 'hourly/', 'minute/']):
                            continue
                        
                        old_path = key
                        new_path = key.replace(f'{data_type}/', f'{data_type}/daily/')
                        
                        migration_report['files_to_migrate'].append({
                            'old_path': old_path,
                            'new_path': new_path,
                            'size': obj['Size']
                        })
                        
                        if not dry_run:
                            try:
                                self.s3_client.copy_object(
                                    Bucket=self.bucket_name,
                                    CopySource={'Bucket': self.bucket_name, 'Key': old_path},
                                    Key=new_path
                                )
                                
                                self.s3_client.delete_object(
                                    Bucket=self.bucket_name,
                                    Key=old_path
                                )
                                
                                migration_report['migration_actions'].append(f"Migrated {old_path} -> {new_path}")
                                
                            except Exception as e:
                                migration_report['errors'].append(f"Failed to migrate {old_path}: {str(e)}")
            
            logger.info(f"Migration report: {len(migration_report['files_to_migrate'])} files to migrate")
            return migration_report
            
        except Exception as e:
            migration_report['errors'].append(f"Migration failed: {str(e)}")
            logger.error(f"Error during migration: {str(e)}")
            return migration_report

# Factory function for backward compatibility
def create_timeframe_storage() -> TimeframeEnhancedStorage:
    """Create a timeframe-enhanced storage instance."""
    return TimeframeEnhancedStorage()

# Example usage and testing
if __name__ == "__main__":
    # Test the storage manager
    storage = create_timeframe_storage()
    
    print("Supported timeframes:", storage.supported_timeframes)
    
    # Test storage paths
    print("\n=== Storage Path Examples ===")
    for timeframe in ['1day', '1hour', '15min']:
        ohlcv_path = storage.get_storage_path('ohlcv', timeframe)
        signals_path = storage.get_storage_path('trading_signals', timeframe)
        print(f"{timeframe}: OHLCV={ohlcv_path}, Signals={signals_path}")
    
    # Test file naming
    print("\n=== File Name Examples ===")
    for timeframe in ['1day', '1hour', '15min']:
        file_name = storage.get_file_name('AAPL', '2024-08-15', '2024-08-16', timeframe)
        print(f"{timeframe}: {file_name}")
    
    # Test timeframe summary
    print("\n=== Timeframe Summary ===")
    summary = storage.get_timeframe_summary()
    print(f"Total files across all timeframes: {summary['total_files']}")
    print(f"Total size: {summary['total_size_mb']} MB")
