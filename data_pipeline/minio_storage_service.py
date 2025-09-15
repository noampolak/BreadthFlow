"""
MinIO Storage Service for ML Pipeline

Handles data storage and retrieval from MinIO object storage
with proper organization for ML training workflows.
"""

import logging
import io
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
from pyspark.sql import DataFrame

try:
    from minio import Minio
    from minio.error import S3Error

    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    Minio = None
    S3Error = Exception

logger = logging.getLogger(__name__)


class MinIOStorageService:
    """
    Service for storing and retrieving data from MinIO object storage.

    Features:
    - Organized data storage by type and date
    - Support for multiple data formats (Parquet, CSV, JSON)
    - Data versioning and lineage tracking
    - Efficient data retrieval for ML training
    - Data lifecycle management
    """

    def __init__(
        self, endpoint: str = "minio:9000", access_key: str = "admin", secret_key: str = "password123", secure: bool = False
    ):
        """
        Initialize MinIO storage service.

        Args:
            endpoint: MinIO server endpoint
            access_key: MinIO access key
            secret_key: MinIO secret key
            secure: Use HTTPS connection
        """
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.client = None
        self.bucket_name = "breadthflow-ml-data"

        if MINIO_AVAILABLE:
            try:
                self.client = Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
                logger.info(f"MinIO client initialized for {endpoint}")
            except Exception as e:
                logger.error(f"Failed to initialize MinIO client: {str(e)}")
                self.client = None
        else:
            logger.warning("MinIO library not available. Install with: pip install minio")

        logger.info("MinIOStorageService initialized")

    def create_bucket(self, bucket_name: str = None) -> bool:
        """
        Create a bucket in MinIO.

        Args:
            bucket_name: Name of bucket to create (default: self.bucket_name)

        Returns:
            True if bucket created successfully
        """
        if not self.client:
            logger.error("MinIO client not available")
            return False

        bucket = bucket_name or self.bucket_name

        try:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                logger.info(f"Created bucket: {bucket}")
            else:
                logger.info(f"Bucket already exists: {bucket}")
            return True

        except S3Error as e:
            logger.error(f"Error creating bucket {bucket}: {str(e)}")
            return False

    def store_dataframe(
        self, df: Union[DataFrame, pd.DataFrame], object_path: str, format: str = "parquet", bucket_name: str = None
    ) -> Dict[str, Any]:
        """
        Store DataFrame in MinIO.

        Args:
            df: DataFrame to store
            object_path: Path in bucket to store the data
            format: Data format (parquet, csv, json)
            bucket_name: Bucket name (default: self.bucket_name)

        Returns:
            Dictionary with storage results
        """
        if not self.client:
            return {"success": False, "error": "MinIO client not available"}

        bucket = bucket_name or self.bucket_name

        try:
            # Ensure bucket exists
            if not self.create_bucket(bucket):
                return {"success": False, "error": "Failed to create bucket"}

            # Convert Spark DataFrame to Pandas if needed
            if hasattr(df, "toPandas"):
                pandas_df = df.toPandas()
            else:
                pandas_df = df

            # Convert to bytes based on format
            if format.lower() == "parquet":
                data_bytes = pandas_df.to_parquet()
            elif format.lower() == "csv":
                data_bytes = pandas_df.to_csv(index=False).encode("utf-8")
            elif format.lower() == "json":
                data_bytes = pandas_df.to_json(orient="records").encode("utf-8")
            else:
                return {"success": False, "error": f"Unsupported format: {format}"}

            # Store in MinIO
            self.client.put_object(
                bucket_name=bucket, object_name=object_path, data=io.BytesIO(data_bytes), length=len(data_bytes)
            )

            logger.info(f"Stored DataFrame at {bucket}/{object_path}")

            return {
                "success": True,
                "bucket": bucket,
                "object_path": object_path,
                "format": format,
                "size_bytes": len(data_bytes),
                "records": len(pandas_df),
            }

        except Exception as e:
            logger.error(f"Error storing DataFrame: {str(e)}")
            return {"success": False, "error": str(e)}

    def load_dataframe(self, object_path: str, format: str = "parquet", bucket_name: str = None) -> Dict[str, Any]:
        """
        Load DataFrame from MinIO.

        Args:
            object_path: Path in bucket to load data from
            format: Data format (parquet, csv, json)
            bucket_name: Bucket name (default: self.bucket_name)

        Returns:
            Dictionary with loaded data and metadata
        """
        if not self.client:
            return {"success": False, "error": "MinIO client not available"}

        bucket = bucket_name or self.bucket_name

        try:
            # Get object from MinIO
            response = self.client.get_object(bucket, object_path)
            data_bytes = response.read()
            response.close()
            response.release_conn()

            # Convert bytes to DataFrame based on format
            if format.lower() == "parquet":
                df = pd.read_parquet(io.BytesIO(data_bytes))
            elif format.lower() == "csv":
                df = pd.read_csv(io.BytesIO(data_bytes))
            elif format.lower() == "json":
                df = pd.read_json(io.BytesIO(data_bytes))
            else:
                return {"success": False, "error": f"Unsupported format: {format}"}

            logger.info(f"Loaded DataFrame from {bucket}/{object_path}")

            return {
                "success": True,
                "data": df,
                "bucket": bucket,
                "object_path": object_path,
                "format": format,
                "size_bytes": len(data_bytes),
                "records": len(df),
            }

        except S3Error as e:
            if e.code == "NoSuchKey":
                return {"success": False, "error": f"Object not found: {object_path}"}
            else:
                logger.error(f"Error loading DataFrame: {str(e)}")
                return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error loading DataFrame: {str(e)}")
            return {"success": False, "error": str(e)}

    def list_objects(self, prefix: str = "", bucket_name: str = None, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        List objects in MinIO bucket.

        Args:
            prefix: Object name prefix to filter
            bucket_name: Bucket name (default: self.bucket_name)
            recursive: List objects recursively

        Returns:
            List of object metadata
        """
        if not self.client:
            return []

        bucket = bucket_name or self.bucket_name

        try:
            objects = self.client.list_objects(bucket_name=bucket, prefix=prefix, recursive=recursive)

            object_list = []
            for obj in objects:
                object_list.append(
                    {"object_name": obj.object_name, "size": obj.size, "last_modified": obj.last_modified, "etag": obj.etag}
                )

            logger.info(f"Listed {len(object_list)} objects with prefix '{prefix}'")
            return object_list

        except Exception as e:
            logger.error(f"Error listing objects: {str(e)}")
            return []

    def delete_object(self, object_path: str, bucket_name: str = None) -> bool:
        """
        Delete object from MinIO.

        Args:
            object_path: Path in bucket to delete
            bucket_name: Bucket name (default: self.bucket_name)

        Returns:
            True if object deleted successfully
        """
        if not self.client:
            return False

        bucket = bucket_name or self.bucket_name

        try:
            self.client.remove_object(bucket, object_path)
            logger.info(f"Deleted object: {bucket}/{object_path}")
            return True

        except Exception as e:
            logger.error(f"Error deleting object: {str(e)}")
            return False

    def get_object_info(self, object_path: str, bucket_name: str = None) -> Dict[str, Any]:
        """
        Get object metadata from MinIO.

        Args:
            object_path: Path in bucket
            bucket_name: Bucket name (default: self.bucket_name)

        Returns:
            Dictionary with object metadata
        """
        if not self.client:
            return {"success": False, "error": "MinIO client not available"}

        bucket = bucket_name or self.bucket_name

        try:
            stat = self.client.stat_object(bucket, object_path)

            return {
                "success": True,
                "object_name": stat.object_name,
                "size": stat.size,
                "last_modified": stat.last_modified,
                "etag": stat.etag,
                "content_type": stat.content_type,
            }

        except S3Error as e:
            if e.code == "NoSuchKey":
                return {"success": False, "error": f"Object not found: {object_path}"}
            else:
                logger.error(f"Error getting object info: {str(e)}")
                return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error getting object info: {str(e)}")
            return {"success": False, "error": str(e)}

    def organize_data_path(self, data_type: str, symbol_list: str, date: datetime = None, filename: str = None) -> str:
        """
        Generate organized path for data storage.

        Args:
            data_type: Type of data (ohlcv, features, models, etc.)
            symbol_list: Name of symbol list
            date: Date for organization (default: current date)
            filename: Custom filename (default: auto-generated)

        Returns:
            Organized object path
        """
        if date is None:
            date = datetime.now()

        if filename is None:
            timestamp = date.strftime("%Y%m%d_%H%M%S")
            filename = f"data_{timestamp}.parquet"

        # Create organized path: data_type/symbol_list/year/month/day/filename
        path_parts = [data_type, symbol_list, str(date.year), f"{date.month:02d}", f"{date.day:02d}", filename]

        return "/".join(path_parts)

    def get_data_summary(self, data_type: str = None, symbol_list: str = None, bucket_name: str = None) -> Dict[str, Any]:
        """
        Get summary of stored data.

        Args:
            data_type: Filter by data type
            symbol_list: Filter by symbol list
            bucket_name: Bucket name (default: self.bucket_name)

        Returns:
            Dictionary with data summary
        """
        try:
            # Build prefix for filtering
            prefix_parts = []
            if data_type:
                prefix_parts.append(data_type)
            if symbol_list:
                prefix_parts.append(symbol_list)

            prefix = "/".join(prefix_parts) if prefix_parts else ""

            # List objects
            objects = self.list_objects(prefix=prefix, bucket_name=bucket_name)

            if not objects:
                return {"total_objects": 0, "total_size_bytes": 0, "data_types": {}, "symbol_lists": {}, "date_range": None}

            # Analyze objects
            total_size = sum(obj["size"] for obj in objects)
            data_types = {}
            symbol_lists = {}
            dates = []

            for obj in objects:
                path_parts = obj["object_name"].split("/")

                if len(path_parts) >= 1:
                    data_type = path_parts[0]
                    data_types[data_type] = data_types.get(data_type, 0) + 1

                if len(path_parts) >= 2:
                    symbol_list = path_parts[1]
                    symbol_lists[symbol_list] = symbol_lists.get(symbol_list, 0) + 1

                if len(path_parts) >= 5:
                    try:
                        year, month, day = int(path_parts[2]), int(path_parts[3]), int(path_parts[4])
                        dates.append(datetime(year, month, day))
                    except (ValueError, IndexError):
                        pass

            date_range = None
            if dates:
                date_range = {"start": min(dates), "end": max(dates)}

            return {
                "total_objects": len(objects),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "data_types": data_types,
                "symbol_lists": symbol_lists,
                "date_range": date_range,
            }

        except Exception as e:
            logger.error(f"Error getting data summary: {str(e)}")
            return {"error": str(e)}

    def cleanup_old_data(self, days_to_keep: int = 30, data_type: str = None, bucket_name: str = None) -> Dict[str, Any]:
        """
        Clean up old data based on retention policy.

        Args:
            days_to_keep: Number of days to keep data
            data_type: Filter by data type
            bucket_name: Bucket name (default: self.bucket_name)

        Returns:
            Dictionary with cleanup results
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            prefix = data_type + "/" if data_type else ""

            objects = self.list_objects(prefix=prefix, bucket_name=bucket_name)

            deleted_objects = []
            total_size_deleted = 0

            for obj in objects:
                if obj["last_modified"] < cutoff_date:
                    if self.delete_object(obj["object_name"], bucket_name):
                        deleted_objects.append(obj["object_name"])
                        total_size_deleted += obj["size"]

            return {
                "success": True,
                "deleted_objects": len(deleted_objects),
                "size_deleted_bytes": total_size_deleted,
                "size_deleted_mb": total_size_deleted / (1024 * 1024),
                "cutoff_date": cutoff_date,
            }

        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get MinIO connection status.

        Returns:
            Dictionary with connection status
        """
        if not self.client:
            return {"connected": False, "error": "MinIO client not initialized"}

        try:
            # Try to list buckets to test connection
            buckets = self.client.list_buckets()
            bucket_names = [bucket.name for bucket in buckets]

            return {
                "connected": True,
                "endpoint": self.endpoint,
                "bucket_name": self.bucket_name,
                "available_buckets": bucket_names,
                "bucket_exists": self.bucket_name in bucket_names,
            }

        except Exception as e:
            return {"connected": False, "error": str(e), "endpoint": self.endpoint}
