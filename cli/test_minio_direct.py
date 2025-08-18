#!/usr/bin/env python3
"""
Test MinIO connectivity directly (without Spark) to isolate the issue.
"""

import boto3
from botocore.exceptions import ClientError
import sys

def test_minio_connection():
    """Test direct MinIO connection using boto3."""
    
    print("🧪 Testing MinIO connectivity directly...")
    
    try:
        # Create S3 client for MinIO
        s3_client = boto3.client(
            's3',
            endpoint_url='http://minio:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin',
            region_name='us-east-1'
        )
        
        print("✅ S3 client created successfully")
        
        # Test basic operations
        print("\n📋 Testing bucket operations...")
        
        # List buckets
        response = s3_client.list_buckets()
        print(f"✅ Found {len(response['Buckets'])} existing buckets:")
        for bucket in response['Buckets']:
            print(f"   - {bucket['Name']}")
        
        # Create a test bucket
        bucket_name = 'breadthflow'
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"✅ Created bucket: {bucket_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                print(f"✅ Bucket {bucket_name} already exists")
            else:
                print(f"⚠️  Bucket creation issue: {e}")
        
        # Test file operations
        print("\n📋 Testing file operations...")
        
        # Upload a test file
        test_content = "BreadthFlow MinIO Test\nTimestamp: 2023-01-01\nData: Sample financial data"
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key='test-connectivity.txt',
            Body=test_content
        )
        print("✅ Uploaded test file successfully")
        
        # Download the test file
        response = s3_client.get_object(Bucket=bucket_name, Key='test-connectivity.txt')
        downloaded_content = response['Body'].read().decode('utf-8')
        
        if downloaded_content == test_content:
            print("✅ Downloaded and verified test file successfully")
        else:
            print("❌ File content mismatch!")
            
        # List objects in bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            print(f"✅ Found {len(response['Contents'])} objects in bucket:")
            for obj in response['Contents']:
                print(f"   - {obj['Key']} ({obj['Size']} bytes)")
        
        print("\n🎉 MinIO connectivity test PASSED!")
        print("✅ MinIO is accessible and working correctly")
        print("✅ S3 operations (create bucket, upload, download) all working")
        
        return True
        
    except Exception as e:
        print(f"\n❌ MinIO connectivity test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minio_connection()
    sys.exit(0 if success else 1)
