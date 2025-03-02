from typing import Dict
import boto3
from botocore.exceptions import ClientError
import logging
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class StorageManager:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket = settings.AWS_S3_BUCKET

    def get_user_storage_path(self, user_id: str, video_id: str = None) -> str:
        """Generate the user-specific storage path"""
        base_path = f"users/{user_id}"
        return f"{base_path}" if video_id else base_path

    async def calculate_user_storage(self, user_id: str) -> Dict:
        """Calculate total storage used by user"""
        try:
            total_size = 0
            file_count = 0
            storage_by_type = {
                'videos': 0,
                'masks': 0,
                'thumbnails': 0,
                'other': 0
            }

            # List all objects in user's directory
            paginator = self.s3.get_paginator('list_objects_v2')
            prefix = self.get_user_storage_path(user_id)
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        size = obj['Size']
                        total_size += size
                        file_count += 1
                        
                        # Categorize storage by file type
                        key = obj['Key']
                        if key.endswith(('.mp4', '.mov', '.avi')):
                            storage_by_type['videos'] += size
                        elif 'mask' in key:
                            storage_by_type['masks'] += size
                        elif 'thumbnail' in key:
                            storage_by_type['thumbnails'] += size
                        else:
                            storage_by_type['other'] += size

            return {
                'total_bytes': total_size,
                'total_gb': round(total_size / (1024 ** 3), 2),
                'file_count': file_count,
                'breakdown': {
                    'videos_gb': round(storage_by_type['videos'] / (1024 ** 3), 2),
                    'masks_gb': round(storage_by_type['masks'] / (1024 ** 3), 2),
                    'thumbnails_gb': round(storage_by_type['thumbnails'] / (1024 ** 3), 2),
                    'other_gb': round(storage_by_type['other'] / (1024 ** 3), 2)
                }
            }

        except ClientError as e:
            logger.error(f"Error calculating storage for user {user_id}: {str(e)}")
            raise

    async def update_file_location(self, old_key: str, user_id: str, video_id: str, file_type: str) -> str:
        """Move file to user-specific directory and return new key"""
        try:
            new_key = f"{self.get_user_storage_path(user_id, video_id)}/{file_type}/{old_key.split('/')[-1]}"
            
            # Copy object to new location
            self.s3.copy_object(
                Bucket=self.bucket,
                CopySource={'Bucket': self.bucket, 'Key': old_key},
                Key=new_key
            )
            
            # Delete old object
            self.s3.delete_object(Bucket=self.bucket, Key=old_key)
            
            return new_key
        except ClientError as e:
            logger.error(f"Error updating file location: {str(e)}")
            raise 