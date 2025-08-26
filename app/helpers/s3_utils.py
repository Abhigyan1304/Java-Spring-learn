import os
import boto3
from botocore.exceptions import ClientError
from app.logger import logger

def get_s3_client():
    return boto3.client("s3")

def download_file(bucket: str, s3_key: str, local_path: str):
    s3 = get_s3_client()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        logger.info(f"Downloading s3://{bucket}/{s3_key} -> {local_path}")
        s3.download_file(bucket, s3_key, local_path)
    except ClientError as e:
        logger.error(f"Failed to download {s3_key}: {e}")
        raise

def download_folder(bucket: str, s3_prefix: str, local_dir: str):
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        if "Contents" not in page:
            logger.warning(f"No objects at s3://{bucket}/{s3_prefix}")
            return
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith("/"): 
                continue
            rel_path = os.path.relpath(key, s3_prefix)
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                s3.download_file(bucket, key, local_path)
                logger.info(f"Downloaded {key}")
            except ClientError as e:
                logger.error(f"Failed to download {key}: {e}")
                raise
