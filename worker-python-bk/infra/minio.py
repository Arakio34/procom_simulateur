from __future__ import annotations

from io import BytesIO
import logging
from typing import Optional

from minio import Minio
from minio.error import S3Error

from exceptions import InfrastructureError


class MinioStorage:
    def __init__(self, client: Minio, bucket: str, dry_run: bool = False) -> None:
        self._client = client
        self._bucket = bucket
        self._dry_run = dry_run

    def ensure_bucket(self) -> None:
        if self._dry_run:
            logging.info("Dry-run: skipping bucket creation")
            return
        try:
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
                logging.info("Created MinIO bucket '%s'", self._bucket)
        except S3Error as exc:
            raise InfrastructureError(f"Unable to ensure bucket '{self._bucket}': {exc}")

    def download_bytes(self, object_key: str) -> bytes:
        if self._dry_run:
            logging.info("Dry-run: skipping download for %s", object_key)
            return b""
        try:
            response = self._client.get_object(self._bucket, object_key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except S3Error as exc:
            raise InfrastructureError(f"Unable to download '{object_key}': {exc}")

    def upload_bytes(self, object_key: str, data: bytes, content_type: str) -> None:
        if self._dry_run:
            logging.info("Dry-run: skipping upload for %s", object_key)
            return
        try:
            stream = BytesIO(data)
            self._client.put_object(
                self._bucket,
                object_key,
                stream,
                length=len(data),
                content_type=content_type,
            )
        except S3Error as exc:
            raise InfrastructureError(f"Unable to upload '{object_key}': {exc}")
