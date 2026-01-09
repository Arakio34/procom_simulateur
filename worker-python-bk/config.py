from __future__ import annotations

from dataclasses import dataclass
import os


def _get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    redis_url: str
    input_stream: str
    output_stream: str
    sse_channel: str
    redis_group: str
    redis_consumer: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    minio_bucket: str
    read_count: int
    block_ms: int
    log_level: str
    dry_run: bool

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            input_stream=os.getenv("REDIS_INPUT_STREAM", "jobs-stream"),
            output_stream=os.getenv("REDIS_OUTPUT_STREAM", "jobs-events"),
            sse_channel=os.getenv("REDIS_SSE_CHANNEL", "jobs:sse"),
            redis_group=os.getenv("REDIS_GROUP", "ml-workers"),
            redis_consumer=os.getenv("REDIS_CONSUMER", "worker-1"),
            minio_endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            minio_access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
            minio_secret_key=os.getenv("MINIO_SECRET_KEY", "admin123"),
            minio_secure=_get_env_bool("MINIO_SECURE", False),
            minio_bucket=os.getenv("MINIO_BUCKET", "uploads"),
            read_count=int(os.getenv("REDIS_READ_COUNT", "1")),
            block_ms=int(os.getenv("REDIS_BLOCK_MS", "5000")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            dry_run=_get_env_bool("DRY_RUN", False),
        )
