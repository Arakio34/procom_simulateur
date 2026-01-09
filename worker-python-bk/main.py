from __future__ import annotations

import logging
import signal
import threading

import redis
from minio import Minio

from config import Config
from infra.minio import MinioStorage
from infra.redis_events import RedisEvents
from infra.redis_streams import RedisStreams
from worker import Worker


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def main() -> int:
    config = Config.from_env()
    _configure_logging(config.log_level)

    redis_client = redis.Redis.from_url(config.redis_url, decode_responses=True)
    minio_client = Minio(
        config.minio_endpoint,
        access_key=config.minio_access_key,
        secret_key=config.minio_secret_key,
        secure=config.minio_secure,
    )

    streams = RedisStreams(
        redis_client,
        config.input_stream,
        config.redis_group,
        config.redis_consumer,
        config.read_count,
        config.block_ms,
    )
    events = RedisEvents(redis_client, config.output_stream, config.sse_channel)
    storage = MinioStorage(minio_client, config.minio_bucket, dry_run=config.dry_run)

    storage.ensure_bucket()
    streams.ensure_group()

    worker = Worker(config, streams, events, storage)

    stop_event = threading.Event()

    def _handle_signal(signum, frame):
        logging.info("Received signal %s, shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logging.info("Worker started, waiting for jobs...")

    while not stop_event.is_set():
        messages = streams.read()
        if not messages:
            continue
        for message_id, fields in messages:
            worker.process_message(message_id, fields)

    logging.info("Worker stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
