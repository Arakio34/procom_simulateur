from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import redis


class RedisStreams:
    def __init__(
        self,
        client: redis.Redis,
        stream: str,
        group: str,
        consumer: str,
        read_count: int,
        block_ms: int,
    ) -> None:
        self._client = client
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._read_count = read_count
        self._block_ms = block_ms

    def ensure_group(self) -> None:
        try:
            self._client.xgroup_create(
                self._stream,
                self._group,
                id="0",
                mkstream=True,
            )
            logging.info("Created Redis consumer group '%s'", self._group)
        except redis.exceptions.ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                logging.info("Redis consumer group '%s' already exists", self._group)
            else:
                raise

    def read(self) -> List[Tuple[str, Dict[str, str]]]:
        resp = self._client.xreadgroup(
            self._group,
            self._consumer,
            {self._stream: ">"},
            count=self._read_count,
            block=self._block_ms,
        )
        if not resp:
            return []

        messages: List[Tuple[str, Dict[str, str]]] = []
        for _, entries in resp:
            for message_id, fields in entries:
                messages.append((message_id, fields))
        return messages

    def ack(self, message_id: str) -> None:
        self._client.xack(self._stream, self._group, message_id)
