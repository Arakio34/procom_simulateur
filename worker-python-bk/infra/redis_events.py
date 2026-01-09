from __future__ import annotations

import json
from typing import Dict, Optional

import redis


class RedisEvents:
    def __init__(
        self,
        client: redis.Redis,
        output_stream: str,
        sse_channel: str,
    ) -> None:
        self._client = client
        self._output_stream = output_stream
        self._sse_channel = sse_channel

    def publish_done(self, job_id: str, user_id: str, outputs: Dict[str, str]) -> None:
        self._client.xadd(
            self._output_stream,
            {
                "jobId": job_id,
                "userId": user_id,
                "status": "done",
                "outputs": json.dumps(outputs),
            },
        )
        self._client.publish(
            self._sse_channel,
            json.dumps(
                {
                    "type": "jobUpdate",
                    "userId": user_id,
                    "jobId": job_id,
                    "status": "done",
                }
            ),
        )

    def publish_failed(self, job_id: str, user_id: str, error: str) -> None:
        self._client.xadd(
            self._output_stream,
            {
                "jobId": job_id,
                "userId": user_id,
                "status": "failed",
                "error": error,
            },
        )
        self._client.publish(
            self._sse_channel,
            json.dumps(
                {
                    "type": "jobUpdate",
                    "userId": user_id,
                    "jobId": job_id,
                    "status": "failed",
                    "error": error,
                }
            ),
        )

    def publish_progress(
        self,
        job_id: str,
        user_id: str,
        status: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        payload = {
            "type": "jobUpdate",
            "userId": user_id,
            "jobId": job_id,
            "status": status,
        }
        if metadata:
            payload.update(metadata)
        self._client.publish(self._sse_channel, json.dumps(payload))
