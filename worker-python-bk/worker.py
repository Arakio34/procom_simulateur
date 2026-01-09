from __future__ import annotations

import json
import logging
from typing import Dict, Optional

from config import Config
from exceptions import InfrastructureError, JobValidationError, ProcessingError
from infra.minio import MinioStorage
from infra.redis_events import RedisEvents
from infra.redis_streams import RedisStreams
from models.job import Job
from processing.router import process_task


class Worker:
    def __init__(
        self,
        config: Config,
        streams: RedisStreams,
        events: RedisEvents,
        storage: MinioStorage,
    ) -> None:
        self._config = config
        self._streams = streams
        self._events = events
        self._storage = storage

    def process_message(self, message_id: str, fields: Dict[str, str]) -> None:
        job: Optional[Job] = None
        try:
            job = Job.from_payload(fields)
            outputs = self._handle_job(job)
            self._events.publish_done(job.job_id, job.user_id, outputs)
            self._streams.ack(message_id)
            logging.info("Job %s done", job.job_id)
        except JobValidationError as exc:
            logging.error("Invalid job payload: %s", exc)
            if job is not None:
                self._events.publish_failed(job.job_id, job.user_id, str(exc))
            else:
                self._events.publish_failed("unknown", "unknown", str(exc))
            self._streams.ack(message_id)
        except (ProcessingError, InfrastructureError) as exc:
            logging.exception("Job %s failed", job.job_id if job else "unknown")
            if job is not None:
                self._events.publish_failed(job.job_id, job.user_id, str(exc))
            # No ACK here so Redis can retry on transient failures.
        except Exception as exc:
            logging.exception("Unexpected error: %s", exc)
            if job is not None:
                self._events.publish_failed(job.job_id, job.user_id, "unexpected error")
            # No ACK on unexpected errors.

    def _handle_job(self, job: Job) -> Dict[str, str]:
        outputs: Dict[str, str] = {}
        for task in job.tasks:
            input_bytes = None
            if task in {"DAS", "MV"}:
                if not job.input_key:
                    raise JobValidationError(
                        "Missing input key for DAS/MV processing"
                    )
                input_bytes = self._storage.download_bytes(job.input_key)

            self._events.publish_progress(
                job.job_id,
                job.user_id,
                "processing",
                {"task": task},
            )
            artifacts = process_task(task, job, input_bytes)
            if not artifacts:
                raise ProcessingError(f"No artifacts produced for task {task}")

            for artifact in artifacts:
                filename = str(artifact["filename"])
                data = artifact["data"]
                content_type = str(artifact["content_type"])

                object_key = f"{job.user_id}/{job.job_id}/output/{task}/{filename}"
                if isinstance(data, bytes):
                    self._storage.upload_bytes(object_key, data, content_type)
                else:
                    raise ProcessingError("Artifact data must be bytes")

                outputs[task] = object_key
                if filename.lower().endswith(".png"):
                    outputs[f"{task}_png"] = object_key
                else:
                    outputs[f"{task}_{filename}"] = object_key

        return outputs
