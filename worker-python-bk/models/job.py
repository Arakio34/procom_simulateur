from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from exceptions import JobValidationError

VALID_TASKS = {"ML", "DAS", "MV"}
INPUT_KEY_FIELDS = ("inputKey", "input", "objectKey", "rfKey")


def _parse_tasks(raw: Optional[str]) -> List[str]:
    if not raw:
        return ["ML"]
    tasks = []
    for part in raw.split(","):
        item = part.strip().upper()
        if item in VALID_TASKS:
            tasks.append(item)
    return tasks or ["ML"]


def _extract_input_key(payload: Dict[str, str]) -> Optional[str]:
    for field in INPUT_KEY_FIELDS:
        value = payload.get(field)
        if value:
            return value
    return None


@dataclass(frozen=True)
class Job:
    job_id: str
    user_id: str
    tasks: List[str]
    input_key: Optional[str]
    payload: Dict[str, str]

    @classmethod
    def from_payload(cls, payload: Dict[str, str]) -> "Job":
        job_id = payload.get("jobId") or payload.get("job_id")
        user_id = payload.get("userId") or payload.get("user_id")

        if not job_id:
            raise JobValidationError("Missing jobId in payload")
        if not user_id:
            raise JobValidationError("Missing userId in payload")

        tasks = _parse_tasks(payload.get("tasks"))
        input_key = _extract_input_key(payload)

        return cls(
            job_id=str(job_id),
            user_id=str(user_id),
            tasks=tasks,
            input_key=input_key,
            payload=dict(payload),
        )
