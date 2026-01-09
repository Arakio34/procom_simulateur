from __future__ import annotations

from typing import Dict, List, Optional

from exceptions import ProcessingError
from models.job import Job
from processing.das import process_das
from processing.ml import process_ml
from processing.mv import process_mv


def process_task(task: str, job: Job, input_bytes: Optional[bytes]) -> List[Dict[str, object]]:
    task_upper = task.upper()
    if task_upper == "DAS":
        return process_das(job, input_bytes)
    if task_upper == "MV":
        return process_mv(job, input_bytes)
    if task_upper == "ML":
        return process_ml(job, input_bytes)
    raise ProcessingError(f"Unsupported task '{task}'")
