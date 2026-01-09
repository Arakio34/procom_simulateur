from __future__ import annotations

from typing import Dict, List, Optional

from models.job import Job


def process_ml(job: Job, input_bytes: Optional[bytes]) -> List[Dict[str, object]]:
    payload = f"MOCK RESULT ML for job {job.job_id}\n"
    return [
        {
            "filename": "result.rf",
            "data": payload.encode("ascii"),
            "content_type": "application/octet-stream",
        }
    ]
