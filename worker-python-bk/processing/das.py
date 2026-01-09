from __future__ import annotations

from typing import Dict, List, Optional

from exceptions import ProcessingError
from models.job import Job
from processing.beamforming import beamforming
from processing.utils import (
    load_rf_from_bytes,
    normalize_rf,
    parse_float,
    parse_int,
    render_bmode_png,
)


def process_das(job: Job, input_bytes: Optional[bytes]) -> List[Dict[str, object]]:
    if input_bytes is None:
        raise ProcessingError("DAS processing requires input data")

    snr_db = parse_float(job.payload.get("snr")) or 10.0
    nelem = parse_int(job.payload.get("nelem"))

    rf = load_rf_from_bytes(input_bytes, job.input_key)
    rf, nelem = normalize_rf(rf, nelem)

    data = beamforming(rf, nelem, snr_db)
    png_bytes = render_bmode_png(data)

    return [
        {
            "filename": "result.png",
            "data": png_bytes,
            "content_type": "image/png",
        }
    ]
