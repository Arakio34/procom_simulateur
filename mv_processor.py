import os
import tempfile
from typing import Dict

from minio import Minio
from minio.error import S3Error

from simulateur.beamforming import mvdr_beamforming
from simulateur.utils import save_image

from das_processor import (
    DasProcessingError,
    _extract_input_key,
    _download_input,
    _load_rf,
    _parse_float,
    _parse_int,
)


class MVProcessingError(Exception):
    """Raised when MV processing cannot proceed."""


def process_mv_task(
    minio_client: Minio,
    bucket: str,
    payload: Dict[str, str],
) -> Dict[str, str]:
    """
    Run MV beamforming for a job.

    Returns a mapping with uploaded object keys.
    """
    object_key = _extract_input_key(payload)
    snr_db = _parse_float(payload.get("snr")) or 10.0
    regularization = _parse_float(payload.get("mv_reg")) or 0.1
    nelem = _parse_int(payload.get("nelem"))
    job_id = payload.get("jobId", "unknown-job")
    user_id = payload.get("userId", "unknown-user")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, os.path.basename(object_key))
        try:
            _download_input(minio_client, bucket, object_key, input_path)
        except DasProcessingError as exc:
            raise MVProcessingError(str(exc))

        try:
            rf = _load_rf(input_path)
        except DasProcessingError as exc:
            raise MVProcessingError(str(exc))
        nelem = nelem or (rf.shape[1] if rf.ndim == 2 else None)

        if nelem is None:
            raise MVProcessingError("Unable to infer number of elements (nelem) from RF input")

        data = mvdr_beamforming(
            rf,
            Nelem=nelem,
            SNR_dB=snr_db,
            regularization=regularization,
        )

        png_path = os.path.join(tmpdir, "mv_result.png")
        save_image(png_path, data)

        base_prefix = f"{user_id}/{job_id}/output/MV"
        png_object_key = f"{base_prefix}/result.png"

        try:
            minio_client.fput_object(
                bucket,
                png_object_key,
                png_path,
                content_type="image/png",
            )
        except S3Error as exc:
            raise MVProcessingError(f"Unable to upload MV PNG result: {exc}")

        return {
            "png": png_object_key,
        }
