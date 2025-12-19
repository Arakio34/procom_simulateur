import os
import tempfile
from typing import Dict, Optional

import numpy as np
from minio import Minio
from minio.error import S3Error

# Force a non-interactive backend for matplotlib before importing helpers
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import h5py  # type: ignore
except ImportError as exc:  # pragma: no cover - env specific
    h5py = None
    _h5py_import_error = exc
else:
    _h5py_import_error = None

from simulateur.beamforming import beamforming
from simulateur.utils import save_image


class DasProcessingError(Exception):
    """Raised when DAS processing cannot proceed."""


def _parse_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: Optional[str], default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_input_key(payload: Dict[str, str]) -> str:
    """Find the MinIO object key containing RF data."""
    candidate_fields = (
        "inputKey",
        "input",
        "objectKey",
        "rfKey",
    )
    for field in candidate_fields:
        value = payload.get(field)
        if value:
            return value
    raise DasProcessingError(
        f"DAS task requires an input object key ({', '.join(candidate_fields)})"
    )


def _download_input(minio_client: Minio, bucket: str, object_key: str, dest_path: str):
    try:
        minio_client.fget_object(bucket, object_key, dest_path)
    except S3Error as exc:
        raise DasProcessingError(f"Unable to download input object '{object_key}': {exc}")


def _load_rf(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".npy", ".npz"):
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "rf" in data:
                return np.array(data["rf"])
            raise DasProcessingError("NPZ input must contain an 'rf' array")
        return np.array(data)

    if ext in (".h5", ".hdf5", ".rf"):  # .rf is an HDF5 container with a different extension
        if h5py is None:
            raise DasProcessingError(
                "h5py is required to read HDF5 input files"
            ) from _h5py_import_error
        with h5py.File(path, "r") as f:
            if "rf" not in f:
                raise DasProcessingError("HDF5 input must contain an 'rf' dataset")
            return np.array(f["rf"])

    raise DasProcessingError(
        f"Unsupported input format '{ext}'. Expected .h5/.hdf5/.rf/.npy/.npz"
    )


def process_das_task(
    minio_client: Minio,
    bucket: str,
    payload: Dict[str, str],
) -> Dict[str, str]:
    """
    Run Delay-and-Sum beamforming for a job.

    Returns a mapping with uploaded object keys.
    """
    object_key = _extract_input_key(payload)
    snr_db = _parse_float(payload.get("snr")) or 10.0
    nelem = _parse_int(payload.get("nelem"))
    job_id = payload.get("jobId", "unknown-job")
    user_id = payload.get("userId", "unknown-user")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, os.path.basename(object_key))
        _download_input(minio_client, bucket, object_key, input_path)

        rf = _load_rf(input_path)
        nelem = nelem or (rf.shape[1] if rf.ndim == 2 else None)

        if nelem is None:
            raise DasProcessingError("Unable to infer number of elements (nelem) from RF input")

        data = beamforming(
            rf,
            Nelem=nelem,
            SNR_dB=snr_db,
        )

        png_path = os.path.join(tmpdir, "das_result.png")
        save_image(png_path, data)

        # Upload artifacts to MinIO (bucket-relative keys)
        base_prefix = f"{user_id}/{job_id}/output/DAS"
        png_object_key = f"{base_prefix}/result.png"

        minio_client.fput_object(
            bucket,
            png_object_key,
            png_path,
            content_type="image/png",
        )

        return {
            "png": png_object_key,
        }
