from __future__ import annotations

from io import BytesIO
import os
import tempfile
from typing import Optional, Tuple

import numpy as np

from exceptions import ProcessingError

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    h5py = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rf_from_path(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".npy", ".npz"}:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if "rf" not in loaded:
                raise ProcessingError("NPZ input must contain an 'rf' array")
            return np.array(loaded["rf"])
        return np.array(loaded)

    if ext in {".h5", ".hdf5", ".rf"}:
        if h5py is None:
            raise ProcessingError("h5py is required to read HDF5 input files")
        try:
            with h5py.File(path, "r") as h5f:
                if "rf" not in h5f:
                    raise ProcessingError("HDF5 input must contain an 'rf' dataset")
                return np.array(h5f["rf"])
        except OSError as exc:
            raise ProcessingError(f"Unable to read HDF5 input: {exc}") from exc

    raise ProcessingError(
        "Unsupported input format. Expected .h5/.hdf5/.rf/.npy/.npz"
    )


def load_rf_from_bytes(data: bytes, source_key: Optional[str]) -> np.ndarray:
    if not data:
        raise ProcessingError("Input payload is empty")

    ext = ""
    if source_key and "." in source_key:
        ext = source_key.rsplit(".", 1)[-1].lower()

    suffix = f".{ext}" if ext else ""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        return load_rf_from_path(tmp.name)


def normalize_rf(rf: np.ndarray, nelem: Optional[int]) -> Tuple[np.ndarray, int]:
    if rf.ndim != 2:
        raise ProcessingError("RF input must be a 2D array")

    inferred = nelem or rf.shape[1]
    if rf.shape[1] == inferred:
        return rf, inferred
    if rf.shape[0] == inferred:
        return rf.T, inferred
    raise ProcessingError("Unable to infer number of elements (nelem) from RF input")


def render_bmode_png(data: dict) -> bytes:
    fig = plt.figure()
    plt.imshow(
        data["bmode_dB"],
        extent=[
            data["x_img"][0] * 1e3,
            data["x_img"][-1] * 1e3,
            data["z_img"][-1] * 1e3,
            data["z_img"][0] * 1e3,
        ],
        cmap="gray",
        aspect="equal",
    )
    plt.clim(-60, 0)
    plt.xlabel("x [mm]")
    plt.ylabel("z [mm]")
    plt.title("B-mode (dB)")
    plt.colorbar(label="dB")

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()
