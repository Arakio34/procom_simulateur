import os
import sys
from datetime import datetime, timezone

import numpy as np


def _read_git_revision(repo_root):
    head_path = os.path.join(repo_root, ".git", "HEAD")
    if not os.path.exists(head_path):
        return None
    with open(head_path, "r", encoding="utf-8") as handle:
        ref = handle.read().strip()
    if ref.startswith("ref: "):
        ref_path = os.path.join(repo_root, ".git", ref.split(" ", 1)[1])
        if os.path.exists(ref_path):
            with open(ref_path, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        return None
    return ref or None


def build_run_manifest(
    params,
    scene,
    seed,
    num_images,
    beamforming_mode,
    outputs,
    cli_args=None,
    repo_root=None,
):
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "created_at": created_at,
        "seed": seed,
        "num_images": num_images,
        "beamforming": beamforming_mode,
        "parameters": params.to_dict(),
        "scene": scene.to_dict() if scene is not None else None,
        "outputs": outputs,
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }
    if cli_args is not None:
        manifest["cli"] = cli_args
    if repo_root:
        manifest["git_revision"] = _read_git_revision(repo_root)
    return manifest
