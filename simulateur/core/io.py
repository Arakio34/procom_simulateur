import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def prepare_output_dirs(root_dir):
    h5_dir = os.path.join(root_dir, "h5")
    img_dir = os.path.join(root_dir, "images")
    os.makedirs(h5_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    return h5_dir, img_dir


def save_image(save_png_path, data):
    plt.figure()
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
    plt.title("B-mode (DAS, dB)")
    plt.colorbar(label="dB")
    plt.savefig(save_png_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_h5(path, data):
    def write_group(h5group, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                subgroup = h5group.create_group(key)
                write_group(subgroup, value)
            elif isinstance(value, (list, tuple)):
                arr = np.asarray(value)
                if arr.dtype != object:
                    h5group.create_dataset(key, data=arr)
                else:
                    subgroup = h5group.create_group(key)
                    for idx, item in enumerate(value):
                        item_key = str(idx)
                        if isinstance(item, dict):
                            item_group = subgroup.create_group(item_key)
                            write_group(item_group, item)
                        else:
                            subgroup.create_dataset(item_key, data=np.asarray(item))
            else:
                h5group.create_dataset(key, data=value)

    with h5py.File(path, "w") as handle:
        write_group(handle, data)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
