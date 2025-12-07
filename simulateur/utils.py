import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.linalg import inv, pinv
from scipy.io import savemat
from scipy.signal.windows import hann
import argparse
import os

def save_image(save_png_path,data):
    # ============================
    # Affichage / sauvegarde PNG
    # ============================
    plt.figure()
    plt.imshow(
        data["bmode_dB"],
        extent=[data["x_img"][0] * 1e3, data["x_img"][-1] * 1e3, data["z_img"][-1] * 1e3, data["z_img"][0] * 1e3],
        cmap='gray',
        aspect='equal'
    )
    plt.clim(-60, 0)
    plt.xlabel('x [mm]')
    plt.ylabel('z [mm]')
    plt.title('B-mode (DAS, dB)')
    plt.colorbar(label='dB')

    plt.savefig(save_png_path, dpi=300, bbox_inches='tight')

    plt.close()


def save_h5(path, data):
    """
    Sauvegarde un dictionnaire Python dans un fichier HDF5 générique.
    Gère les scalaires, arrays, et dictionnaires imbriqués.
    """

    def write_group(h5group, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                subgroup = h5group.create_group(key)
                write_group(subgroup, value)
            else:
                h5group.create_dataset(key, data=value)

    with h5py.File(path, "w") as f:
        write_group(f, data)

    print(f"[OK] Saved HDF5 → {path}")
