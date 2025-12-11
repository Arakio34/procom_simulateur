import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.linalg import inv, pinv
from scipy.io import savemat
from scipy.signal.windows import hann
import argparse
import os

import argparse
import json
import os

def load_json_data(filepath):
    """
    Fonction de type personnalisée pour argparser.
    Charge un fichier JSON, vérifie s'il contient la clé 'points',
    et retourne les données.
    """
    if not os.path.exists(filepath):
        raise argparse.ArgumentTypeError(f"Le fichier spécifié n'existe pas : '{filepath}'")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"Erreur de décodage JSON dans le fichier : '{filepath}'")
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Erreur lors de la lecture du fichier : {e}")

    # --- Validation du contenu ---

    if 'points' not in data:
        raise argparse.ArgumentTypeError(
            f"Le fichier JSON doit contenir la clé principale 'points'."
        )

    points_list = data['points']

    if not isinstance(points_list, list):
        raise argparse.ArgumentTypeError(
            f"La valeur de la clé 'points' doit être une liste (actuellement {type(points_list).__name__})."
        )
    
    # Vous pouvez ajouter une validation plus poussée ici (ex: vérifier que chaque sous-liste a 3 éléments)
    for i, point in enumerate(points_list):
        if not isinstance(point, list) or len(point) != 3:
             raise argparse.ArgumentTypeError(
                f"Le point à l'index {i} ('{point}') n'est pas une liste de 3 éléments. Format attendu : [x, y, z]."
            )
        try:
            # S'assurer que les éléments sont des nombres
            [float(val) for val in point] 
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(
                f"Le point à l'index {i} ('{point}') doit contenir uniquement des valeurs numériques (float/int)."
            )

    # Retourne la liste des points validés
    return points_list

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
