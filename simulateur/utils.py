import argparse
import json
import os

from core.scene import Scene
from core.exceptions import ValidationError




def load_scene_data(filepath):
    """
    Fonction de type personnalisée pour argparser.
    Charge un fichier JSON, vérifie s'il contient la clé 'points',
    et retourne les données.
    """
    if not os.path.exists(filepath):
        raise argparse.ArgumentTypeError(f"Le fichier spécifié n'existe pas : '{filepath}'")

    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        scene = Scene.from_json(data)
        scene.validate()
        return scene.to_dict()
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(
            f"Erreur de décodage JSON dans le fichier : '{filepath}'"
        )
    except ValidationError as exc:
        raise argparse.ArgumentTypeError(str(exc))
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Erreur lors de la lecture du fichier : {exc}")

def save_image(save_png_path, data):
    # ============================
    # Affichage / sauvegarde PNG
    # ============================
    from core.io import save_image as core_save_image
    core_save_image(save_png_path, data)


def save_h5(path, data):
    """
    Sauvegarde un dictionnaire Python dans un fichier HDF5 générique.
    Gère les scalaires, arrays, et dictionnaires imbriqués.
    """

    from core.io import save_h5 as core_save_h5
    core_save_h5(path, data)
