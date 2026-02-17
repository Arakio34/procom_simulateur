# Simulateur echographique (CLI)

Ce depot expose un simulateur B-mode avec une CLI legere. Le coeur est dans `core/`, la CLI dans `cli/`, et `main.py` sert de point d'entree.

## Prerequis
- Python 3
- Dependances: `numpy`, `scipy`, `h5py`, `matplotlib`

## Usage rapide
Exemples:

- Simulation a partir d'une scene JSON:
  - `python main.py --scene sample/scene.json --out data`
- Simulation avec parametres explicites:
  - `python main.py --config parameters/parameters1.json --out data`
- Activer MVDR:
  - `python main.py --scene sample/scene.json --mvdr`

La commande `simulate` est implicite. Vous pouvez aussi l'ecrire explicitement:
- `python main.py simulate --scene sample/scene.json --out data`

## Options principales
- `--scene` / `--json-file`: chemin vers une scene JSON (points/couches).
- `--config`: chemin vers un fichier de parametres JSON.
- `--num`: nombre d'images a generer.
  - Si `--scene` est fourni et `--num` absent: 1 image.
  - Si `--scene` absent et `--num` absent: 10 images.
- `--seed`: seed de base (deterministe, default: 0).
- `--mvdr`: active le beamforming MVDR (sinon DAS).
- `--out`: dossier de sortie (default: `data`).
- `--snr`: surcharge SNR en dB.
- `--nelem`: surcharge du nombre d'elements.
- `--max-point`: max de points pour scene aleatoire (default: 3).
- `--dry-run`: valide les inputs sans executer la simulation.
- `--preview-scene`: affiche un resume de scene (requiert `--scene`).

## Sorties
Avec `--out data`, la sortie est organisee ainsi:
- `data/h5/` : fichiers HDF5 par image
- `data/images/` : PNG B-mode associes
- `data/run.json` : manifest du run (seed, parametres, scene, outputs, versions)

## Determinisme
Le simulateur est deterministe: a scene/parametres identiques et seed identique, les sorties sont reproduites. Le seed de base est derive par image pour chaque index.

## Exemples utiles
- Generer 5 images aleatoires, seed 42:
  - `python main.py --num 5 --seed 42 --out data`
- Valider un fichier de scene sans lancer la simu:
  - `python main.py --scene sample/scene.json --dry-run`
- Afficher le resume d'une scene:
  - `python main.py --scene sample/scene.json --preview-scene`

## Notes
- `model.py` et `model2.py` ne font pas partie de la CLI.
- Les erreurs de validation sont affichees proprement sans stacktrace.
