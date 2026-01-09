# Simulateur Echographique V2

Ce projet est un simulateur léger d'imagerie ultrasonore (échographie) codé en Python. Il génère des données brutes RF (Radio-Frequency) et reconstitue des images B-mode.

Le simulateur modélise une émission en Onde Plane (Plane Wave) et permet de simuler des cibles ponctuelles (points brillants) de manière aléatoire ou déterministe. Il supporte deux méthodes de reconstruction d'image : le DAS (Delay-and-Sum) classique et le MVDR (Minimum Variance Distortionless Response).

## Fonctionnalites

- Simulation Physique : Modelisation de la reponse impulsionnelle, delais de vol et attenuation geometrique (1/R).
- Beamforming Flexible :
  - DAS : Reconstruction standard "Delay-and-Sum".
  - MV : Reconstruction adaptative (Capon) pour une meilleure resolution (implementation corrigee avec Diagonal Loading).
- Scenarios Personnalisables : Generation aleatoire de cibles ou chargement precis via un fichier JSON.
- Export de Donnees : Sauvegarde des donnees brutes (RF), traitees (B-mode) et des metadonnees au format .h5 (HDF5).
- Outils Pedagogiques : Scripts d'animation pour visualiser les phenomenes physiques (Huygens, interferences, interpolation).

## Installation

Il est recommande d'utiliser un environnement virtuel pour isoler les dependances.

# Creation et activation de l'environnement virtuel (Linux/macOS)

python -m venv venv
source venv/bin/activate

# Installation des dependances

pip install -r requirements.txt

## Utilisation du Simulateur (main.py)

Le script principal simulateur/main.py permet de lancer la simulation et la reconstruction.

### Ce que fait le simulateur

1. Genere des RF a partir d'une scene (aleatoire ou definie par JSON).
2. Reconstruit une image B-mode (DAS par defaut, MVDR si active).
3. Exporte les resultats en HDF5 et PNG.

### Exemples de commandes

1. Generation aleatoire (DAS standard) :
   Genere 5 images contenant chacune jusqu'a 3 cibles aleatoires.
   python simulateur/main.py --num 5 --maxpoint 3 --out data_random

2. Simulation definie par JSON avec MVDR :
   Simule une scene precise decrite dans un fichier JSON et utilise l'algorithme MVDR.
   python simulateur/main.py --json-file scene.json --mvdr True --out data_json

3. Exemple avec les donnees du repo :
   python simulateur/main.py --num 10 --out simulateur/data

### Arguments disponibles

| Argument    | Type  | Defaut | Description                                                                             |
| ----------- | ----- | ------ | --------------------------------------------------------------------------------------- |
| --config    | str   | None   | Chemin vers un JSON de parametres de simulation (voir `simulateur/parameters.py`).      |
| --json-file | str   | None   | Scene definie par JSON (`points` + optionnel `layers`). Ignore `--num` et `--maxpoint`. |
| --num       | int   | 10     | Nombre d'images en mode aleatoire.                                                      |
| --out       | str   | data   | Dossier racine de sortie (cree `h5/` et `images/`).                                     |
| --snr       | float | 15.0   | Rapport signal/bruit (dB).                                                              |
| --nelem     | int   | 80     | Nombre de capteurs de la sonde.                                                         |
| --mvdr      | flag  | False  | Active le beamforming MVDR au lieu du DAS.                                              |
| --maxpoint  | int   | 3      | Nombre maximum de cibles par image (mode aleatoire).                                    |

### Format du fichier JSON (--json-file)

Le fichier JSON doit contenir une cle principale "points" qui est une liste de listes. Chaque point est defini par [position_x, position_z, amplitude].
Optionnellement, une cle "layers" permet de definir des couches (vitesses et densites) qui modifient les temps de vol.

- x : Position laterale en metres (ex: 0.0 pour le centre).
- z : Profondeur en metres (ex: 0.03 pour 30mm).
- amplitude : Reflectivite de la cible.

Exemple scene.json :
{
"points": [
[-0.005, 0.030, 2.0],
[0.005, 0.040, 5.0]
],
"layers": [
{"name": "Couche 1", "z_min": 0.015, "z_max": 0.020, "c": 1480.0, "rho": 1000.0}
]
}

### Sorties generees

- `out/h5/` : fichiers `.h5` (RF, B-mode, metadonnees).
- `out/images/` : PNG des images reconstruites.

## Animations Pedagogiques (anime/)

Le dossier anime/ contient des scripts autonomes pour visualiser les concepts physiques des ultrasons. Ils generent des fichiers .gif ou affichent des animations.

- huygen.py : Visualisation du principe de Huygens (aller-retour onde plane/onde spherique).
- interference.py : Simulation d'interferences destructives/constructives entre deux sources.
- interpolation.py : Explication visuelle du re-echantillonnage (necessaire pour le beamforming).
- test.py : Animation d'une onde plane rencontrant deux cibles.

Pour lancer une animation :
python anime/huygen.py

## Structure des donnees (.h5)

Les fichiers HDF5 generes contiennent :

- rf : Donnees brutes (forme [Temps, Capteurs]).
- bmode_dB : Image finale en echelle logarithmique.
- y_align : Signaux alignes (apres decalage temporel).
- meta : Dictionnaire contenant les parametres physiques (c, fs, f0, pitch, etc.).

## Modele ABLE (model.py)

Le script `simulateur/model.py` entraine un petit reseau (ABLE) pour apprendre des poids de beamforming a partir des RF, puis applique ces poids pour reconstruire une image.

### Pre-requis donnees

- Le dossier `data/h5/` doit contenir des fichiers `.h5` avec au minimum le dataset `rf`.
- Pour un entrainement de qualite, il est recommande d'utiliser des fichiers contenant `target_rf` (genere par MVDR) afin de fournir une cible utile au modele.

### Entrainement

Genere un modele dans `weight/able_model.pth`.

```
uv run simulateur/model.py --training --data-dir simulateur/data --epochs 50
```

### Inférence (beamforming ABLE)

Charge `weight/able_model.pth` et produit des images PNG dans `data/able_images/`.

```
uv run simulateur/model.py --beamforming --data-dir data
```

## TODO

- Creer une classe contenant tout les parametre de simulation qui sont pour l'instant hard coder dans simulation.
- Pouvoir donner un Json avec tout les parametre de simulation
- Creer une classe Scene contenant tout les element de la scene. => Le simulateur ne prendra plus qu'un objet Scene et Parameter
