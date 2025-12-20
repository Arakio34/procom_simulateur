# Simulateur Echographique B-Mode (Python)

Ce projet est un simulateur léger d'imagerie ultrasonore (échographie) codé en Python. Il génère des données brutes RF (Radio-Frequency) et reconstitue des images B-mode.

Le simulateur modélise une émission en Onde Plane (Plane Wave) et permet de simuler des cibles ponctuelles (points brillants) de manière aléatoire ou déterministe. Il supporte deux méthodes de reconstruction d'image : le DAS (Delay-and-Sum) classique et le MVDR (Minimum Variance Distortionless Response).

## Fonctionnalites

- Simulation Physique : Modelisation de la reponse impulsionnelle, delais de vol et attenuation geometrique (1/R).
- Beamforming Flexible :
  - DAS : Reconstruction standard "Delay-and-Sum".
  - MVDR : Reconstruction adaptative (Capon) pour une meilleure resolution (implementation corrigee avec Diagonal Loading).
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

### Exemples de commandes

1. Generation aleatoire (DAS standard) :
Genere 5 images contenant chacune jusqu'a 3 cibles aleatoires.
python simulateur/main.py --num 5 --maxpoint 3 --out data_random

2. Simulation definie par JSON avec MVDR :
Simule une scene precise decrite dans un fichier JSON et utilise l'algorithme MVDR.
python simulateur/main.py --json-file scene.json --mvdr True --out data_json

### Arguments disponibles

--num (int, Defaut: 10) : Nombre d'images a generer (si aucun fichier JSON n'est fourni).
--json-file (str, Defaut: None) : Chemin vers un fichier JSON definissant les cibles (voir format ci-dessous). Ignore --num et --maxpoint si utilise.
--out (str, Defaut: data) : Dossier racine de sortie (cree automatiquement les sous-dossiers /h5 et /images).
--snr (float, Defaut: 15.0) : Rapport Signal/Bruit (SNR) desire en decibels (dB).
--nelem (int, Defaut: 80) : Nombre de capteurs de la sonde.
--mvdr (bool, Defaut: False) : Active le beamforming MVDR (Capon) au lieu du DAS standard.
--maxpoint (int, Defaut: 3) : Nombre maximum de cibles par image (en mode generation aleatoire).

### Format du fichier JSON (--json-file)

Le fichier JSON doit contenir une cle principale "points" qui est une liste de listes. Chaque point est defini par [position_x, position_z, amplitude].

- x : Position laterale en metres (ex: 0.0 pour le centre).
- z : Profondeur en metres (ex: 0.03 pour 30mm).
- amplitude : Reflectivite de la cible.

Exemple scene.json :
{
    "points": [
        [-0.005, 0.030, 2.0],
        [0.005, 0.040, 5.0]
    ]
}

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
