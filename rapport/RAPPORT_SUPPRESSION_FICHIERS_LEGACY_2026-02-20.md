# Rapport de suppression des fichiers legacy

Date: 2026-02-20

## 1. Verification initiale

Lecture realisee:
- `README.md`
- `rapport/RAPPORT_REFACTO_SIMULATION.md`

Test simple avant modification:
- Commande: `cd simulateur && ../.venv/bin/python main.py --scene scene/scene_simple.json --dry-run`
- Resultat: `Dry run: 1 image(s), mode=das, seed=0`

Test execution minimale avant modification:
- Commande: `cd simulateur && ../.venv/bin/python main.py --scene scene/scene_simple.json --out /tmp/procom_precheck --num 1`
- Resultat: fichier H5 genere (`sample_0000.h5`)

## 2. Objectif

Supprimer les fichiers suivants tout en conservant un simulateur fonctionnel:
- `simulateur/beamforming.py`
- `simulateur/model.py`
- `simulateur/model2.py`
- `simulateur/parameters.py`
- `simulateur/simulateur.py`
- `simulateur/utils.py`

## 3. Modifications effectuees

1. Deplacement du beamforming dans `core/`:
- Ajout de `simulateur/core/beamforming.py`
- Fonctions migrees:
  - `beamforming(...)`
  - `mvdr_beamforming(...)`

2. Mise a jour des imports CLI:
- Fichier modifie: `simulateur/cli/main.py`
- Changement:
  - import beamforming depuis `core` au lieu de `beamforming.py` racine

3. Exposition via API `core`:
- Fichier modifie: `simulateur/core/__init__.py`
- Ajout des exports:
  - `beamforming`
  - `mvdr_beamforming`

4. Suppression des fichiers legacy:
- Supprimes:
  - `simulateur/beamforming.py`
  - `simulateur/model.py`
  - `simulateur/model2.py`
  - `simulateur/parameters.py`
  - `simulateur/simulateur.py`
  - `simulateur/utils.py`

5. Mise a jour documentation:
- Fichier modifie: `README.md`
- Ajustements:
  - exemples de chemin scene (`scene/scene.json`)
  - note sur suppression des fichiers legacy

## 4. Verification apres modifications

Test dry-run:
- Commande: `cd simulateur && ../.venv/bin/python main.py --scene scene/scene_simple.json --dry-run`
- Resultat: OK

Test simulation DAS:
- Commande: `cd simulateur && ../.venv/bin/python main.py --scene scene/scene_simple.json --out /tmp/procom_postcheck --num 1`
- Resultat: OK, H5 genere (`sample_0000.h5`)

Test simulation MVDR:
- Commande: `cd simulateur && ../.venv/bin/python main.py --scene scene/scene_simple.json --out /tmp/procom_postcheck_mvdr --num 1 --mvdr`
- Resultat: OK, H5 genere (`sample_0000_mvdr.h5`)

## 5. Conclusion

La suppression des 6 fichiers legacy a ete realisee.
Le simulateur reste fonctionnel en mode DAS et MVDR via la CLI actuelle.
