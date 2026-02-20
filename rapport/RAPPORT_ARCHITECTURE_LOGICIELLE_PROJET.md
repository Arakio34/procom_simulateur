# Compte rendu d'architecture logicielle du projet

Date: 2026-02-20

## 1. Objectif du projet

Le projet fournit un simulateur echographique B-mode en ligne de commande.
Il permet de:
- generer des signaux RF simules a partir d'une scene (points + couches),
- reconstruire une image B-mode via beamforming DAS ou MVDR,
- sauvegarder les resultats (HDF5, PNG) et un manifest de run (`run.json`).

Le code est organise pour separer:
- l'orchestration CLI,
- le coeur metier (validation, simulation, beamforming, IO),
- les assets de configuration (scenes JSON, parametres JSON).

## 2. Vue d'ensemble de l'architecture

Architecture actuelle (post-refacto):

- `simulateur/main.py`
  - Point d'entree minimal. Delegue a la CLI.
- `simulateur/cli/main.py`
  - Parsing des arguments, orchestration du run, gestion d'erreurs utilisateur.
- `simulateur/core/`
  - Logique metier decoupee en modules:
    - `parameters.py` (modele + validation des parametres),
    - `scene.py` (modele + validation des scenes),
    - `simulation.py` (generation RF),
    - `beamforming.py` (DAS et MVDR),
    - `io.py` (HDF5/PNG/JSON),
    - `rng.py` (determinisme via seeds),
    - `manifest.py` (metadonnees d'execution),
    - `exceptions.py` (`ValidationError`).

Le module `simulateur/core/__init__.py` expose une API de facade pour importer les composants principaux depuis un seul point.

## 3. Arborescence fonctionnelle (dossier `simulateur/`)

- `main.py`: wrapper d'entree.
- `cli/`
  - `main.py`: commande `simulate` (implicite), options CLI.
- `core/`
  - `__init__.py`: exports unifies.
  - `exceptions.py`: erreur de validation.
  - `parameters.py`: dataclass `Parameters`.
  - `scene.py`: dataclasses `Scene` et `Layer`.
  - `rng.py`: gestion des seeds (`ensure_seed`, `spawn_rngs`).
  - `simulation.py`: `simulate_us_scene(...)`.
  - `beamforming.py`: `beamforming(...)` et `mvdr_beamforming(...)`.
  - `io.py`: `prepare_output_dirs`, `save_h5`, `save_image`, `write_json`.
  - `manifest.py`: `build_run_manifest(...)`.
- `scene/`: jeux de scenes JSON.
- `parameters/`: fichiers de parametres JSON.
- `weight/`: artefacts de modele (historique/IA).

## 4. Flux d'execution complet

### 4.1 Entree CLI

Commande type:
- `python main.py --scene scene/scene.json --out data`

`main.py` appelle `cli.main.main()`.

### 4.2 Parsing et validation

Dans `cli/main.py`:
- construit un parser avec sous-commande `simulate`,
- accepte les options principales: `--scene`, `--config`, `--num`, `--seed`, `--mvdr`, `--snr`, `--nelem`, `--dry-run`, `--preview-scene`, etc.

Chargement des entrees:
- `Parameters.from_json_file()` ou `Parameters()` par defaut,
- `Scene.from_json_file()` si `--scene` est fourni,
- validation stricte via `validate()`.

### 4.3 Determinisme

Le seed de base est normalise avec `ensure_seed(seed)`.
Pour `N` images, `spawn_rngs(base_seed, N)` cree des RNG derives:
- image 0 -> `base_seed + 0`,
- image 1 -> `base_seed + 1`, etc.

Ce design garantit la reproductibilite run-a-run.

### 4.4 Simulation physique (RF)

Pour chaque image:
- si pas de scene fournie, generation d'une scene aleatoire (`Scene.random(...)`),
- generation du RF via `simulate_us_scene(params, scene_run, rng, ...)`.

`simulate_us_scene`:
- calcule la geometrie des elements (`x_el`),
- cree une impulsion emise bande-limitee,
- calcule les temps de vol tx/rx pour les diffuseurs,
- prend en compte les couches (vitesses/densites, reflexions/attenuation),
- somme les contributions et ajoute un bruit selon `SNR_dB`.

Sortie: matrice RF `(Nt, Nelem)`.

### 4.5 Reconstruction d'image

Deux modes:
- DAS (`beamforming`): retard-somme + apodisation Hann + enveloppe (Hilbert) + passage en dB,
- MVDR (`mvdr_beamforming`): alignement complexe + covariance + diagonal loading + Capon weights.

Le resultat est un dictionnaire de donnees contenant notamment:
- `rf`, `x_img`, `z_img`, `bmode_dB`, `env`, `meta`.

### 4.6 Persistance des sorties

`io.prepare_output_dirs(out)` cree:
- `out/h5/`
- `out/images/`

Pour chaque image:
- `io.save_h5(...)` persiste la structure complete,
- `io.save_image(...)` exporte un PNG de l'image B-mode.

En fin de run:
- `manifest.build_run_manifest(...)` construit le `run.json`,
- `io.write_json(...)` l'ecrit sur disque.

## 5. Contrats de donnees

### 5.1 Scene JSON

Format attendu:
- objet avec cles `points` et optionnellement `layers`.

`points`:
- liste de triplets `[x, z, amplitude]`.

`layers`:
- liste d'objets avec cles obligatoires:
  - `z_min`, `z_max`, `c`, `rho`
- cle optionnelle: `name`.

Validation forte:
- rejet des cles inconnues,
- types numeriques imposes,
- bornes physiques elementaires (ex: `z_max > z_min`).

### 5.2 Parametres JSON

Format mappe sur la dataclass `Parameters`:
- `c`, `f0`, `fs`, `fracBW`, `nCycles`, `Nelem`, `pitch`,
- `x_span`, `z_min`, `z_max`, `Nx`, `Nz`, `SNR_dB`, `p`.

Regles:
- cles inconnues refusees,
- `Nelem`, `Nx`, `Nz` castes en int,
- verification des bornes (strictement positives, coherence z).

### 5.3 HDF5 de sortie

Structure depend du mode, mais contient typiquement:
- `rf`: signal brut,
- `x_img`, `z_img`:
- `bmode_dB`, `env`,
- `meta/`:
  - metadonnees physiques,
  - scene, seed, index de run,
  - parametres utilises.

### 5.4 Manifest `run.json`

Inclut:
- date UTC de creation,
- seed de base,
- nombre d'images,
- mode de beamforming,
- parametres,
- scene (si fournie),
- liste des sorties generees,
- versions logicielles (Python, NumPy),
- revision git si disponible.

## 6. Gestion des erreurs

Strategie:
- les erreurs metier de validation passent par `ValidationError`,
- la CLI capte les erreurs et affiche un message utilisateur sans stacktrace,
- code de retour non-zero en cas d'erreur (`2`).

Implication:
- comportement robuste en usage CLI,
- feedback lisible pour les utilisateurs non experts Python.

## 7. Decisions d'architecture importantes

- Separation nette CLI / coeur metier:
  - facilite tests unitaires futurs,
  - facilite reutilisation du coeur depuis d'autres interfaces.
- Determinisme explicite:
  - essentiel pour reproduire datasets et experiences.
- Validation stricte des entrees:
  - evite des echecs silencieux plus loin dans la chaine.
- Manifest de run:
  - supporte tracabilite et audit d'execution.

## 8. Guide de reprise pour un nouveau developpeur

### 8.1 Demarrage rapide

1. Installer dependances (ex: `uv sync` ou environnement Python equivalent).
2. Tester un dry-run:
   - `cd simulateur && uv run python main.py --scene scene/scene_simple.json --dry-run`
3. Lancer une simulation complete:
   - `cd simulateur && uv run python main.py --scene scene/scene_simple.json --out data --num 1`

### 8.2 Points d'entree a lire en priorite

1. `simulateur/cli/main.py` (orchestration complete).
2. `simulateur/core/simulation.py` (modele physique RF).
3. `simulateur/core/beamforming.py` (reconstruction image).
4. `simulateur/core/io.py` et `simulateur/core/manifest.py` (sorties/tracabilite).

### 8.3 Cas d'evolution frequents

- Ajouter une nouvelle option CLI:
  - parser dans `cli/main.py`,
  - validation dans `_run_simulation`.
- Ajouter un nouveau mode de beamforming:
  - nouvelle fonction dans `core/beamforming.py`,
  - branchement dans `cli/main.py`,
  - enrichissement metadata si necessaire.
- Ajouter un nouveau format de sortie:
  - fonction dans `core/io.py`,
  - branchement apres reconstruction.

### 8.4 Bonnes pratiques recommandees

- conserver les validations en amont,
- eviter la logique metier dans la CLI,
- enrichir `run.json` a chaque nouvelle feature de sortie,
- garder la compatibilite des schemas HDF5/JSON autant que possible.

## 9. Limites connues

- Pas de suite de tests automatises (`tests/`) a ce jour.
- MVDR est couteux (double boucle pixel + inversion de covariance).
- Le titre PNG est fixe (`B-mode (DAS, dB)`) meme en mode MVDR.
- Le README historique mentionne encore des anciens chemins/termes dans certains contextes externes (penser a maintenir la doc synchronisee).

## 10. Recommandations de roadmap

1. Ajouter des tests unitaires pour `core/parameters.py`, `core/scene.py`, `core/rng.py`.
2. Ajouter des tests d'integration CLI minimalistes (dry-run + run 1 image).
3. Centraliser des constantes d'affichage/metadata (ex: titre image par mode).
4. Definir un schema versionne pour les sorties HDF5/manifest.
5. Documenter des profils de performance (DAS vs MVDR) pour guider les usages.

## 11. Statut de licence

Une licence MIT a ete ajoutee a la racine du projet via le fichier `LICENSE` pour autoriser la reutilisation, modification et distribution du code.
