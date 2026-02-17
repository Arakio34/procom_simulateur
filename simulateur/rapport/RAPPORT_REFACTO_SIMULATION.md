### 1. Introduction
- Objectif: refacto "library-first" pour isoler le coeur de simulation, rendre la CLI legere, et garantir validation + reproductibilite.
- Perimetre: nouveaux modules `core/` et `cli/`, wrappers dans `main.py`, `simulateur.py`, `parameters.py`, mise a jour `beamforming.py` et `utils.py`, ajout du manifest `run.json`.

### 2. Vue d'ensemble des changements
- Creation d'un package `core/` (Scene, Parameters, RNG, simulation, IO, manifest).
- CLI re-ecrite dans `cli/main.py` avec options ergonomiques et gestion d'erreurs lisible.
- Determinisme: RNG centralise, seed explicite, propagation du seed dans la simulation et les outputs.
- Tracabilite: ajout `run.json` + enrichissement des metadonnees HDF5.
- Nettoyage: suppression des effets de bord (plot debug) dans la simulation.

### 3. Modifications d'architecture
- Fichiers: `core/__init__.py`, `core/simulation.py`, `core/scene.py`, `core/parameters.py`, `core/rng.py`, `core/io.py`, `core/manifest.py`, `core/exceptions.py`
  - Avant: logique metier dispersee entre `main.py`, `simulateur.py`, `utils.py`.
  - Apres: coeur de simulation et IO regroupes dans `core/` avec API explicite.
  - Raison: isoler le coeur utilisable en bibliotheque.
  - Impact logiciel: meilleure maintenabilite, modules testables independamment.
  - Impact UX: previsible, erreurs mieux formatees.
- Fichiers: `main.py`, `cli/main.py`, `cli/__init__.py`
  - Avant: `main.py` contenait toute la logique (lecture scene, loops, outputs).
  - Apres: `main.py` devient un wrapper minimal; la CLI vit dans `cli/main.py`.
  - Raison: separer CLI et coeur, rendre l'API reutilisable.
  - Impact logiciel: reduction du couplage, CLI plus simple a faire evoluer.
  - Impact UX: options plus claires, messages d'erreur propres.
- Fichiers: `simulateur.py`, `parameters.py`, `utils.py`
  - Avant: duplication de logique et validation manuelle.
  - Apres: wrappers vers `core` pour compatibilite.
  - Raison: conserver les points d'entree historiques sans logique lourde.
  - Impact logiciel: transition progressive vers la nouvelle architecture.
  - Impact UX: comportement stable pour l'utilisateur existant.

### 4. Modele de donnees : Scene & Parameters
- Ajouts: classes `Scene`, `Layer`, `Parameters` avec validation stricte.
- Validation: types numeriques, tailles de listes, bornes (z_min/z_max, >0).
- Erreurs utilisateur: messages explicites via `ValidationError` et gestion CLI.
- Exemple avant/apres:
  - Avant: `load_scene_data` acceptait des scenes partielles, validation partielle.
  - Apres: `Scene.from_json_file()` refuse cles inconnues et points mal formes.

### 5. Reproductibilite et determinisme
- Modifie: generation aleatoire centralisee via `core/rng.py`.
- Seed: `--seed` dans la CLI, derive en `seed + index` pour chaque image.
- Garantie: meme seed + memes inputs => scenes, bruit et RF identiques.

### 6. Outputs & tracabilite
- HDF5: `meta` enrichi avec `scene`, `seed`, `run_index`, `parameters`.
- Nouveau `run.json`: manifest avec seed, params, scene (si fournie), liste des outputs, versions logicielles, revision git.
- Benefice: audit complet d'un run et relance exacte a partir du manifest.

### 7. CLI & UX
- Commandes: mode par defaut `simulate` + options explicites.
- Options ajoutees: `--dry-run`, `--preview-scene`, `--seed`, `--scene` alias `--json-file`.
- Experience: erreurs lisibles sans stacktrace; sortie standard plus concise.
- Workflow avant/apres:
  - Avant: `python main.py --json-file sample/scene.json --out data`
  - Apres: `python main.py --scene sample/scene.json --out data --seed 0`

### 8. Tests et validation
- Tests: aucun test automatise ajoute.
- Validation manuelle: simulation via CLI, verification H5 + PNG + `run.json`.

### 9. Limites actuelles
- MVDR conserve sa grille fixe 128x128.
- `model.py` et `model2.py` non refactores (hors perimetre CLI/core).
- Pas de schemas JSON auto-generes.

### 10. Conclusion
- Refacto terminee: coeur isole, CLI propre, determinisme et tracabilite.
- Base plus saine pour integrer des tests, UI, ou benchmarks.
- Prochaines etapes possibles: tests unitaires `core/`, schemas JSON, consolidation MVDR.
