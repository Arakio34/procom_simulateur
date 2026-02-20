# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the primary CLI entry point for running simulations and beamforming.
- Core simulation logic lives in `simulateur.py`, with helpers in `utils.py`.
- Beamforming implementations are in `beamforming.py`; model variants are in `model.py` and `model2.py`.
- Configuration examples live in `parameters/` and JSON scenes in `sample/` (e.g., `sample/scene.json`).
- Generated outputs typically go to `data/` (H5) and `data/images/` (PNG); sample outputs live in `exemple/`.

## Build, Test, and Development Commands
- Run a basic simulation with a JSON scene:
  - `python main.py --json-file sample/scene.json --out data`
- Run with explicit parameters:
  - `python main.py --config parameters/parameters1.json --out data`
- Enable MVDR beamforming:
  - `python main.py --json-file sample/scene.json --mvdr`
- There is no build step; ensure Python deps like `numpy`, `scipy`, `h5py`, and `matplotlib` are installed.

## Coding Style & Naming Conventions
- Use 4-space indentation and PEP 8 style for Python.
- Prefer `snake_case` for functions/variables and `UpperCamelCase` for classes.
- Keep filenames lowercase (e.g., `beamforming.py`, `simulateur.py`).
- No formatter or linter is configured in this repo; keep changes minimal and consistent.

## Testing Guidelines
- No automated tests are present. Validate changes by running `main.py` and verifying:
  - H5 output in `data/h5/`
  - Images in `data/images/`
  - Console output matches the chosen mode (JSON vs random).
- If you add tests, place them under a new `tests/` directory and name files `test_*.py`.

## Commit & Pull Request Guidelines
- Existing commits use a short, conventional-ish style (e.g., `feat(model): ajout ...`).
- Use concise messages with optional scope: `type(scope): short description`.
- For PRs, include:
  - A brief description of changes and why.
  - Sample command(s) to reproduce results.
  - Screenshots or example PNGs if output images change.

## Configuration & Data Tips
- Use `parameters/parameters1.json` as a starting point for tuning simulation parameters.
- Keep large generated outputs out of the repo unless explicitly requested.
