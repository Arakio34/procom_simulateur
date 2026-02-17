import argparse
import os
import sys

from beamforming import beamforming, mvdr_beamforming
from core import (
    Parameters,
    Scene,
    ValidationError,
    build_run_manifest,
    ensure_seed,
    prepare_output_dirs,
    save_h5,
    save_image,
    spawn_rngs,
    write_json,
)
from core.simulation import simulate_us_scene


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Simulateur Echographique B-Mode (refacto)"
    )
    subparsers = parser.add_subparsers(dest="command")

    sim = subparsers.add_parser("simulate", help="Run a simulation")
    sim.add_argument("--config", type=str, help="Path to parameters JSON")
    sim.add_argument(
        "--scene",
        "--json-file",
        dest="scene_path",
        type=str,
        help="Path to scene JSON (points/layers)",
    )
    sim.add_argument(
        "--num",
        type=int,
        default=None,
        help="Number of images (ignored if scene is provided unless forced).",
    )
    sim.add_argument("--out", type=str, default="data", help="Output directory")
    sim.add_argument("--snr", type=float, default=None, help="SNR in dB")
    sim.add_argument("--nelem", type=int, default=None, help="Number of elements")
    sim.add_argument("--seed", type=int, default=0, help="Base seed")
    sim.add_argument("--mvdr", action="store_true", help="Enable MVDR")
    sim.add_argument(
        "--max-point",
        type=int,
        default=3,
        help="Max points per random scene",
    )
    sim.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without running simulation",
    )
    sim.add_argument(
        "--preview-scene",
        action="store_true",
        help="Print scene summary (requires --scene)",
    )

    return parser


def _parse_args(argv):
    parser = _build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "simulate":
        return parser.parse_args(argv)
    return parser.parse_args(["simulate"] + argv)


def _load_parameters(path, snr, nelem):
    if path:
        try:
            params = Parameters.from_json_file(path)
        except (OSError, ValueError) as exc:
            raise ValidationError(f"Invalid parameters file: {exc}")
    else:
        params = Parameters()
    params = params.with_overrides(SNR_dB=snr, Nelem=nelem)
    params.validate()
    return params


def _load_scene(path):
    if not path:
        return None
    try:
        scene = Scene.from_json_file(path)
    except (OSError, ValueError) as exc:
        raise ValidationError(f"Invalid scene file: {exc}")
    scene.validate()
    return scene


def _build_random_scene(params, rng, max_point):
    x_range = (-params.x_span / 2, params.x_span / 2)
    z_range = (params.z_min, params.z_max)
    scene = Scene.random(rng, max_point, x_range, z_range)
    scene.validate()
    return scene


def _run_simulation(args):
    params = _load_parameters(args.config, args.snr, args.nelem)
    scene = _load_scene(args.scene_path)
    base_seed = ensure_seed(args.seed)

    if args.preview_scene:
        if scene is None:
            raise ValidationError("--preview-scene requires --scene")
        print(scene.summary())

    if scene is not None and args.num is None:
        num_images = 1
    elif scene is None and args.num is None:
        num_images = 10
    else:
        num_images = args.num

    if num_images <= 0:
        raise ValidationError("--num must be > 0")

    if args.dry_run:
        mode = "mvdr" if args.mvdr else "das"
        print(f"Dry run: {num_images} image(s), mode={mode}, seed={base_seed}")
        return

    h5_dir, img_dir = prepare_output_dirs(args.out)

    outputs = []
    rngs = spawn_rngs(base_seed, num_images)
    for index, (seed, rng) in enumerate(rngs):
        if scene is None:
            scene_run = _build_random_scene(params, rng, args.max_point)
        else:
            scene_run = scene

        rf = simulate_us_scene(
            params,
            scene_run,
            rng,
            snr_db=params.SNR_dB,
            nelem=params.Nelem,
        )

        if args.mvdr:
            data = mvdr_beamforming(params, rf, nelem=params.Nelem, snr_db=params.SNR_dB)
            filename_base = f"sample_{index:04d}_mvdr"
        else:
            data = beamforming(params, rf, nelem=params.Nelem, snr_db=params.SNR_dB)
            filename_base = f"sample_{index:04d}"

        data["meta"]["scene"] = scene_run.to_dict()
        data["meta"]["seed"] = seed
        data["meta"]["run_index"] = index
        data["meta"]["parameters"] = params.to_dict()

        h5_path = os.path.join(h5_dir, f"{filename_base}.h5")
        png_path = os.path.join(img_dir, f"{filename_base}.png")
        save_h5(h5_path, data)
        save_image(png_path, data)
        outputs.append(
            {
                "h5": h5_path,
                "png": png_path,
                "index": index,
                "seed": seed,
                "scene": scene_run.to_dict(),
            }
        )

        print(f"Saved: {h5_path}")

    manifest = build_run_manifest(
        params=params,
        scene=scene,
        seed=base_seed,
        num_images=num_images,
        beamforming_mode="mvdr" if args.mvdr else "das",
        outputs=outputs,
        cli_args=vars(args),
        repo_root=os.getcwd(),
    )
    write_json(os.path.join(args.out, "run.json"), manifest)


def main(argv=None):
    try:
        args = _parse_args(argv)
        if args.command != "simulate":
            raise ValidationError("Unknown command.")
        _run_simulation(args)
        return 0
    except ValidationError as exc:
        print(f"Error: {exc}")
        return 2
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
