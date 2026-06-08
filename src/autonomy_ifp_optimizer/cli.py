from __future__ import annotations

import argparse
from pathlib import Path

from .ai_surrogate.train_flax_model import train_surrogate
from .config import ExportConfig, GeometryConfig, LoadCase, OptimizationConfig, SurrogateConfig
from .core.geometry import load_surface
from .core.physics import optimize_ifp_path
from .export.toolpath import compute_metrics, export_kinematics, load_optimized_path, write_metrics, write_optimized_path
from .visualize import save_preview


def _parse_direction(text: str) -> tuple[float, float, float]:
    values = [float(part.strip()) for part in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Direction must have three comma-separated values.")
    return values[0], values[1], values[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Differentiable IFP optimizer for autonomous composites workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize = subparsers.add_parser("optimize", help="Optimize a differentiable IFP path on a supported surface.")
    optimize.add_argument("--mesh", default="examples/drone_plate.obj", help="Mesh path or surface alias.")
    optimize.add_argument("--surface", choices=["plate_with_hole", "cylinder"], default=None, help="Override procedural surface type.")
    optimize.add_argument("--load", type=float, default=500.0, help="Load magnitude in Newtons.")
    optimize.add_argument("--direction", type=_parse_direction, default=(1.0, 0.0, 0.0), help="Load direction as x,y,z.")
    optimize.add_argument("--min-radius", type=float, default=50.0, help="Minimum steering radius in millimetres.")
    optimize.add_argument("--max-thickness", type=float, default=1.2, help="Maximum allowable thickness field value.")
    optimize.add_argument("--steps", type=int, default=350, help="Number of Adam optimization steps.")
    optimize.add_argument("--outdir", default="outputs", help="Directory for optimized artifacts.")
    optimize.add_argument("--format", choices=["json", "csv"], default="json", help="Kinematics export format.")

    export = subparsers.add_parser("export", help="Export kinematic arrays from an existing optimized path JSON.")
    export.add_argument("--input", required=True, help="Path to optimized_path.json.")
    export.add_argument("--format", choices=["json", "csv"], default="json", help="Kinematics export format.")
    export.add_argument("--outdir", default="outputs", help="Directory for exported files.")

    train = subparsers.add_parser("train-surrogate", help="Generate a dataset and train a Flax surrogate model.")
    train.add_argument("--samples", type=int, default=512, help="Number of physics samples to generate.")
    train.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    train.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    train.add_argument("--surface", choices=["plate_with_hole", "cylinder"], default="plate_with_hole", help="Surface family for the dataset.")
    train.add_argument("--outdir", default="outputs", help="Directory for surrogate artifacts.")

    return parser


def _run_optimize(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    geometry_config = GeometryConfig(surface=args.surface or "plate_with_hole")
    surface = load_surface(mesh=args.mesh, surface=args.surface, geometry_config=geometry_config)
    load_case = LoadCase(magnitude_n=args.load, direction_xyz=args.direction)
    opt_config = OptimizationConfig(
        steps=args.steps,
        min_steering_radius_m=args.min_radius / 1000.0,
        max_thickness=args.max_thickness,
    )
    result = optimize_ifp_path(surface, load_case=load_case, config=opt_config)

    export_config = ExportConfig(output_dir=outdir)
    result["metrics"] = compute_metrics(result, export_config)
    save_preview(result, outdir)
    write_optimized_path(result, outdir)
    write_metrics(result["metrics"], outdir)
    export_kinematics(result, outdir, fmt=args.format)

    print(f"Optimized surface     : {result['surface']['name']} ({result['surface']['kind']})")
    print(f"Objective             : {result['metrics']['objective']:.4f}")
    print(f"Min steering radius   : {result['metrics']['min_steering_radius_mm']:.2f} mm")
    print(f"Peak thickness        : {result['metrics']['peak_thickness']:.3f}")
    print(f"Estimated cycle time  : {result['metrics']['estimated_cycle_time_s']:.2f} s")
    print(f"Manufacturable        : {result['metrics']['manufacturable']}")
    print(f"Artifacts written to  : {outdir.resolve()}")
    return 0


def _run_export(args: argparse.Namespace) -> int:
    result = load_optimized_path(args.input)
    output = export_kinematics(result, args.outdir, fmt=args.format)
    metrics = compute_metrics(result)
    write_metrics(metrics, args.outdir)
    print(f"Kinematics exported to {output}")
    return 0


def _run_train(args: argparse.Namespace) -> int:
    config = SurrogateConfig(
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    artifacts = train_surrogate(surface_name=args.surface, config=config, output_dir=args.outdir)
    print(f"Dataset written to    : {artifacts['dataset_path']}")
    print(f"Metrics written to    : {artifacts['metrics_path']}")
    print(f"Model parameters      : {artifacts['params_path']}")
    print(f"Validation RMSE       : {artifacts['validation_rmse']:.4f}")
    print(f"Inference latency     : {artifacts['inference_latency_ms']:.3f} ms")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "optimize":
        return _run_optimize(args)
    if args.command == "export":
        return _run_export(args)
    if args.command == "train-surrogate":
        return _run_train(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2
