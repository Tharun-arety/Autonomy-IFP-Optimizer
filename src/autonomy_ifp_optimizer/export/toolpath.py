from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ExportConfig


def _normalize(vector: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, eps)


def load_optimized_path(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _kinematics_arrays(result: dict[str, Any]) -> dict[str, np.ndarray]:
    xyz = np.asarray(result["path_xyz"], dtype=float)
    normals = _normalize(np.asarray(result["normals"], dtype=float))
    tangents = _normalize(np.asarray(result["tangents"], dtype=float))
    binormals = _normalize(np.cross(normals, tangents))
    radius = np.asarray(result["radius_profile_m"], dtype=float)
    segments = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(segments)])
    return {
        "xyz": xyz,
        "normals": normals,
        "tangents": tangents,
        "binormals": binormals,
        "radius": radius,
        "arclength": arclength,
    }


def compute_metrics(result: dict[str, Any], export_config: ExportConfig | None = None) -> dict[str, Any]:
    export_config = export_config or ExportConfig()
    arrays = _kinematics_arrays(result)
    base_metrics = dict(result.get("metrics", {}))
    path_length = float(arrays["arclength"][-1])
    material_weight_kg = path_length * export_config.roving_linear_density_kg_per_m * float(base_metrics.get("thickness_scale", 1.0))
    cycle_time_s = path_length / export_config.placement_speed_mps
    base_metrics.update(
        {
            "placement_speed_mps": export_config.placement_speed_mps,
            "estimated_cycle_time_s": cycle_time_s,
            "estimated_cycle_time_min": cycle_time_s / 60.0,
            "estimated_material_weight_kg": material_weight_kg,
            "estimated_material_weight_g": 1000.0 * material_weight_kg,
            "toolpath_points": int(arrays["xyz"].shape[0]),
        }
    )
    return base_metrics


def write_optimized_path(result: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "optimized_path.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output_path


def write_metrics(metrics: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path


def export_kinematics(
    result: dict[str, Any] | str | Path,
    output_dir: str | Path,
    fmt: str = "json",
) -> Path:
    if isinstance(result, (str, Path)):
        result = load_optimized_path(result)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _kinematics_arrays(result)

    records = []
    for idx in range(arrays["xyz"].shape[0]):
        records.append(
            {
                "index": idx,
                "s_m": float(arrays["arclength"][idx]),
                "x_m": float(arrays["xyz"][idx, 0]),
                "y_m": float(arrays["xyz"][idx, 1]),
                "z_m": float(arrays["xyz"][idx, 2]),
                "nx": float(arrays["normals"][idx, 0]),
                "ny": float(arrays["normals"][idx, 1]),
                "nz": float(arrays["normals"][idx, 2]),
                "tx": float(arrays["tangents"][idx, 0]),
                "ty": float(arrays["tangents"][idx, 1]),
                "tz": float(arrays["tangents"][idx, 2]),
                "bx": float(arrays["binormals"][idx, 0]),
                "by": float(arrays["binormals"][idx, 1]),
                "bz": float(arrays["binormals"][idx, 2]),
                "local_radius_m": float(arrays["radius"][idx]),
            }
        )

    if fmt == "csv":
        output_path = output_dir / "ifp_kinematics.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        return output_path

    output_path = output_dir / "ifp_kinematics.json"
    payload = {
        "surface": result.get("surface", {}),
        "metrics": result.get("metrics", {}),
        "records": records,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
