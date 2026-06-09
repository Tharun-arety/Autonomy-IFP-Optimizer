from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ExportConfig
from ..core.geometry import SurfaceDefinition, preview_keepouts, preview_surface, surface_from_dict


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


def _surface_from_result(result: dict[str, Any]) -> SurfaceDefinition:
    surface = result.get("surface", {})
    if isinstance(surface, SurfaceDefinition):
        return surface
    return surface_from_dict(surface)


def _result_like(
    result: dict[str, Any],
    *,
    path_uv: Any,
    path_xyz: Any,
    normals: Any,
    tangents: Any,
    radius_profile_m: Any,
) -> dict[str, Any]:
    return {
        "surface": result.get("surface", {}),
        "optimization_config": result.get("optimization_config", {}),
        "metrics": result.get("metrics", {}),
        "path_uv": path_uv,
        "path_xyz": path_xyz,
        "normals": normals,
        "tangents": tangents,
        "radius_profile_m": radius_profile_m,
    }


def _segment_to_node_values(values: np.ndarray, count: int) -> np.ndarray:
    if count <= 1:
        return np.zeros((count,), dtype=float)
    if values.size == 0:
        return np.zeros((count,), dtype=float)
    if values.size == 1:
        return np.full((count,), float(values[0]), dtype=float)
    middle = 0.5 * (values[:-1] + values[1:])
    return np.concatenate([[values[0]], middle, [values[-1]]])


def _rotation_rate(vectors: np.ndarray, arclength: np.ndarray) -> np.ndarray:
    if vectors.shape[0] <= 1:
        return np.zeros((vectors.shape[0],), dtype=float)
    dots = np.clip(np.sum(vectors[1:] * vectors[:-1], axis=1), -1.0, 1.0)
    angles = np.arccos(dots)
    segment_length = np.maximum(np.diff(arclength), 1.0e-6)
    return _segment_to_node_values(angles / segment_length, vectors.shape[0])


def _keepout_clearance(surface: SurfaceDefinition, path_uv: np.ndarray) -> np.ndarray:
    if not surface.keep_outs:
        return np.full((path_uv.shape[0],), 0.12, dtype=float)
    distances = []
    for zone in surface.keep_outs:
        center = np.asarray(zone.center_uv, dtype=float)
        distances.append(np.linalg.norm(path_uv - center[None, :], axis=1) - float(zone.radius_uv))
    return np.min(np.stack(distances, axis=0), axis=0)


def _display_aspect_ratio(surface: SurfaceDefinition, mesh_xyz: np.ndarray) -> dict[str, float]:
    span = np.maximum(np.ptp(mesh_xyz, axis=0), 1.0e-6)
    major = float(max(span[0], span[1]))
    if surface.kind == "plate_with_hole":
        span[2] = max(float(span[2]), 0.24 * major)
    scale = float(np.max(span))
    return {
        "x": float(span[0] / scale),
        "y": float(span[1] / scale),
        "z": float(span[2] / scale),
    }


def compute_local_routing_effort(result: dict[str, Any] | str | Path) -> np.ndarray:
    if isinstance(result, (str, Path)):
        result = load_optimized_path(result)

    arrays = _kinematics_arrays(result)
    surface = _surface_from_result(result)
    path_uv = np.asarray(result["path_uv"], dtype=float)
    optimization_config = dict(result.get("optimization_config", {}))
    metrics = dict(result.get("metrics", {}))

    radius_limit_m = float(metrics.get("radius_limit_mm", 1000.0 * optimization_config.get("min_steering_radius_m", 0.05))) / 1000.0
    steering_term = np.clip(radius_limit_m / np.maximum(arrays["radius"], 1.0e-6), 0.0, 3.0)

    tool_axis_rate = _rotation_rate(arrays["normals"], arrays["arclength"])
    tangent_rate = _rotation_rate(arrays["tangents"], arrays["arclength"])
    tool_axis_term = np.clip(tool_axis_rate / 10.0, 0.0, 2.0)
    tangent_term = np.clip(tangent_rate / 14.0, 0.0, 2.0)

    keepout_clearance = _keepout_clearance(surface, path_uv)
    keepout_term = np.clip((0.08 - keepout_clearance) / 0.08, 0.0, 2.5)

    boundary_clearance = np.min(
        np.column_stack(
            [
                path_uv[:, 0],
                path_uv[:, 1],
                1.0 - path_uv[:, 0],
                1.0 - path_uv[:, 1],
            ]
        ),
        axis=1,
    )
    boundary_term = np.clip((0.05 - boundary_clearance) / 0.05, 0.0, 2.0)

    return 0.50 * steering_term + 0.20 * tool_axis_term + 0.15 * tangent_term + 0.10 * keepout_term + 0.05 * boundary_term


def compute_metrics(result: dict[str, Any], export_config: ExportConfig | None = None) -> dict[str, Any]:
    export_config = export_config or ExportConfig()
    arrays = _kinematics_arrays(result)
    routing_effort = compute_local_routing_effort(result)
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
            "mean_routing_effort": float(np.mean(routing_effort)),
            "peak_routing_effort": float(np.max(routing_effort)),
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


def write_interactive_toolpath_html(
    result: dict[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    filename: str = "interactive_toolpath.html",
    vector_stride: int = 12,
) -> Path:
    if isinstance(result, (str, Path)):
        result = load_optimized_path(result)

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Plotly is required to export interactive HTML toolpaths.") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    surface = _surface_from_result(result)
    mesh_xyz, faces = preview_surface(surface)
    keepout_curves = preview_keepouts(surface)
    arrays = _kinematics_arrays(result)
    routing_effort = compute_local_routing_effort(result)
    baseline = result.get("baseline")

    span = np.maximum(np.ptp(mesh_xyz, axis=0), 1.0e-6)
    aspect_ratio = _display_aspect_ratio(surface, mesh_xyz)
    vector_scale = 0.10 * float(np.max(span))
    sample_idx = np.arange(0, arrays["xyz"].shape[0], max(vector_stride, 1), dtype=int)
    if sample_idx[-1] != arrays["xyz"].shape[0] - 1:
        sample_idx = np.append(sample_idx, arrays["xyz"].shape[0] - 1)

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=mesh_xyz[:, 0],
            y=mesh_xyz[:, 1],
            z=mesh_xyz[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="#d6d3d1",
            opacity=0.22,
            flatshading=True,
            name="Surface",
            hoverinfo="skip",
        )
    )

    for idx, curve in enumerate(keepout_curves):
        fig.add_trace(
            go.Scatter3d(
                x=curve[:, 0],
                y=curve[:, 1],
                z=curve[:, 2],
                mode="lines",
                line=dict(color="#22c55e", width=5),
                name="Keep-out" if idx == 0 else f"Keep-out {idx + 1}",
                hoverinfo="skip",
            )
        )

    if isinstance(baseline, dict):
        baseline_result = _result_like(
            result,
            path_uv=baseline["path_uv"],
            path_xyz=baseline["path_xyz"],
            normals=baseline["normals"],
            tangents=baseline["tangents"],
            radius_profile_m=baseline["radius_profile_m"],
        )
        baseline_effort = compute_local_routing_effort(baseline_result)
        baseline_xyz = np.asarray(baseline["path_xyz"], dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=baseline_xyz[:, 0],
                y=baseline_xyz[:, 1],
                z=baseline_xyz[:, 2],
                mode="lines+markers",
                line=dict(color="#94a3b8", width=5),
                marker=dict(size=3, color=baseline_effort, colorscale="Turbo", cmin=0.0, cmax=max(1.0, float(np.max(routing_effort)))),
                name="Naive seed path",
                customdata=np.column_stack([baseline_effort]),
                hovertemplate="Naive path<br>effort=%{customdata[0]:.2f}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=arrays["xyz"][:, 0],
            y=arrays["xyz"][:, 1],
            z=arrays["xyz"][:, 2],
            mode="lines+markers",
            line=dict(color="#0f766e", width=7),
            marker=dict(
                size=4,
                color=routing_effort,
                colorscale="Turbo",
                cmin=0.0,
                cmax=max(1.0, float(np.max(routing_effort))),
                colorbar=dict(title="Routing effort"),
            ),
            name="Optimized path",
            customdata=np.column_stack([arrays["arclength"], routing_effort]),
            hovertemplate="Optimized path<br>s=%{customdata[0]:.3f} m<br>effort=%{customdata[1]:.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Cone(
            x=arrays["xyz"][sample_idx, 0],
            y=arrays["xyz"][sample_idx, 1],
            z=arrays["xyz"][sample_idx, 2],
            u=arrays["normals"][sample_idx, 0] * vector_scale,
            v=arrays["normals"][sample_idx, 1] * vector_scale,
            w=arrays["normals"][sample_idx, 2] * vector_scale,
            sizemode="absolute",
            sizeref=vector_scale,
            showscale=False,
            colorscale=[[0.0, "#2563eb"], [1.0, "#2563eb"]],
            opacity=0.72,
            name="Tool axis",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Cone(
            x=arrays["xyz"][sample_idx, 0],
            y=arrays["xyz"][sample_idx, 1],
            z=arrays["xyz"][sample_idx, 2],
            u=arrays["binormals"][sample_idx, 0] * vector_scale,
            v=arrays["binormals"][sample_idx, 1] * vector_scale,
            w=arrays["binormals"][sample_idx, 2] * vector_scale,
            sizemode="absolute",
            sizeref=vector_scale,
            showscale=False,
            colorscale=[[0.0, "#f97316"], [1.0, "#f97316"]],
            opacity=0.48,
            name="Roll axis",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Interactive IFP Toolpath",
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=46),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.74)"),
        scene=dict(
            aspectmode="manual",
            aspectratio=aspect_ratio,
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            camera=dict(eye=dict(x=1.55, y=-1.75, z=0.95)),
        ),
    )

    output_path = output_dir / filename
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    return output_path
