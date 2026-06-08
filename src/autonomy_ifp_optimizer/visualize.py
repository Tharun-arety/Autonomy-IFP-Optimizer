from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _mesh_grids(fem: dict[str, object]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    mesh_u, mesh_v = [int(value) for value in fem["mesh_shape"]]
    node_uv = np.asarray(fem["node_uv"], dtype=float)
    u_grid = node_uv[:, 0].reshape(mesh_v + 1, mesh_u + 1)
    v_grid = node_uv[:, 1].reshape(mesh_v + 1, mesh_u + 1)
    return u_grid, v_grid, (mesh_u, mesh_v)


def save_preview(result: dict[str, object], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = result.get("history", [])
    path_uv = np.asarray(result["path_uv"], dtype=float)
    fem = result["fem"]
    radius_profile_mm = 1000.0 * np.asarray(result["radius_profile_m"], dtype=float)
    radius_limit_mm = float(result["metrics"]["radius_limit_mm"])
    element_centers = np.asarray(fem["element_centers_uv"], dtype=float)
    active_elements = np.asarray(fem["active_elements"], dtype=bool)
    von_mises_mpa = 1.0e-6 * np.asarray(fem["element_von_mises_pa"], dtype=float)

    u_grid, v_grid, (mesh_u, mesh_v) = _mesh_grids(fem)
    displacement_mm = 1000.0 * np.asarray(fem["node_displacement_magnitude_m"], dtype=float).reshape(mesh_v + 1, mesh_u + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    displacement_plot = axes[0, 0].pcolormesh(u_grid, v_grid, displacement_mm, shading="auto", cmap="viridis")
    axes[0, 0].plot(path_uv[:, 0], path_uv[:, 1], color="#f8fafc", linewidth=2.2, label="Optimized path")
    for zone in result["surface"].get("keep_outs", []):
        circle = plt.Circle(zone["center_uv"], zone["radius_uv"], color="#86efac", fill=False, linewidth=2.0)
        axes[0, 0].add_patch(circle)
    axes[0, 0].set_title("FEM displacement magnitude")
    axes[0, 0].set_xlabel("u")
    axes[0, 0].set_ylabel("v")
    axes[0, 0].legend(loc="lower right")
    fig.colorbar(displacement_plot, ax=axes[0, 0], fraction=0.046, pad=0.04, label="mm")

    stress_plot = axes[0, 1].scatter(
        element_centers[active_elements, 0],
        element_centers[active_elements, 1],
        c=von_mises_mpa[active_elements],
        cmap="magma",
        s=130,
        marker="s",
    )
    axes[0, 1].plot(path_uv[:, 0], path_uv[:, 1], color="#dbeafe", linewidth=2.0)
    for zone in result["surface"].get("keep_outs", []):
        circle = plt.Circle(zone["center_uv"], zone["radius_uv"], color="#86efac", fill=False, linewidth=2.0)
        axes[0, 1].add_patch(circle)
    axes[0, 1].set_title("Element von Mises stress")
    axes[0, 1].set_xlabel("u")
    axes[0, 1].set_ylabel("v")
    fig.colorbar(stress_plot, ax=axes[0, 1], fraction=0.046, pad=0.04, label="MPa")

    if history:
        steps = [entry["step"] for entry in history]
        axes[1, 0].plot(steps, [entry["loss"] for entry in history], label="Loss", color="#fb6a4a", linewidth=2.0)
        axes[1, 0].plot(
            steps,
            [entry["normalized_compliance"] for entry in history],
            label="Normalized compliance",
            color="#1d4ed8",
            linewidth=1.8,
        )
        axes[1, 0].plot(
            steps,
            [1000.0 * entry["maximum_displacement_m"] for entry in history],
            label="Max displacement [mm]",
            color="#10b981",
            linewidth=1.8,
        )
    axes[1, 0].set_title("Optimization history")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Objective / response")
    axes[1, 0].legend(loc="upper right")

    axes[1, 1].plot(radius_profile_mm, color="#80d0ff", linewidth=2.0, label="Local steering radius")
    axes[1, 1].axhline(radius_limit_mm, color="#ff7f50", linestyle="--", linewidth=1.5, label="Manufacturing limit")
    axes[1, 1].set_title("Manufacturability check")
    axes[1, 1].set_xlabel("Path sample")
    axes[1, 1].set_ylabel("Radius [mm]")
    axes[1, 1].legend(loc="upper right")

    for ax in axes.flat:
        ax.set_xlim(0.0, 1.0) if ax in (axes[0, 0], axes[0, 1]) else None
        ax.set_ylim(0.0, 1.0) if ax in (axes[0, 0], axes[0, 1]) else None

    fig.tight_layout()
    output_path = output_dir / "ifp_preview.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
