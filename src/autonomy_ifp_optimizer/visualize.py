from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def _uv_triangulation(fem: dict[str, object]) -> tuple[np.ndarray, mtri.Triangulation]:
    node_uv = np.asarray(fem["node_uv"], dtype=float)
    element_nodes = np.asarray(fem["element_nodes"], dtype=np.int32)
    triangulation = mtri.Triangulation(node_uv[:, 0], node_uv[:, 1], element_nodes)
    active_elements = np.asarray(fem["active_elements"], dtype=bool)
    if active_elements.shape[0] == element_nodes.shape[0]:
        triangulation.set_mask(~active_elements)
    return node_uv, triangulation


def save_preview(result: dict[str, object], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = result.get("history", [])
    path_uv = np.asarray(result["path_uv"], dtype=float)
    fem = result["fem"]
    radius_profile_mm = 1000.0 * np.asarray(result["radius_profile_m"], dtype=float)
    radius_limit_mm = float(result["metrics"]["radius_limit_mm"])
    von_mises_mpa = 1.0e-6 * np.asarray(fem["element_von_mises_pa"], dtype=float)
    displacement_mm = 1000.0 * np.asarray(fem["node_displacement_magnitude_m"], dtype=float)

    _, triangulation = _uv_triangulation(fem)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))

    displacement_plot = axes[0, 0].tripcolor(triangulation, displacement_mm, shading="gouraud", cmap="viridis")
    axes[0, 0].triplot(triangulation, color=(1.0, 1.0, 1.0, 0.18), linewidth=0.25)
    axes[0, 0].plot(path_uv[:, 0], path_uv[:, 1], color="#f8fafc", linewidth=2.2, label="Optimized path")
    for zone in result["surface"].get("keep_outs", []):
        circle = plt.Circle(zone["center_uv"], zone["radius_uv"], color="#86efac", fill=False, linewidth=2.0)
        axes[0, 0].add_patch(circle)
    axes[0, 0].set_title("FEM displacement magnitude")
    axes[0, 0].set_xlabel("u")
    axes[0, 0].set_ylabel("v")
    axes[0, 0].legend(loc="lower right")
    fig.colorbar(displacement_plot, ax=axes[0, 0], fraction=0.046, pad=0.04, label="mm")

    stress_plot = axes[0, 1].tripcolor(
        triangulation,
        facecolors=von_mises_mpa,
        shading="flat",
        cmap="magma",
        edgecolors="none",
    )
    axes[0, 1].triplot(triangulation, color=(1.0, 1.0, 1.0, 0.10), linewidth=0.2)
    axes[0, 1].plot(path_uv[:, 0], path_uv[:, 1], color="#dbeafe", linewidth=2.0)
    for zone in result["surface"].get("keep_outs", []):
        circle = plt.Circle(zone["center_uv"], zone["radius_uv"], color="#86efac", fill=False, linewidth=2.0)
        axes[0, 1].add_patch(circle)
    axes[0, 1].set_title("Triangle von Mises stress")
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

    for ax in axes.flat[:2]:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    output_path = output_dir / "ifp_preview.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
