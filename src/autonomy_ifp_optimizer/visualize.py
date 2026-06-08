from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_preview(result: dict[str, object], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = result.get("history", [])
    path_uv = np.asarray(result["path_uv"], dtype=float)
    control_points = np.asarray(result["control_points_uv"], dtype=float)
    thickness_field = np.asarray(result["thickness_field"], dtype=float)
    radius_profile_mm = 1000.0 * np.asarray(result["radius_profile_m"], dtype=float)
    radius_limit_mm = float(result["metrics"]["radius_limit_mm"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im = axes[0].imshow(thickness_field, origin="lower", extent=[0.0, 1.0, 0.0, 1.0], cmap="magma")
    axes[0].plot(path_uv[:, 0], path_uv[:, 1], color="#f8f5f0", linewidth=2.5, label="Optimized path")
    axes[0].plot(control_points[:, 0], control_points[:, 1], "o--", color="#80d0ff", linewidth=1.5, label="Control polygon")
    for zone in result["surface"].get("keep_outs", []):
        circle = plt.Circle(zone["center_uv"], zone["radius_uv"], color="#7be495", fill=False, linewidth=2.0)
        axes[0].add_patch(circle)
    axes[0].set_title("IFP route + thickness map")
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("v")
    axes[0].legend(loc="lower right")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    if history:
        steps = [entry["step"] for entry in history]
        axes[1].plot(steps, [entry["loss"] for entry in history], label="Loss", color="#ff7f50", linewidth=2.0)
        axes[1].plot(steps, [entry["steering_penalty"] for entry in history], label="Steering", color="#4dd0e1", linewidth=1.8)
        axes[1].plot(steps, [entry["thickness_penalty"] for entry in history], label="Thickness", color="#fdd835", linewidth=1.8)
        axes[1].plot(steps, [entry["keepout_penalty"] for entry in history], label="Keep-out", color="#9ccc65", linewidth=1.8)
    axes[1].set_title("Optimization history")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Objective / penalties")
    axes[1].legend(loc="upper right")

    axes[2].plot(radius_profile_mm, color="#80d0ff", linewidth=2.0, label="Local steering radius")
    axes[2].axhline(radius_limit_mm, color="#ff7f50", linestyle="--", linewidth=1.5, label="Manufacturing limit")
    axes[2].set_title("Manufacturability check")
    axes[2].set_xlabel("Path sample")
    axes[2].set_ylabel("Radius [mm]")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    output_path = output_dir / "ifp_preview.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
