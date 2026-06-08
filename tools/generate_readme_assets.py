from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
OUTPUTS_DIR = REPO_ROOT / "outputs"

CASES = [
    {
        "slug": "drone_frame_demo",
        "title": "Drone Frame Cutout Avoidance",
        "subtitle": "2.5D plate with keep-out routing",
        "color": "#0f766e",
    },
    {
        "slug": "robotic_limb_demo",
        "title": "Robotic Limb Routing",
        "subtitle": "Cylindrical surface path planning",
        "color": "#1d4ed8",
    },
]


def _load_case(case: dict[str, str]) -> dict[str, object]:
    case_dir = OUTPUTS_DIR / case["slug"]
    metrics = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    optimized = json.loads((case_dir / "optimized_path.json").read_text(encoding="utf-8"))
    preview = plt.imread(case_dir / "ifp_preview.png")
    history = optimized["history"]
    first_history = history[0]
    last_history = history[-1]
    objective_drop_pct = 100.0 * (float(last_history["loss"]) - float(first_history["loss"])) / max(
        float(first_history["loss"]), 1.0e-8
    )
    return {
        "meta": case,
        "metrics": metrics,
        "optimized": optimized,
        "preview": preview,
        "thickness_field": np.asarray(optimized["thickness_field"], dtype=float),
        "path_uv": np.asarray(optimized["path_uv"], dtype=float),
        "control_points_uv": np.asarray(optimized["control_points_uv"], dtype=float),
        "radius_profile_mm": 1000.0 * np.asarray(optimized["radius_profile_m"], dtype=float),
        "history_steps": np.asarray([point["step"] for point in history], dtype=float),
        "history_loss": np.asarray([point["loss"] for point in history], dtype=float),
        "history_steering": np.asarray([point["steering_penalty"] for point in history], dtype=float),
        "history_thickness": np.asarray([point["thickness_penalty"] for point in history], dtype=float),
        "history_keepout": np.asarray([point["keepout_penalty"] for point in history], dtype=float),
        "history_radius_mm": 1000.0 * np.asarray([point["min_steering_radius_m"] for point in history], dtype=float),
        "history_peak_thickness": np.asarray([point["peak_thickness"] for point in history], dtype=float),
        "objective_drop_pct": objective_drop_pct,
    }


def _metric_lines(case_data: dict[str, object]) -> list[str]:
    metrics = case_data["metrics"]
    optimized = case_data["optimized"]
    surface = optimized["surface"]
    keep_outs = surface.get("keep_outs", [])
    clearance = metrics["minimum_keepout_clearance_uv"]
    lines = [
        f"Manufacturable: {'yes' if metrics['manufacturable'] else 'no'}",
        f"Objective: {metrics['objective']:.3f} ({case_data['objective_drop_pct']:+.1f}% vs step 0)",
        f"Path length: {metrics['path_length_m']:.3f} m",
        f"Min steering radius: {metrics['min_steering_radius_mm']:.1f} mm",
        f"Cycle time: {metrics['estimated_cycle_time_s']:.3f} s",
        f"Material weight: {metrics['estimated_material_weight_g']:.3f} g",
        f"Toolpath points: {metrics['toolpath_points']}",
    ]
    if keep_outs:
        lines.append(f"Keep-out clearance: {clearance:.3f} uv")
    return lines


def create_showcase_overview(case_data: list[dict[str, object]], output_path: Path) -> None:
    fig = plt.figure(figsize=(15.0, 10.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.8, 1.0], height_ratios=[1.0, 1.0])
    fig.suptitle("Two validated differentiable IFP workflows", fontsize=20, weight="bold")

    for row, item in enumerate(case_data):
        preview_ax = fig.add_subplot(grid[row, 0])
        preview_ax.imshow(item["preview"])
        preview_ax.set_axis_off()
        preview_ax.set_title(
            f"{item['meta']['title']} | {item['meta']['subtitle']}",
            fontsize=14,
            weight="bold",
            loc="left",
            pad=10,
        )

        info_ax = fig.add_subplot(grid[row, 1])
        info_ax.axis("off")
        info_ax.text(
            0.0,
            1.0,
            "What this case demonstrates",
            va="top",
            ha="left",
            fontsize=13,
            weight="bold",
        )
        case_copy = item["meta"]["subtitle"]
        surface_kind = item["optimized"]["surface"]["kind"]
        info_ax.text(
            0.0,
            0.86,
            f"{case_copy}\nSurface kind: {surface_kind}",
            va="top",
            ha="left",
            fontsize=10.5,
            color="#374151",
            linespacing=1.5,
        )
        info_ax.text(
            0.0,
            0.60,
            "\n".join(_metric_lines(item)),
            va="top",
            ha="left",
            fontsize=10.5,
            linespacing=1.75,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#f4f6f8", edgecolor="#d0d7de"),
        )

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_uv_case(ax: plt.Axes, item: dict[str, object]) -> None:
    path_uv = item["path_uv"]
    control_points = item["control_points_uv"]
    surface = item["optimized"]["surface"]
    keep_outs = surface.get("keep_outs", [])

    ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, fill=False, linewidth=1.2, edgecolor="#6b7280"))
    ax.plot(path_uv[:, 0], path_uv[:, 1], color=item["meta"]["color"], linewidth=2.6, label="Optimized path")
    ax.plot(
        control_points[:, 0],
        control_points[:, 1],
        linestyle="--",
        color="#111827",
        linewidth=1.4,
        marker="o",
        markersize=4.5,
        label="Control polygon",
    )
    ax.scatter(path_uv[0, 0], path_uv[0, 1], color="#22c55e", s=45, zorder=3, label="Start")
    ax.scatter(path_uv[-1, 0], path_uv[-1, 1], color="#ef4444", s=45, zorder=3, label="End")

    for keep_out in keep_outs:
        center = keep_out["center_uv"]
        radius = keep_out["radius_uv"]
        ax.add_patch(Circle(center, radius, fill=False, linewidth=2.2, edgecolor="#86efac"))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title(item["meta"]["title"], fontsize=12.5, weight="bold")


def create_toolpath_diagnostics(case_data: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), layout="constrained")

    _plot_uv_case(axes[0], case_data[0])
    _plot_uv_case(axes[1], case_data[1])

    limit_mm = float(case_data[0]["metrics"]["radius_limit_mm"])
    for item in case_data:
        axes[2].plot(
            np.arange(item["radius_profile_mm"].shape[0]),
            item["radius_profile_mm"],
            linewidth=2.4,
            color=item["meta"]["color"],
            label=item["meta"]["title"],
        )
    axes[2].axhline(limit_mm, color="#f97316", linestyle="--", linewidth=1.8, label="Manufacturing limit")
    axes[2].fill_between(
        np.arange(case_data[0]["radius_profile_mm"].shape[0]),
        0.0,
        limit_mm,
        color="#f97316",
        alpha=0.08,
    )
    axes[2].set_title("Local steering radius along the exported toolpath", fontsize=12.5, weight="bold")
    axes[2].set_xlabel("Path sample")
    axes[2].set_ylabel("Radius [mm]")
    axes[2].grid(True, linestyle="--", alpha=0.3)
    axes[2].legend(frameon=False, fontsize=9)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_optimization_profiles(case_data: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), layout="constrained")

    for item in case_data:
        axes[0].plot(
            item["history_steps"],
            item["history_loss"],
            linewidth=2.4,
            color=item["meta"]["color"],
            label=item["meta"]["title"],
        )
    axes[0].set_yscale("log")
    axes[0].set_title("Objective value over optimization steps", fontsize=12.5, weight="bold")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9)

    for item in case_data:
        axes[1].plot(
            item["history_steps"],
            item["history_radius_mm"],
            linewidth=2.4,
            color=item["meta"]["color"],
            label=item["meta"]["title"],
        )
    axes[1].axhline(float(case_data[0]["metrics"]["radius_limit_mm"]), color="#f97316", linestyle="--", linewidth=1.8)
    axes[1].set_title("Minimum steering radius during optimization", fontsize=12.5, weight="bold")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Radius [mm]")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    metric_names = ["Path length [m]", "Cycle time [s]", "Material [g]"]
    drone_values = [
        float(case_data[0]["metrics"]["path_length_m"]),
        float(case_data[0]["metrics"]["estimated_cycle_time_s"]),
        float(case_data[0]["metrics"]["estimated_material_weight_g"]),
    ]
    limb_values = [
        float(case_data[1]["metrics"]["path_length_m"]),
        float(case_data[1]["metrics"]["estimated_cycle_time_s"]),
        float(case_data[1]["metrics"]["estimated_material_weight_g"]),
    ]
    y_pos = np.arange(len(metric_names))
    axes[2].barh(y_pos + 0.16, drone_values, height=0.28, color=case_data[0]["meta"]["color"], label=case_data[0]["meta"]["title"])
    axes[2].barh(y_pos - 0.16, limb_values, height=0.28, color=case_data[1]["meta"]["color"], label=case_data[1]["meta"]["title"])
    axes[2].set_yticks(y_pos, metric_names)
    axes[2].invert_yaxis()
    axes[2].set_title("Final process metrics", fontsize=12.5, weight="bold")
    axes[2].grid(axis="x", linestyle="--", alpha=0.3)
    axes[2].legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, -0.08))
    for idx, value in enumerate(drone_values):
        axes[2].text(value, idx + 0.16, f" {value:.3f}", va="center", ha="left", fontsize=8.5)
    for idx, value in enumerate(limb_values):
        axes[2].text(value, idx - 0.16, f" {value:.3f}", va="center", ha="left", fontsize=8.5)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_output_breakdown(case_item: dict[str, object], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 12.0), layout="constrained", height_ratios=[1.15, 1.0, 1.0])
    fig.suptitle(f"{case_item['meta']['title']} output", fontsize=18, weight="bold")

    top_ax = axes[0]
    heatmap = top_ax.imshow(
        case_item["thickness_field"],
        extent=(0.0, 1.0, 0.0, 1.0),
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    top_ax.plot(
        case_item["path_uv"][:, 0],
        case_item["path_uv"][:, 1],
        color="#f8fafc",
        linewidth=2.4,
        label="Optimized path",
    )
    top_ax.plot(
        case_item["control_points_uv"][:, 0],
        case_item["control_points_uv"][:, 1],
        linestyle="--",
        color="#7dd3fc",
        linewidth=1.5,
        marker="o",
        markersize=4.5,
        label="Control polygon",
    )
    for keep_out in case_item["optimized"]["surface"].get("keep_outs", []):
        top_ax.add_patch(
            Circle(
                keep_out["center_uv"],
                keep_out["radius_uv"],
                fill=False,
                linewidth=2.0,
                edgecolor="#86efac",
            )
        )
    top_ax.set_title("Route and deposited thickness field", fontsize=13.5, weight="bold", loc="left", pad=12)
    top_ax.text(
        0.0,
        1.01,
        "README-friendly reconstruction from the saved optimization output.",
        transform=top_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.0,
        color="#4b5563",
    )
    top_ax.set_xlabel("u")
    top_ax.set_ylabel("v")
    top_ax.set_xlim(0.0, 1.0)
    top_ax.set_ylim(0.0, 1.0)
    top_ax.set_aspect("equal", adjustable="box")
    top_ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    top_ax.grid(False)
    fig.colorbar(heatmap, ax=top_ax, fraction=0.046, pad=0.03, label="Relative thickness")

    middle_ax = axes[1]
    middle_ax.plot(case_item["history_steps"], case_item["history_loss"], color="#fb6a4a", linewidth=2.2, label="Loss")
    middle_ax.plot(case_item["history_steps"], case_item["history_steering"], color="#22d3ee", linewidth=1.8, label="Steering")
    middle_ax.plot(case_item["history_steps"], case_item["history_thickness"], color="#facc15", linewidth=1.8, label="Thickness")
    middle_ax.plot(case_item["history_steps"], case_item["history_keepout"], color="#84cc16", linewidth=1.8, label="Keep-out")
    middle_ax.set_title("Optimization history", fontsize=13.5, weight="bold", loc="left", pad=12)
    middle_ax.text(
        0.0,
        1.01,
        "Objective and penalty evolution over solver steps.",
        transform=middle_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.0,
        color="#4b5563",
    )
    middle_ax.set_xlabel("Step")
    middle_ax.set_ylabel("Objective / penalties")
    middle_ax.grid(True, linestyle="--", alpha=0.28)
    middle_ax.legend(loc="upper right", frameon=True, fontsize=9.5)

    bottom_ax = axes[2]
    bottom_ax.plot(
        np.arange(case_item["radius_profile_mm"].shape[0]),
        case_item["radius_profile_mm"],
        color="#7dd3fc",
        linewidth=2.2,
        label="Local steering radius",
    )
    bottom_ax.axhline(
        float(case_item["metrics"]["radius_limit_mm"]),
        color="#fb923c",
        linestyle="--",
        linewidth=1.8,
        label="Manufacturing limit",
    )
    bottom_ax.set_title("Manufacturability check", fontsize=13.5, weight="bold", loc="left", pad=12)
    bottom_ax.text(
        0.0,
        1.01,
        "Local steering radius along the exported toolpath against the configured limit.",
        transform=bottom_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.0,
        color="#4b5563",
    )
    bottom_ax.set_xlabel("Path sample")
    bottom_ax.set_ylabel("Radius [mm]")
    bottom_ax.grid(True, linestyle="--", alpha=0.28)
    bottom_ax.legend(loc="upper right", frameon=True, fontsize=9.5)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    case_data = [_load_case(case) for case in CASES]
    create_showcase_overview(case_data, ASSETS_DIR / "demo_showcase.png")
    create_toolpath_diagnostics(case_data, ASSETS_DIR / "toolpath_diagnostics.png")
    create_optimization_profiles(case_data, ASSETS_DIR / "optimization_profiles.png")
    create_output_breakdown(case_data[0], ASSETS_DIR / "drone_frame_output_breakdown.png")
    create_output_breakdown(case_data[1], ASSETS_DIR / "robotic_limb_output_breakdown.png")


if __name__ == "__main__":
    main()
