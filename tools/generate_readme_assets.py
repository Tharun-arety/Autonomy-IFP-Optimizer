from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autonomy_ifp_optimizer.export.toolpath import (
    compute_local_routing_effort,
    compute_metrics,
    load_optimized_path,
    write_interactive_toolpath_html,
)
from autonomy_ifp_optimizer.core.geometry import preview_keepouts, preview_surface, surface_from_dict


ASSETS_DIR = REPO_ROOT / "assets"
OUTPUTS_DIR = REPO_ROOT / "outputs"
SURROGATE_DIR = OUTPUTS_DIR / "surrogate_smoke"

CASES = [
    {
        "slug": "drone_frame_demo",
        "title": "Drone Frame Cutout Avoidance",
        "subtitle": "Plate-with-hole route rerouting around a central keep-out",
        "surface_color": "#0f766e",
        "accent": "#0ea5e9",
    },
    {
        "slug": "robotic_limb_demo",
        "title": "Robotic Limb Routing",
        "subtitle": "Cylindrical route optimization with the same differentiable loop",
        "surface_color": "#1d4ed8",
        "accent": "#f97316",
    },
]


def _normalize(vectors: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norm, eps)


def _percent_change(baseline: float, optimized: float) -> float:
    return 100.0 * (optimized - baseline) / max(abs(baseline), 1.0e-8)


def _format_keepout_transition(
    case_item: dict[str, object],
    baseline_metrics: dict[str, object],
    optimized_metrics: dict[str, object],
) -> str:
    if not case_item["surface"].keep_outs:
        return "Keep-out clearance: n/a (no keep-out zones)"
    return (
        f"Keep-out clearance: {baseline_metrics['minimum_keepout_clearance_uv']:+.3f} -> "
        f"{optimized_metrics['minimum_keepout_clearance_uv']:+.3f} uv"
    )


def _result_like(parent: dict[str, object], snapshot: dict[str, object]) -> dict[str, object]:
    return {
        "surface": parent["surface"],
        "optimization_config": parent.get("optimization_config", {}),
        "metrics": snapshot.get("metrics", {}),
        "path_uv": snapshot["path_uv"],
        "path_xyz": snapshot["path_xyz"],
        "normals": snapshot["normals"],
        "tangents": snapshot["tangents"],
        "radius_profile_m": snapshot["radius_profile_m"],
    }


def _set_display_3d_axes(ax: plt.Axes, case_item: dict[str, object]) -> None:
    xyz = np.asarray(case_item["mesh_xyz"], dtype=float)
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    span = np.maximum(maxs - mins, 1.0e-6)

    pad = np.array([0.08 * span[0], 0.10 * span[1], 0.08 * max(span[2], 0.01)], dtype=float)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])

    if case_item["surface"].kind == "plate_with_hole":
        major = max(span[0], span[1])
        center_z = 0.5 * (mins[2] + maxs[2])
        half_z = 0.17 * major
        ax.set_zlim(center_z - half_z, center_z + half_z)
    else:
        z_pad = 0.05 * span[2]
        ax.set_zlim(mins[2] - z_pad, maxs[2] + z_pad)


def _display_box_aspect(case_item: dict[str, object]) -> tuple[float, float, float]:
    span = np.maximum(np.ptp(np.asarray(case_item["mesh_xyz"], dtype=float), axis=0), 1.0e-6)
    major = float(max(span[0], span[1]))
    if case_item["surface"].kind == "plate_with_hole":
        span[2] = max(float(span[2]), 0.24 * major)
    return float(span[0]), float(span[1]), float(span[2])


def _sample_indices(count: int, target: int = 12) -> np.ndarray:
    stride = max(count // target, 1)
    indices = np.arange(0, count, stride, dtype=int)
    if indices[-1] != count - 1:
        indices = np.append(indices, count - 1)
    return indices


def _load_case(case: dict[str, str]) -> dict[str, object]:
    case_dir = OUTPUTS_DIR / case["slug"]
    optimized = load_optimized_path(case_dir / "optimized_path.json")
    metrics = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    optimized["metrics"] = metrics

    surface = surface_from_dict(optimized["surface"])
    mesh_xyz, mesh_faces = preview_surface(surface)
    keepout_curves = preview_keepouts(surface)
    preview = plt.imread(case_dir / "ifp_preview.png")

    baseline_snapshot = dict(optimized["baseline"])
    baseline_result = _result_like(optimized, baseline_snapshot)
    baseline_metrics = baseline_snapshot.get("metrics", {})
    if "estimated_cycle_time_s" not in baseline_metrics:
        baseline_metrics = compute_metrics(baseline_result)
        baseline_snapshot["metrics"] = baseline_metrics

    history = optimized["history"]
    frames = optimized.get("frames", [])
    if not frames:
        frames = [
            {
                "step": float(history[-1]["step"] if history else 0.0),
                "control_points_uv": optimized["control_points_uv"],
                "path_uv": optimized["path_uv"],
                "path_xyz": optimized["path_xyz"],
                "normals": optimized["normals"],
                "tangents": optimized["tangents"],
                "radius_profile_m": optimized["radius_profile_m"],
                "metrics": {
                    "loss": metrics["objective"],
                    "normalized_compliance": metrics["normalized_compliance"],
                    "min_steering_radius_m": metrics["min_steering_radius_m"],
                    "minimum_keepout_clearance_uv": metrics["minimum_keepout_clearance_uv"],
                    "maximum_displacement_m": metrics["maximum_displacement_m"],
                    "maximum_von_mises_mpa": metrics["maximum_von_mises_mpa"],
                },
            }
        ]

    optimized_effort = compute_local_routing_effort(optimized)
    baseline_effort = compute_local_routing_effort(baseline_result)
    frame_efforts = [compute_local_routing_effort(_result_like(optimized, frame)) for frame in frames]
    effort_max = max(
        1.0,
        float(np.max(optimized_effort)),
        float(np.max(baseline_effort)),
        *(float(np.max(values)) for values in frame_efforts),
    )

    write_interactive_toolpath_html(optimized, case_dir)

    return {
        "meta": case,
        "case_dir": case_dir,
        "surface": surface,
        "mesh_xyz": mesh_xyz,
        "mesh_faces": mesh_faces,
        "keepout_curves": keepout_curves,
        "preview": preview,
        "optimized": optimized,
        "metrics": metrics,
        "baseline": baseline_snapshot,
        "baseline_result": baseline_result,
        "baseline_metrics": baseline_metrics,
        "optimized_effort": optimized_effort,
        "baseline_effort": baseline_effort,
        "frame_efforts": frame_efforts,
        "effort_max": effort_max,
        "history_steps": np.asarray([entry["step"] for entry in history], dtype=float),
        "history_loss": np.asarray([entry["loss"] for entry in history], dtype=float),
        "history_compliance": np.asarray([entry["normalized_compliance"] for entry in history], dtype=float),
        "history_displacement_mm": 1000.0 * np.asarray([entry["maximum_displacement_m"] for entry in history], dtype=float),
        "history_stress_mpa": np.asarray([entry["maximum_von_mises_mpa"] for entry in history], dtype=float),
        "history_radius_mm": 1000.0 * np.asarray([entry["min_steering_radius_m"] for entry in history], dtype=float),
        "frames": frames,
    }


def _draw_surface(ax: plt.Axes, case_item: dict[str, object]) -> None:
    mesh_xyz = np.asarray(case_item["mesh_xyz"], dtype=float)
    mesh_faces = np.asarray(case_item["mesh_faces"], dtype=np.int32)
    ax.plot_trisurf(
        mesh_xyz[:, 0],
        mesh_xyz[:, 1],
        mesh_xyz[:, 2],
        triangles=mesh_faces,
        color="#e7e5e4",
        alpha=0.24,
        linewidth=0.15,
        edgecolor=(0.50, 0.50, 0.50, 0.12),
        shade=False,
    )
    for curve in case_item["keepout_curves"]:
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#22c55e", linewidth=2.0, alpha=0.95)
    _set_display_3d_axes(ax, case_item)
    if case_item["surface"].kind == "cylinder":
        ax.view_init(elev=18, azim=-36)
    else:
        ax.view_init(elev=23, azim=-61)
    ax.set_box_aspect(_display_box_aspect(case_item))
    ax.set_axis_off()
    ax.grid(False)


def _draw_path_3d(
    ax: plt.Axes,
    case_item: dict[str, object],
    snapshot: dict[str, object],
    effort: np.ndarray,
    *,
    title: str | None,
) -> plt.Artist:
    xyz = np.asarray(snapshot["path_xyz"], dtype=float)
    normals = _normalize(np.asarray(snapshot["normals"], dtype=float))
    tangents = _normalize(np.asarray(snapshot["tangents"], dtype=float))
    roll = _normalize(np.cross(normals, tangents))
    vector_scale = 0.11 * float(np.max(np.ptp(np.asarray(case_item["mesh_xyz"], dtype=float), axis=0)))
    sampled = _sample_indices(xyz.shape[0], target=14)

    line_color = case_item["meta"]["surface_color"]
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=line_color, linewidth=2.4, alpha=0.96)
    scatter = ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=effort,
        cmap="turbo",
        vmin=0.0,
        vmax=float(case_item["effort_max"]),
        s=14,
        depthshade=False,
    )
    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color="#22c55e", s=40, depthshade=False)
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color="#ef4444", s=40, depthshade=False)
    ax.quiver(
        xyz[sampled, 0],
        xyz[sampled, 1],
        xyz[sampled, 2],
        normals[sampled, 0],
        normals[sampled, 1],
        normals[sampled, 2],
        length=vector_scale,
        normalize=True,
        color="#2563eb",
        linewidth=0.8,
        alpha=0.80,
    )
    ax.quiver(
        xyz[sampled, 0],
        xyz[sampled, 1],
        xyz[sampled, 2],
        roll[sampled, 0],
        roll[sampled, 1],
        roll[sampled, 2],
        length=0.72 * vector_scale,
        normalize=True,
        color="#f97316",
        linewidth=0.7,
        alpha=0.52,
    )
    if title:
        ax.set_title(title, fontsize=13, weight="bold", pad=10)
    return scatter


def _draw_uv_domain(ax: plt.Axes, case_item: dict[str, object], snapshot: dict[str, object], frame_index: int | None = None) -> None:
    baseline = case_item["baseline"]
    frames = case_item["frames"]
    baseline_path = np.asarray(baseline["path_uv"], dtype=float)
    current_path = np.asarray(snapshot["path_uv"], dtype=float)
    control = np.asarray(snapshot["control_points_uv"], dtype=float)

    ax.plot(baseline_path[:, 0], baseline_path[:, 1], linestyle="--", linewidth=1.8, color="#94a3b8", label="Naive seed")
    ax.plot(current_path[:, 0], current_path[:, 1], linewidth=2.5, color=case_item["meta"]["surface_color"], label="Current path")
    ax.plot(control[:, 0], control[:, 1], linestyle=":", linewidth=1.4, color="#111827", marker="o", markersize=4, label="Bezier control")
    ax.scatter(current_path[0, 0], current_path[0, 1], color="#22c55e", s=38, zorder=3)
    ax.scatter(current_path[-1, 0], current_path[-1, 1], color="#ef4444", s=38, zorder=3)

    if frame_index is not None and frame_index >= 0:
        cp_history = np.asarray([frame["control_points_uv"] for frame in frames[: frame_index + 1]], dtype=float)
        ax.plot(cp_history[:, 1, 0], cp_history[:, 1, 1], color="#cbd5e1", linewidth=1.0, alpha=0.9)
        ax.plot(cp_history[:, 2, 0], cp_history[:, 2, 1], color="#cbd5e1", linewidth=1.0, alpha=0.9)

    for zone in case_item["optimized"]["surface"].get("keep_outs", []):
        circle = plt.Circle(zone["center_uv"], zone["radius_uv"], fill=False, linewidth=2.0, edgecolor="#22c55e")
        ax.add_patch(circle)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_title("Path evolution in UV", fontsize=12, weight="bold", loc="left")


def _draw_convergence(ax: plt.Axes, case_item: dict[str, object], frame_index: int) -> None:
    steps = np.asarray(case_item["history_steps"], dtype=float)
    loss = np.asarray(case_item["history_loss"], dtype=float)
    compliance = np.asarray(case_item["history_compliance"], dtype=float)
    radius_mm = np.asarray(case_item["history_radius_mm"], dtype=float)
    current_step = float(case_item["frames"][frame_index]["step"])
    current_mask = steps <= current_step + 1.0e-6

    loss_norm = loss / max(loss[0], 1.0e-8)
    compliance_norm = compliance / max(compliance[0], 1.0e-8)
    radius_limit_mm = float(case_item["metrics"]["radius_limit_mm"])
    radius_ratio = radius_mm / radius_limit_mm

    radius_ax = ax.twinx()
    ax.plot(steps, loss_norm, color="#ef4444", linewidth=1.3, alpha=0.16)
    ax.plot(steps, compliance_norm, color="#2563eb", linewidth=1.3, alpha=0.16)
    radius_ax.plot(steps, radius_ratio, color="#0f766e", linewidth=1.3, alpha=0.16)
    loss_line = ax.plot(steps[current_mask], loss_norm[current_mask], color="#ef4444", linewidth=2.2, label="Loss / step 0")[0]
    compliance_line = ax.plot(
        steps[current_mask],
        compliance_norm[current_mask],
        color="#2563eb",
        linewidth=2.0,
        label="Compliance / step 0",
    )[0]
    radius_line = radius_ax.plot(
        steps[current_mask],
        radius_ratio[current_mask],
        color="#0f766e",
        linewidth=2.0,
        label="Radius / limit",
    )[0]
    ax.axvline(current_step, color="#111827", linewidth=1.3, linestyle="--", alpha=0.7)
    ax.set_xlim(float(steps[0]), float(steps[-1]))
    ax.set_ylim(0.0, 1.08 * max(float(np.max(loss_norm)), float(np.max(compliance_norm))))
    radius_ax.set_ylim(0.0, 1.08 * float(np.max(radius_ratio)))
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss / compliance trend")
    radius_ax.set_ylabel("Radius / limit")
    ax.set_title("Loss, compliance, and steering", fontsize=12, weight="bold", loc="left")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(handles=[loss_line, compliance_line, radius_line], frameon=False, fontsize=9, loc="upper right")


def create_optimization_animation(case_item: dict[str, object], output_path: Path) -> None:
    frames = case_item["frames"]
    images: list[Image.Image] = []
    final_metrics = case_item["metrics"]
    baseline_metrics = case_item["baseline_metrics"]
    total_steps = int(case_item["history_steps"][-1]) if case_item["history_steps"].size else 0

    for index, frame in enumerate(frames):
        effort = case_item["frame_efforts"][index]
        fig = plt.figure(figsize=(13.8, 7.7))
        grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 0.86], wspace=0.18, hspace=0.28)
        ax3d = fig.add_subplot(grid[:, 0], projection="3d")
        ax_uv = fig.add_subplot(grid[0, 1])
        ax_conv = fig.add_subplot(grid[1, 1])

        _draw_surface(ax3d, case_item)
        _draw_path_3d(
            ax3d,
            case_item,
            frame,
            effort,
            title=None,
        )
        _draw_uv_domain(ax_uv, case_item, frame, frame_index=index)
        _draw_convergence(ax_conv, case_item, index)

        fig.subplots_adjust(left=0.035, right=0.98, top=0.90, bottom=0.10)
        fig.suptitle("Optimization Evolution", fontsize=18, weight="bold", y=0.985)
        fig.text(
            0.05,
            0.925,
            f"{case_item['meta']['title']} | step {int(frame['step'])}/{total_steps}",
            fontsize=13.5,
            weight="bold",
            color="#111827",
        )
        fig.text(
            0.56,
            0.035,
            "Blue arrows show the tool axis aligned to the local surface normal. Orange arrows show the roll axis exported with the path.",
            fontsize=9.5,
            color="#4b5563",
        )

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        frame_image = Image.open(buffer)
        images.append(frame_image.convert("P", palette=Image.ADAPTIVE).copy())
        frame_image.close()
        buffer.close()

    hold_frames = [images[-1].copy() for _ in range(6)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:] + hold_frames,
        duration=[130] * max(len(images) - 1, 0) + [220] + [250] * len(hold_frames),
        loop=0,
        disposal=2,
    )


def _draw_effort_comparison(
    ax: plt.Axes,
    case_item: dict[str, object],
    snapshot: dict[str, object],
    effort: np.ndarray,
    title: str,
) -> plt.Artist:
    _draw_surface(ax, case_item)
    artist = _draw_path_3d(ax, case_item, snapshot, effort, title=title)
    return artist


def create_naive_vs_optimized_heatmap(case_item: dict[str, object], output_path: Path) -> None:
    fig = plt.figure(figsize=(13.2, 8.4))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.055], height_ratios=[1.02, 0.84], hspace=0.28, wspace=0.16)
    ax_base = fig.add_subplot(grid[0, 0], projection="3d")
    ax_opt = fig.add_subplot(grid[0, 1], projection="3d")
    cax = fig.add_subplot(grid[0, 2])
    ax_profile = fig.add_subplot(grid[1, 0])
    ax_text = fig.add_subplot(grid[1, 1])
    ax_blank = fig.add_subplot(grid[1, 2])
    ax_blank.axis("off")

    scatter = _draw_effort_comparison(
        ax_base,
        case_item,
        case_item["baseline"],
        case_item["baseline_effort"],
        "Naive seed path",
    )
    _draw_effort_comparison(
        ax_opt,
        case_item,
        case_item["optimized"],
        case_item["optimized_effort"],
        "Optimized path",
    )
    colorbar = fig.colorbar(scatter, cax=cax)
    colorbar.set_label("Routing effort proxy")

    base_result = case_item["baseline_result"]
    base_length = float(base_result["metrics"]["path_length_m"])
    opt_length = float(case_item["metrics"]["path_length_m"])
    base_arclength = np.linspace(0.0, base_length, len(case_item["baseline_effort"]))
    opt_arclength = np.linspace(0.0, opt_length, len(case_item["optimized_effort"]))

    ax_profile.plot(base_arclength, case_item["baseline_effort"], color="#94a3b8", linewidth=2.2, label="Naive seed")
    ax_profile.plot(opt_arclength, case_item["optimized_effort"], color=case_item["meta"]["surface_color"], linewidth=2.4, label="Optimized")
    ax_profile.set_xlabel("Arclength [m]")
    ax_profile.set_ylabel("Routing effort proxy")
    ax_profile.set_title("How route effort shifts along the course", fontsize=12.5, weight="bold", loc="left")
    ax_profile.grid(True, linestyle="--", alpha=0.26)
    ax_profile.legend(frameon=False, fontsize=9)

    baseline = case_item["baseline_metrics"]
    optimized = case_item["metrics"]
    summary_lines = [
        f"Normalized compliance: {baseline['normalized_compliance']:.3f} -> {optimized['normalized_compliance']:.3f} ({_percent_change(baseline['normalized_compliance'], optimized['normalized_compliance']):+.1f}%)",
        _format_keepout_transition(case_item, baseline, optimized),
        f"Min steering radius: {baseline['min_steering_radius_mm']:.1f} -> {optimized['min_steering_radius_mm']:.1f} mm",
        f"Peak routing effort: {baseline['peak_routing_effort']:.2f} -> {optimized['peak_routing_effort']:.2f}",
        f"Cycle time: {baseline['estimated_cycle_time_s']:.3f} -> {optimized['estimated_cycle_time_s']:.3f} s",
        f"Material usage: {baseline['estimated_material_weight_g']:.3f} -> {optimized['estimated_material_weight_g']:.3f} g",
    ]
    ax_text.axis("off")
    ax_text.set_title("What the optimizer trades to gain stiffness", fontsize=12.5, weight="bold", loc="left")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.75,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    ax_text.text(
        0.0,
        0.12,
        "This repository does not solve joint-space IK yet.\nThe colormap is a robot-facing proxy built from steering-radius margin,\ntool-axis rotation rate, heading change, and keep-out margin.",
        va="top",
        ha="left",
        fontsize=10,
        color="#4b5563",
        linespacing=1.55,
    )

    fig.suptitle("Naive vs Optimized Route Tradeoff", fontsize=18, weight="bold")
    fig.subplots_adjust(left=0.055, right=0.955, top=0.92, bottom=0.08)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_surrogate_validation(output_path: Path) -> None:
    validation = np.load(SURROGATE_DIR / "surrogate_validation.npz")
    metrics = json.loads((SURROGATE_DIR / "surrogate_metrics.json").read_text(encoding="utf-8"))
    target_names = validation["target_names"].tolist()
    y_true = np.asarray(validation["y_true"], dtype=float)
    y_pred = np.asarray(validation["y_pred"], dtype=float)
    history = metrics["history"]

    fig = plt.figure(figsize=(12.8, 7.8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.82], wspace=0.28, hspace=0.32)
    ax_loss = fig.add_subplot(grid[0, 0])
    ax_comp = fig.add_subplot(grid[0, 1])
    ax_hist = fig.add_subplot(grid[1, 0])
    ax_text = fig.add_subplot(grid[1, 1])

    target_index = target_names.index("total_loss")
    comp_index = target_names.index("normalized_compliance")

    min_loss = float(min(np.min(y_true[:, target_index]), np.min(y_pred[:, target_index])))
    max_loss = float(max(np.max(y_true[:, target_index]), np.max(y_pred[:, target_index])))
    ax_loss.scatter(y_true[:, target_index], y_pred[:, target_index], color="#0f766e", alpha=0.82, s=34)
    ax_loss.plot([min_loss, max_loss], [min_loss, max_loss], color="#111827", linewidth=1.4, linestyle="--")
    ax_loss.set_xlabel("Ground truth total loss")
    ax_loss.set_ylabel("Surrogate prediction")
    ax_loss.set_title("Total loss parity", fontsize=12.5, weight="bold")
    ax_loss.grid(True, linestyle="--", alpha=0.25)

    min_comp = float(min(np.min(y_true[:, comp_index]), np.min(y_pred[:, comp_index])))
    max_comp = float(max(np.max(y_true[:, comp_index]), np.max(y_pred[:, comp_index])))
    ax_comp.scatter(y_true[:, comp_index], y_pred[:, comp_index], color="#2563eb", alpha=0.82, s=34)
    ax_comp.plot([min_comp, max_comp], [min_comp, max_comp], color="#111827", linewidth=1.4, linestyle="--")
    ax_comp.set_xlabel("Ground truth normalized compliance")
    ax_comp.set_ylabel("Surrogate prediction")
    ax_comp.set_title("Compliance parity", fontsize=12.5, weight="bold")
    ax_comp.grid(True, linestyle="--", alpha=0.25)

    epochs = np.asarray([entry["epoch"] for entry in history], dtype=float)
    train_mse = np.asarray([entry["train_mse"] for entry in history], dtype=float)
    val_mse = np.asarray([entry["val_mse"] for entry in history], dtype=float)
    ax_hist.plot(epochs, train_mse, color="#0f766e", linewidth=2.1, label="Train MSE")
    ax_hist.plot(epochs, val_mse, color="#ef4444", linewidth=2.1, label="Validation MSE")
    ax_hist.set_xlabel("Epoch")
    ax_hist.set_ylabel("Normalized MSE")
    ax_hist.set_title("Training MSE", fontsize=12.5, weight="bold")
    ax_hist.grid(True, linestyle="--", alpha=0.25)
    ax_hist.legend(frameon=False, fontsize=9)

    ax_text.axis("off")
    ax_text.set_title("Smoke-test surrogate metrics", fontsize=12.5, weight="bold", loc="left")
    ax_text.text(
        0.0,
        1.0,
        (
            f"Samples: {metrics['samples']}\n"
            f"Epochs: {metrics['epochs']}\n"
            f"Validation RMSE: {metrics['validation_rmse']:.3f}\n"
            f"Inference latency: {metrics['inference_latency_ms']:.2f} ms\n"
            f"Outputs learned:\n  {', '.join(target_names)}"
        ),
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.8,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    ax_text.text(
        0.0,
        0.18,
        "The checked-in run is still a smoke test. The point of this figure is to show persisted validation artifacts, not to claim production surrogate accuracy from 64 samples.",
        va="top",
        ha="left",
        fontsize=10,
        color="#4b5563",
        linespacing=1.55,
    )

    fig.suptitle("Surrogate Validation", fontsize=18, weight="bold")
    fig.subplots_adjust(left=0.07, right=0.96, top=0.92, bottom=0.09)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_showcase_overview(case_data: list[dict[str, object]], output_path: Path) -> None:
    fig = plt.figure(figsize=(13.4, 8.8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.12, 0.96], height_ratios=[1.0, 1.0], hspace=0.25, wspace=0.12)
    fig.suptitle("Differentiable IFP workflows with saved structural and robot-facing outputs", fontsize=19, weight="bold")

    for row, item in enumerate(case_data):
        ax3d = fig.add_subplot(grid[row, 0], projection="3d")
        ax_text = fig.add_subplot(grid[row, 1])
        _draw_surface(ax3d, item)
        _draw_path_3d(ax3d, item, item["optimized"], item["optimized_effort"], title=None)
        ax_text.axis("off")
        baseline = item["baseline_metrics"]
        optimized = item["metrics"]
        lines = [
            item["meta"]["title"],
            "",
            item["meta"]["subtitle"],
            f"Compliance: {baseline['normalized_compliance']:.3f} -> {optimized['normalized_compliance']:.3f}",
            _format_keepout_transition(item, baseline, optimized),
            f"Min steering radius: {optimized['min_steering_radius_mm']:.1f} mm",
            f"Peak routing effort: {optimized['peak_routing_effort']:.2f}",
            "",
            f"Open locally: outputs/{item['meta']['slug']}/interactive_toolpath.html",
            "GitHub shows checked-in HTML as source text.",
        ]
        ax_text.text(
            0.0,
            0.96,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=11,
            linespacing=1.62,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#f8fafc", edgecolor="#cbd5e1"),
        )

    fig.subplots_adjust(left=0.05, right=0.96, top=0.93, bottom=0.05)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_optimization_profiles(case_data: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), layout="constrained")

    for item in case_data:
        axes[0].plot(
            item["history_steps"],
            item["history_compliance"],
            linewidth=2.3,
            color=item["meta"]["surface_color"],
            label=item["meta"]["title"],
        )
    axes[0].set_title("Normalized compliance during optimization", fontsize=12.5, weight="bold")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Compliance / reference")
    axes[0].grid(True, linestyle="--", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    for item in case_data:
        axes[1].plot(
            item["history_steps"],
            item["history_radius_mm"],
            linewidth=2.3,
            color=item["meta"]["surface_color"],
            label=item["meta"]["title"],
        )
    axes[1].axhline(float(case_data[0]["metrics"]["radius_limit_mm"]), color="#f97316", linewidth=1.6, linestyle="--")
    axes[1].set_title("Minimum steering radius during optimization", fontsize=12.5, weight="bold")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Radius [mm]")
    axes[1].grid(True, linestyle="--", alpha=0.25)

    labels = [item["meta"]["title"] for item in case_data]
    peak_effort = [float(item["metrics"]["peak_routing_effort"]) for item in case_data]
    cycle_time = [float(item["metrics"]["estimated_cycle_time_s"]) for item in case_data]
    x = np.arange(len(labels), dtype=float)
    axes[2].bar(x - 0.18, peak_effort, width=0.34, color="#0f766e", label="Peak routing effort")
    axes[2].bar(x + 0.18, cycle_time, width=0.34, color="#94a3b8", label="Cycle time [s]")
    axes[2].set_xticks(x, labels, rotation=10, ha="right")
    axes[2].set_title("Final robot-facing export metrics", fontsize=12.5, weight="bold")
    axes[2].grid(axis="y", linestyle="--", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=9)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    case_data = [_load_case(case) for case in CASES]
    create_optimization_animation(case_data[0], ASSETS_DIR / "optimization_evolution.gif")
    create_naive_vs_optimized_heatmap(case_data[0], ASSETS_DIR / "naive_vs_optimized_heatmap.png")
    create_showcase_overview(case_data, ASSETS_DIR / "demo_showcase.png")
    create_optimization_profiles(case_data, ASSETS_DIR / "optimization_profiles.png")
    create_surrogate_validation(ASSETS_DIR / "surrogate_validation.png")


if __name__ == "__main__":
    main()
