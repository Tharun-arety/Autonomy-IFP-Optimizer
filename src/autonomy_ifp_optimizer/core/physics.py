from __future__ import annotations

from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..config import LoadCase, OptimizationConfig
from .constraints import (
    boundary_penalty,
    keepout_penalty,
    smoothness_penalty,
    steering_penalty,
    thickness_penalty,
)
from .geometry import SurfaceDefinition, preferred_fiber_direction, stress_weight, surface_normals, surface_xyz


def _normalize(vector: jnp.ndarray, eps: float = 1.0e-8) -> jnp.ndarray:
    norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / jnp.maximum(norm, eps)


def _sigmoid_bounds(raw: jnp.ndarray, lower: float, upper: float) -> jnp.ndarray:
    return lower + (upper - lower) * jax.nn.sigmoid(raw)


def _inverse_sigmoid_bounds(value: float, lower: float, upper: float) -> float:
    clipped = np.clip((value - lower) / (upper - lower), 1.0e-5, 1.0 - 1.0e-5)
    return float(np.log(clipped / (1.0 - clipped)))


def _logit_unit(value: float) -> float:
    clipped = np.clip(value, 1.0e-5, 1.0 - 1.0e-5)
    return float(np.log(clipped / (1.0 - clipped)))


def initial_raw_params(surface: SurfaceDefinition, config: OptimizationConfig) -> jnp.ndarray:
    start = np.asarray(surface.start_uv, dtype=np.float32)
    end = np.asarray(surface.end_uv, dtype=np.float32)
    p1 = start + (end - start) / 3.0
    p2 = start + 2.0 * (end - start) / 3.0
    if surface.kind == "plate_with_hole":
        p1 = p1 + np.array([0.00, -0.06], dtype=np.float32)
        p2 = p2 + np.array([0.00, 0.06], dtype=np.float32)
    initial_thickness = 0.85 * config.max_thickness
    raw = [
        _logit_unit(float(p1[0])),
        _logit_unit(float(p1[1])),
        _logit_unit(float(p2[0])),
        _logit_unit(float(p2[1])),
        _inverse_sigmoid_bounds(initial_thickness, *config.thickness_scale_bounds),
    ]
    return jnp.asarray(raw, dtype=jnp.float32)


def control_points_from_raw(
    raw_params: jnp.ndarray,
    surface: SurfaceDefinition,
    config: OptimizationConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    p1 = jax.nn.sigmoid(raw_params[:2])
    p2 = jax.nn.sigmoid(raw_params[2:4])
    thickness_scale = _sigmoid_bounds(raw_params[4], *config.thickness_scale_bounds)
    control_points = jnp.stack(
        [
            jnp.asarray(surface.start_uv, dtype=jnp.float32),
            p1,
            p2,
            jnp.asarray(surface.end_uv, dtype=jnp.float32),
        ],
        axis=0,
    )
    return control_points, thickness_scale


def sample_cubic_bezier(control_points_uv: jnp.ndarray, samples: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    t = jnp.linspace(0.0, 1.0, samples)
    omt = 1.0 - t
    points = (
        (omt**3)[:, None] * control_points_uv[0]
        + (3.0 * omt**2 * t)[:, None] * control_points_uv[1]
        + (3.0 * omt * t**2)[:, None] * control_points_uv[2]
        + (t**3)[:, None] * control_points_uv[3]
    )
    return t, points


def _central_first(values: jnp.ndarray, dt: float) -> jnp.ndarray:
    middle = (values[2:] - values[:-2]) / (2.0 * dt)
    start = ((values[1] - values[0]) / dt)[None, :]
    end = ((values[-1] - values[-2]) / dt)[None, :]
    return jnp.concatenate([start, middle, end], axis=0)


def _central_second(values: jnp.ndarray, dt: float) -> jnp.ndarray:
    middle = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / (dt**2)
    return jnp.concatenate([middle[:1], middle, middle[-1:]], axis=0)


def _curve_metrics(xyz: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dt = 1.0 / max(xyz.shape[0] - 1, 1)
    first = _central_first(xyz, dt)
    second = _central_second(xyz, dt)
    tangents = _normalize(first)
    curvature = jnp.linalg.norm(jnp.cross(first, second), axis=-1) / jnp.maximum(
        jnp.linalg.norm(first, axis=-1) ** 3,
        1.0e-8,
    )
    radius = 1.0 / jnp.maximum(curvature, 1.0e-6)
    segment_lengths = jnp.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    total_length = jnp.sum(segment_lengths)
    return tangents, radius, total_length


def evaluate_raw_design(
    raw_params: jnp.ndarray,
    surface: SurfaceDefinition,
    load_case: LoadCase,
    config: OptimizationConfig,
) -> dict[str, jnp.ndarray]:
    control_points, thickness_scale = control_points_from_raw(raw_params, surface, config)
    _, uv_path = sample_cubic_bezier(control_points, config.num_path_samples)
    xyz_path = surface_xyz(surface, uv_path)
    normals = surface_normals(surface, uv_path)
    tangents, radius_profile_m, total_length_m = _curve_metrics(xyz_path)

    preferred = preferred_fiber_direction(surface, uv_path, load_case.direction_xyz)
    alignment = jnp.abs(jnp.sum(tangents * preferred, axis=-1))
    stress = stress_weight(surface, uv_path)
    stiffness_proxy = thickness_scale * jnp.mean(stress * (0.25 + 0.75 * alignment))
    compliance_proxy = (load_case.magnitude_n / 500.0) / jnp.maximum(stiffness_proxy, 1.0e-4)

    chord_length = jnp.linalg.norm(xyz_path[-1] - xyz_path[0]) + 1.0e-6
    length_ratio = total_length_m / chord_length

    steer_loss = steering_penalty(radius_profile_m, config.min_steering_radius_m)
    thick_loss, thickness_field_map, thickness_stats = thickness_penalty(surface, uv_path, thickness_scale, config)
    keepout_loss, keepout_clearance = keepout_penalty(surface, uv_path)
    bound_loss = boundary_penalty(uv_path)
    smooth_loss = smoothness_penalty(control_points)

    total_loss = (
        config.structural_weight * compliance_proxy
        + config.length_weight * length_ratio
        + config.steering_weight * steer_loss
        + config.thickness_weight * thick_loss
        + config.keepout_weight * keepout_loss
        + config.boundary_weight * bound_loss
        + config.smoothness_weight * smooth_loss
    )

    return {
        "loss": total_loss,
        "control_points_uv": control_points,
        "thickness_scale": thickness_scale,
        "path_uv": uv_path,
        "path_xyz": xyz_path,
        "normals": normals,
        "tangents": tangents,
        "radius_profile_m": radius_profile_m,
        "stress_weight": stress,
        "alignment": alignment,
        "thickness_field": thickness_field_map,
        "stiffness_proxy": stiffness_proxy,
        "compliance_proxy": compliance_proxy,
        "path_length_m": total_length_m,
        "length_ratio": length_ratio,
        "steering_penalty": steer_loss,
        "thickness_penalty": thick_loss,
        "keepout_penalty": keepout_loss,
        "boundary_penalty": bound_loss,
        "smoothness_penalty": smooth_loss,
        "peak_thickness": thickness_stats["peak_thickness"],
        "mean_thickness": thickness_stats["mean_thickness"],
        "thickness_std": thickness_stats["thickness_std"],
        "min_steering_radius_m": jnp.min(radius_profile_m),
        "minimum_keepout_clearance_uv": jnp.min(keepout_clearance),
    }


def _scalar_history_entry(step: int, aux: dict[str, jnp.ndarray]) -> dict[str, float]:
    return {
        "step": float(step),
        "loss": float(aux["loss"]),
        "compliance_proxy": float(aux["compliance_proxy"]),
        "stiffness_proxy": float(aux["stiffness_proxy"]),
        "path_length_m": float(aux["path_length_m"]),
        "min_steering_radius_m": float(aux["min_steering_radius_m"]),
        "peak_thickness": float(aux["peak_thickness"]),
        "minimum_keepout_clearance_uv": float(aux["minimum_keepout_clearance_uv"]),
        "steering_penalty": float(aux["steering_penalty"]),
        "thickness_penalty": float(aux["thickness_penalty"]),
        "keepout_penalty": float(aux["keepout_penalty"]),
    }


def _serialize_mapping(mapping: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            result[key] = _serialize_mapping(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [item for item in value]
        elif isinstance(value, (np.ndarray, jnp.ndarray)):
            result[key] = np.asarray(value).tolist()
        elif hasattr(value, "__fspath__"):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def optimize_ifp_path(
    surface: SurfaceDefinition,
    load_case: LoadCase | None = None,
    config: OptimizationConfig | None = None,
) -> dict[str, object]:
    load_case = load_case or LoadCase()
    config = config or OptimizationConfig()

    raw_params = initial_raw_params(surface, config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(config.learning_rate),
    )
    opt_state = optimizer.init(raw_params)

    best_params = raw_params
    best_loss = jnp.inf
    history: list[dict[str, float]] = []

    def loss_fn(params: jnp.ndarray) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        aux = evaluate_raw_design(params, surface, load_case, config)
        return aux["loss"], aux

    for step in range(config.steps):
        current_params = raw_params
        (loss_value, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_params)
        updates, opt_state = optimizer.update(grads, opt_state, current_params)
        raw_params = optax.apply_updates(current_params, updates)

        if loss_value < best_loss:
            best_loss = loss_value
            best_params = current_params

        if step % config.history_stride == 0 or step == config.steps - 1:
            history.append(_scalar_history_entry(step, aux))

    final = evaluate_raw_design(best_params, surface, load_case, config)
    metrics = {
        "objective": float(final["loss"]),
        "stiffness_proxy": float(final["stiffness_proxy"]),
        "compliance_proxy": float(final["compliance_proxy"]),
        "path_length_m": float(final["path_length_m"]),
        "length_ratio": float(final["length_ratio"]),
        "min_steering_radius_m": float(final["min_steering_radius_m"]),
        "min_steering_radius_mm": 1000.0 * float(final["min_steering_radius_m"]),
        "radius_limit_mm": 1000.0 * config.min_steering_radius_m,
        "peak_thickness": float(final["peak_thickness"]),
        "mean_thickness": float(final["mean_thickness"]),
        "thickness_std": float(final["thickness_std"]),
        "thickness_limit": config.max_thickness,
        "minimum_keepout_clearance_uv": float(final["minimum_keepout_clearance_uv"]),
        "thickness_scale": float(final["thickness_scale"]),
    }
    metrics["manufacturable"] = bool(
        metrics["min_steering_radius_m"] >= config.min_steering_radius_m
        and metrics["peak_thickness"] <= config.max_thickness
        and metrics["minimum_keepout_clearance_uv"] >= 0.0
    )

    objective_terms = {
        "loss": float(final["loss"]),
        "compliance_proxy": float(final["compliance_proxy"]),
        "stiffness_proxy": float(final["stiffness_proxy"]),
        "length_ratio": float(final["length_ratio"]),
        "steering_penalty": float(final["steering_penalty"]),
        "thickness_penalty": float(final["thickness_penalty"]),
        "keepout_penalty": float(final["keepout_penalty"]),
        "boundary_penalty": float(final["boundary_penalty"]),
        "smoothness_penalty": float(final["smoothness_penalty"]),
    }

    return {
        "surface": surface.as_dict(),
        "load_case": asdict(load_case),
        "optimization_config": _serialize_mapping(asdict(config)),
        "control_points_uv": np.asarray(final["control_points_uv"]).tolist(),
        "path_uv": np.asarray(final["path_uv"]).tolist(),
        "path_xyz": np.asarray(final["path_xyz"]).tolist(),
        "normals": np.asarray(final["normals"]).tolist(),
        "tangents": np.asarray(final["tangents"]).tolist(),
        "radius_profile_m": np.asarray(final["radius_profile_m"]).tolist(),
        "stress_weight": np.asarray(final["stress_weight"]).tolist(),
        "alignment": np.asarray(final["alignment"]).tolist(),
        "thickness_field": np.asarray(final["thickness_field"]).tolist(),
        "objective_terms": objective_terms,
        "metrics": metrics,
        "history": history,
    }
