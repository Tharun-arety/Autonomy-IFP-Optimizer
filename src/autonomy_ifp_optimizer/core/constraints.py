from __future__ import annotations

import jax
import jax.numpy as jnp

from ..config import OptimizationConfig
from .geometry import SurfaceDefinition, keepout_signed_distance


def boundary_penalty(uv: jnp.ndarray) -> jnp.ndarray:
    lower = jax.nn.relu(-uv)
    upper = jax.nn.relu(uv - 1.0)
    return jnp.mean((lower + upper) ** 2)


def keepout_penalty(surface: SurfaceDefinition, uv: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    clearance = keepout_signed_distance(surface, uv)
    penalty = jnp.mean(jax.nn.softplus(-clearance * 40.0) / 40.0)
    return penalty, clearance


def steering_penalty(radius_profile_m: jnp.ndarray, min_radius_m: float) -> jnp.ndarray:
    deficit = jax.nn.relu(min_radius_m - radius_profile_m)
    scaled = deficit / jnp.maximum(min_radius_m, 1.0e-6)
    return jnp.mean(jnp.expm1(4.0 * scaled))


def smoothness_penalty(control_points_uv: jnp.ndarray) -> jnp.ndarray:
    second_difference = control_points_uv[:-2] - 2.0 * control_points_uv[1:-1] + control_points_uv[2:]
    return jnp.mean(jnp.sum(second_difference**2, axis=-1))


def thickness_field(
    uv_path: jnp.ndarray,
    thickness_scale: jnp.ndarray,
    config: OptimizationConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    lin = jnp.linspace(0.0, 1.0, config.coverage_grid)
    uu, vv = jnp.meshgrid(lin, lin, indexing="xy")
    grid = jnp.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1)
    distances = jnp.linalg.norm(grid[:, None, :] - uv_path[None, :, :], axis=-1)
    kernels = jnp.exp(-0.5 * (distances / config.tow_half_width_uv) ** 2)
    field = thickness_scale * jnp.sum(kernels, axis=1) / uv_path.shape[0]
    return field.reshape(config.coverage_grid, config.coverage_grid), grid


def thickness_penalty(
    surface: SurfaceDefinition,
    uv_path: jnp.ndarray,
    thickness_scale: jnp.ndarray,
    config: OptimizationConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    field, grid = thickness_field(uv_path, thickness_scale, config)
    flat_grid = grid.reshape(-1, 2)
    keepout_clearance = keepout_signed_distance(surface, flat_grid)
    valid_mask = (keepout_clearance >= 0.0).reshape(config.coverage_grid, config.coverage_grid)
    mask = valid_mask.astype(jnp.float32)
    valid_count = jnp.maximum(jnp.sum(mask), 1.0)
    masked_field = field * mask
    mean_thickness = jnp.sum(masked_field) / valid_count
    variance = jnp.sum(((field - mean_thickness) ** 2) * mask) / valid_count
    excess = jax.nn.relu(field - config.max_thickness) * mask
    penalty = jnp.sum(excess**2) / valid_count + 0.25 * variance
    stats = {
        "peak_thickness": jnp.max(masked_field),
        "mean_thickness": mean_thickness,
        "thickness_std": jnp.sqrt(variance),
    }
    return penalty, masked_field, stats
