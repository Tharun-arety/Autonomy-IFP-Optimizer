from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import trimesh

from ..config import GeometryConfig


@dataclass(frozen=True)
class KeepOutZone:
    center_uv: tuple[float, float]
    radius_uv: float
    label: str = "keepout"


@dataclass(frozen=True)
class SurfaceDefinition:
    name: str
    kind: str
    params: dict[str, float]
    start_uv: tuple[float, float]
    end_uv: tuple[float, float]
    keep_outs: tuple[KeepOutZone, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "params": self.params,
            "start_uv": list(self.start_uv),
            "end_uv": list(self.end_uv),
            "keep_outs": [
                {
                    "center_uv": list(zone.center_uv),
                    "radius_uv": zone.radius_uv,
                    "label": zone.label,
                }
                for zone in self.keep_outs
            ],
        }


def _normalize(vector: jnp.ndarray, eps: float = 1.0e-8) -> jnp.ndarray:
    norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / jnp.maximum(norm, eps)


def make_plate_with_hole(config: GeometryConfig | None = None, name: str = "drone_plate") -> SurfaceDefinition:
    config = config or GeometryConfig(surface="plate_with_hole")
    return SurfaceDefinition(
        name=name,
        kind="plate_with_hole",
        params={
            "length_m": float(config.length_m),
            "width_m": float(config.width_m),
            "crown_m": float(config.crown_m),
        },
        start_uv=(0.08, 0.24),
        end_uv=(0.92, 0.76),
        keep_outs=(
            KeepOutZone(
                center_uv=tuple(config.hole_center_uv),
                radius_uv=float(config.hole_radius_uv),
                label="central_cutout",
            ),
        ),
    )


def make_cylinder(config: GeometryConfig | None = None, name: str = "robotic_limb") -> SurfaceDefinition:
    config = config or GeometryConfig(surface="cylinder")
    return SurfaceDefinition(
        name=name,
        kind="cylinder",
        params={
            "radius_m": float(config.cylinder_radius_m),
            "height_m": float(config.cylinder_height_m),
        },
        start_uv=(0.08, 0.16),
        end_uv=(0.92, 0.84),
        keep_outs=(),
    )


def _infer_surface_from_mesh(mesh_path: Path, mesh: trimesh.Trimesh, config: GeometryConfig | None) -> SurfaceDefinition:
    config = config or GeometryConfig()
    stem = mesh_path.stem.lower()
    extents = mesh.bounding_box.extents
    if any(token in stem for token in ("tube", "limb", "cyl", "robot")):
        radius = 0.25 * float(np.mean(sorted(extents[:2])))
        height = float(extents[2] if extents[2] > 0 else config.cylinder_height_m)
        return make_cylinder(
            GeometryConfig(
                surface="cylinder",
                cylinder_radius_m=max(radius, 0.05),
                cylinder_height_m=max(height, 0.2),
            ),
            name=mesh_path.stem,
        )
    return make_plate_with_hole(
        GeometryConfig(
            surface="plate_with_hole",
            length_m=float(extents[0] if extents[0] > 0 else config.length_m),
            width_m=float(extents[1] if extents[1] > 0 else config.width_m),
            crown_m=float(config.crown_m),
            hole_center_uv=config.hole_center_uv,
            hole_radius_uv=config.hole_radius_uv,
        ),
        name=mesh_path.stem,
    )


def load_surface(
    mesh: str | Path | None = None,
    surface: str | None = None,
    geometry_config: GeometryConfig | None = None,
) -> SurfaceDefinition:
    geometry_config = geometry_config or GeometryConfig()
    if surface == "cylinder":
        return make_cylinder(geometry_config)
    if surface in {"plate", "plate_with_hole"}:
        return make_plate_with_hole(geometry_config)
    if mesh is None:
        return make_plate_with_hole(geometry_config)

    mesh_path = Path(mesh)
    if mesh_path.exists():
        loaded = trimesh.load_mesh(mesh_path, process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"Unsupported mesh type for {mesh_path}")
        return _infer_surface_from_mesh(mesh_path, loaded, geometry_config)

    lowered = str(mesh).lower()
    if any(token in lowered for token in ("cyl", "tube", "limb", "robot")):
        return make_cylinder(geometry_config, name=mesh_path.stem or "robotic_limb")
    return make_plate_with_hole(geometry_config, name=mesh_path.stem or "drone_plate")


def surface_xyz(surface: SurfaceDefinition, uv: jnp.ndarray) -> jnp.ndarray:
    u = uv[..., 0]
    v = uv[..., 1]
    if surface.kind == "cylinder":
        radius = surface.params["radius_m"]
        height = surface.params["height_m"]
        theta = 2.0 * jnp.pi * u
        x = radius * jnp.cos(theta)
        y = radius * jnp.sin(theta)
        z = height * (v - 0.5)
        return jnp.stack([x, y, z], axis=-1)

    length = surface.params["length_m"]
    width = surface.params["width_m"]
    crown = surface.params["crown_m"]
    x = length * (u - 0.5)
    y = width * (v - 0.5)
    r2 = (u - 0.5) ** 2 + (v - 0.5) ** 2
    z = crown * jnp.exp(-8.0 * r2)
    return jnp.stack([x, y, z], axis=-1)


def surface_plane_coordinates(surface: SurfaceDefinition, uv: jnp.ndarray) -> jnp.ndarray:
    u = uv[..., 0]
    v = uv[..., 1]
    if surface.kind == "cylinder":
        radius = surface.params["radius_m"]
        height = surface.params["height_m"]
        s = 2.0 * jnp.pi * radius * (u - 0.5)
        z = height * (v - 0.5)
        return jnp.stack([s, z], axis=-1)

    length = surface.params["length_m"]
    width = surface.params["width_m"]
    x = length * (u - 0.5)
    y = width * (v - 0.5)
    return jnp.stack([x, y], axis=-1)


def load_direction_in_plane(surface: SurfaceDefinition, load_direction_xyz: tuple[float, float, float]) -> jnp.ndarray:
    direction = _normalize(jnp.asarray(load_direction_xyz, dtype=jnp.float32))
    if surface.kind == "cylinder":
        circumferential = jnp.linalg.norm(direction[:2])
        planar = jnp.asarray([circumferential, direction[2]], dtype=jnp.float32)
    else:
        planar = direction[:2]
    norm = jnp.linalg.norm(planar)
    fallback = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    return jnp.where(norm > 1.0e-6, planar / norm, fallback)


def surface_partials(surface: SurfaceDefinition, uv: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    u = uv[..., 0]
    v = uv[..., 1]
    if surface.kind == "cylinder":
        radius = surface.params["radius_m"]
        height = surface.params["height_m"]
        theta = 2.0 * jnp.pi * u
        du = jnp.stack(
            [
                -2.0 * jnp.pi * radius * jnp.sin(theta),
                2.0 * jnp.pi * radius * jnp.cos(theta),
                jnp.zeros_like(u),
            ],
            axis=-1,
        )
        dv = jnp.stack(
            [
                jnp.zeros_like(u),
                jnp.zeros_like(u),
                jnp.full_like(u, height),
            ],
            axis=-1,
        )
        return du, dv

    length = surface.params["length_m"]
    width = surface.params["width_m"]
    crown = surface.params["crown_m"]
    exp_term = jnp.exp(-8.0 * ((u - 0.5) ** 2 + (v - 0.5) ** 2))
    dz_du = crown * exp_term * (-16.0 * (u - 0.5))
    dz_dv = crown * exp_term * (-16.0 * (v - 0.5))
    du = jnp.stack(
        [
            jnp.full_like(u, length),
            jnp.zeros_like(u),
            dz_du,
        ],
        axis=-1,
    )
    dv = jnp.stack(
        [
            jnp.zeros_like(u),
            jnp.full_like(u, width),
            dz_dv,
        ],
        axis=-1,
    )
    return du, dv


def surface_normals(surface: SurfaceDefinition, uv: jnp.ndarray) -> jnp.ndarray:
    du, dv = surface_partials(surface, uv)
    return _normalize(jnp.cross(du, dv))


def keepout_signed_distance(surface: SurfaceDefinition, uv: jnp.ndarray) -> jnp.ndarray:
    if not surface.keep_outs:
        return jnp.full(uv.shape[:-1], 1.0e3, dtype=jnp.float32)
    distances = []
    for zone in surface.keep_outs:
        center = jnp.asarray(zone.center_uv, dtype=jnp.float32)
        distances.append(jnp.linalg.norm(uv - center, axis=-1) - zone.radius_uv)
    return jnp.min(jnp.stack(distances, axis=0), axis=0)


def preview_surface(surface: SurfaceDefinition, resolution_u: int = 72, resolution_v: int = 48) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 1.0, resolution_u)
    v = np.linspace(0.0, 1.0, resolution_v)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    uv = jnp.stack([jnp.asarray(uu).reshape(-1), jnp.asarray(vv).reshape(-1)], axis=-1)
    xyz = np.asarray(surface_xyz(surface, uv))

    faces: list[list[int]] = []
    for j in range(resolution_v - 1):
        for i in range(resolution_u - 1):
            a = j * resolution_u + i
            b = a + 1
            c = a + resolution_u
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return xyz, np.asarray(faces, dtype=np.int32)


def preview_keepouts(surface: SurfaceDefinition, points: int = 128) -> list[np.ndarray]:
    curves: list[np.ndarray] = []
    theta = np.linspace(0.0, 2.0 * np.pi, points, endpoint=True)
    for zone in surface.keep_outs:
        circle_uv = np.stack(
            [
                zone.center_uv[0] + zone.radius_uv * np.cos(theta),
                zone.center_uv[1] + zone.radius_uv * np.sin(theta),
            ],
            axis=-1,
        )
        curves.append(np.asarray(surface_xyz(surface, jnp.asarray(circle_uv, dtype=jnp.float32))))
    return curves
