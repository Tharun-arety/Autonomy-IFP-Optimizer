from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..config import LoadCase, OptimizationConfig
from .geometry import SurfaceDefinition, load_direction_in_plane
from .meshing import build_surface_mesh


@dataclass(frozen=True)
class MembraneFEMModel:
    node_uv: jnp.ndarray
    node_xy_m: jnp.ndarray
    element_nodes: jnp.ndarray
    element_dof_indices: jnp.ndarray
    element_centers_uv: jnp.ndarray
    element_b_matrices: jnp.ndarray
    element_areas_m2: jnp.ndarray
    active_elements: jnp.ndarray
    active_nodes: jnp.ndarray
    free_dofs: jnp.ndarray
    fixed_dofs: jnp.ndarray
    force_vector_n: jnp.ndarray
    load_direction_xy: jnp.ndarray
    loaded_node_indices: jnp.ndarray
    matrix_constitutive_pa: jnp.ndarray
    orthotropic_constitutive_pa: jnp.ndarray
    reference_compliance_n_m: float
    mesh_node_count: int
    mesh_element_count: int


def _normalize_xy(vector: jnp.ndarray, eps: float = 1.0e-8) -> jnp.ndarray:
    norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / jnp.maximum(norm, eps)


def _curve_tangents_xy(path_xy_m: jnp.ndarray) -> jnp.ndarray:
    first = jnp.zeros_like(path_xy_m)
    first = first.at[1:-1].set(0.5 * (path_xy_m[2:] - path_xy_m[:-2]))
    first = first.at[0].set(path_xy_m[1] - path_xy_m[0])
    first = first.at[-1].set(path_xy_m[-1] - path_xy_m[-2])
    return _normalize_xy(first)


def _isotropic_plane_stress_q(modulus_pa: float, poisson: float) -> jnp.ndarray:
    scale = modulus_pa / max(1.0 - poisson**2, 1.0e-8)
    return jnp.asarray(
        [
            [scale, scale * poisson, 0.0],
            [scale * poisson, scale, 0.0],
            [0.0, 0.0, scale * (1.0 - poisson) * 0.5],
        ],
        dtype=jnp.float32,
    )


def _orthotropic_plane_stress_q(
    e1_pa: float,
    e2_pa: float,
    g12_pa: float,
    v12: float,
) -> jnp.ndarray:
    v21 = v12 * e2_pa / max(e1_pa, 1.0e-8)
    denom = max(1.0 - v12 * v21, 1.0e-8)
    q11 = e1_pa / denom
    q22 = e2_pa / denom
    q12 = v12 * e2_pa / denom
    return jnp.asarray(
        [
            [q11, q12, 0.0],
            [q12, q22, 0.0],
            [0.0, 0.0, g12_pa],
        ],
        dtype=jnp.float32,
    )


def _rotate_plane_stress_q(local_q_pa: jnp.ndarray, direction_xy: jnp.ndarray) -> jnp.ndarray:
    q11 = local_q_pa[0, 0]
    q22 = local_q_pa[1, 1]
    q12 = local_q_pa[0, 1]
    q66 = local_q_pa[2, 2]
    direction_xy = _normalize_xy(direction_xy)
    m = direction_xy[:, 0]
    n = direction_xy[:, 1]
    m2 = m * m
    n2 = n * n
    m4 = m2 * m2
    n4 = n2 * n2

    qbar11 = q11 * m4 + 2.0 * (q12 + 2.0 * q66) * m2 * n2 + q22 * n4
    qbar22 = q11 * n4 + 2.0 * (q12 + 2.0 * q66) * m2 * n2 + q22 * m4
    qbar12 = (q11 + q22 - 4.0 * q66) * m2 * n2 + q12 * (m4 + n4)
    qbar16 = (q11 - q12 - 2.0 * q66) * m * m2 * n - (q22 - q12 - 2.0 * q66) * m * n2 * n
    qbar26 = (q11 - q12 - 2.0 * q66) * m * n2 * n - (q22 - q12 - 2.0 * q66) * m * m2 * n
    qbar66 = (q11 + q22 - 2.0 * q12 - 2.0 * q66) * m2 * n2 + q66 * (m4 + n4)

    return jnp.stack(
        [
            jnp.stack([qbar11, qbar12, qbar16], axis=-1),
            jnp.stack([qbar12, qbar22, qbar26], axis=-1),
            jnp.stack([qbar16, qbar26, qbar66], axis=-1),
        ],
        axis=-2,
    )


def _select_edges(load_xy: np.ndarray) -> tuple[str, str]:
    if abs(load_xy[0]) >= abs(load_xy[1]):
        return ("left", "right") if load_xy[0] >= 0.0 else ("right", "left")
    return ("bottom", "top") if load_xy[1] >= 0.0 else ("top", "bottom")


def _build_force_vector(
    node_xy_m: np.ndarray,
    boundary_segments: np.ndarray,
    load_direction_xy: np.ndarray,
    load_magnitude_n: float,
    ndof: int,
) -> np.ndarray:
    force = np.zeros(ndof, dtype=np.float32)
    if boundary_segments.size == 0:
        return force

    segment_vectors = node_xy_m[boundary_segments[:, 1]] - node_xy_m[boundary_segments[:, 0]]
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_edge_length = max(float(np.sum(segment_lengths)), 1.0e-8)
    traction = load_magnitude_n * load_direction_xy / total_edge_length

    for (start, end), length in zip(boundary_segments, segment_lengths):
        nodal_force = 0.5 * traction * float(length)
        force[2 * start : 2 * start + 2] += nodal_force
        force[2 * end : 2 * end + 2] += nodal_force
    return force


def _assemble_global_stiffness(
    model: MembraneFEMModel,
    element_membrane_matrix_n_per_m: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    bt = jnp.swapaxes(model.element_b_matrices, 1, 2)
    element_stiffness = jnp.einsum(
        "eia,eab,ebj,e->eij",
        bt,
        element_membrane_matrix_n_per_m,
        model.element_b_matrices,
        model.element_areas_m2,
    )
    ndof = int(model.node_uv.shape[0]) * 2
    stiffness = jnp.zeros((ndof, ndof), dtype=jnp.float32)
    rows = model.element_dof_indices[:, :, None]
    cols = model.element_dof_indices[:, None, :]
    stiffness = stiffness.at[rows, cols].add(element_stiffness)
    return stiffness, element_stiffness


def _solve_displacements(
    model: MembraneFEMModel,
    element_membrane_matrix_n_per_m: jnp.ndarray,
    regularization: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    stiffness, element_stiffness = _assemble_global_stiffness(model, element_membrane_matrix_n_per_m)
    free = model.free_dofs
    reduced = stiffness[free[:, None], free[None, :]]
    force = model.force_vector_n[free]
    diagonal_scale = jnp.maximum(jnp.mean(jnp.diag(reduced)), 1.0)
    reg = jnp.maximum(regularization * diagonal_scale, 1.0e-9)
    reduced = reduced + reg * jnp.eye(reduced.shape[0], dtype=jnp.float32)
    displacement_free = jnp.linalg.solve(reduced, force)
    displacement = jnp.zeros(stiffness.shape[0], dtype=jnp.float32).at[free].set(displacement_free)
    residual = reduced @ displacement_free - force
    return displacement, element_stiffness, residual


def prepare_membrane_fem_model(
    surface: SurfaceDefinition,
    load_case: LoadCase,
    config: OptimizationConfig,
) -> MembraneFEMModel:
    surface_mesh = build_surface_mesh(surface, config)
    load_xy = np.asarray(load_direction_in_plane(surface, load_case.direction_xyz), dtype=np.float32)
    fixed_side, loaded_side = _select_edges(load_xy)

    fixed_segments = surface_mesh.boundary_segments.get(fixed_side, np.zeros((0, 2), dtype=np.int32))
    loaded_segments = surface_mesh.boundary_segments.get(loaded_side, np.zeros((0, 2), dtype=np.int32))
    if fixed_segments.size == 0 or loaded_segments.size == 0:
        raise ValueError(
            f"Missing boundary segments for {surface.name}: fixed='{fixed_side}' loaded='{loaded_side}'"
        )

    fixed_nodes = np.unique(fixed_segments.reshape(-1)).astype(np.int32)
    loaded_nodes = np.unique(loaded_segments.reshape(-1)).astype(np.int32)

    ndof = surface_mesh.node_uv.shape[0] * 2
    force_vector = _build_force_vector(
        surface_mesh.node_xy_m,
        loaded_segments,
        load_xy,
        load_case.magnitude_n,
        ndof,
    )

    fixed_dofs = np.sort(np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1]).astype(np.int32))
    all_nodes = np.arange(surface_mesh.node_uv.shape[0], dtype=np.int32)
    active_dofs = np.sort(np.concatenate([2 * all_nodes, 2 * all_nodes + 1]).astype(np.int32))
    free_dofs = np.setdiff1d(active_dofs, fixed_dofs, assume_unique=False).astype(np.int32)

    dof_indices = np.zeros((surface_mesh.element_nodes.shape[0], 6), dtype=np.int32)
    for index, nodes in enumerate(surface_mesh.element_nodes):
        dof_indices[index] = np.asarray(
            [
                2 * nodes[0],
                2 * nodes[0] + 1,
                2 * nodes[1],
                2 * nodes[1] + 1,
                2 * nodes[2],
                2 * nodes[2] + 1,
            ],
            dtype=np.int32,
        )

    matrix_q = _isotropic_plane_stress_q(config.matrix_modulus_pa, config.matrix_poisson)
    ortho_q = _orthotropic_plane_stress_q(
        config.fiber_modulus_longitudinal_pa,
        config.fiber_modulus_transverse_pa,
        config.fiber_shear_modulus_pa,
        config.fiber_poisson,
    )

    provisional_model = MembraneFEMModel(
        node_uv=jnp.asarray(surface_mesh.node_uv, dtype=jnp.float32),
        node_xy_m=jnp.asarray(surface_mesh.node_xy_m, dtype=jnp.float32),
        element_nodes=jnp.asarray(surface_mesh.element_nodes, dtype=jnp.int32),
        element_dof_indices=jnp.asarray(dof_indices, dtype=jnp.int32),
        element_centers_uv=jnp.asarray(surface_mesh.element_centers_uv, dtype=jnp.float32),
        element_b_matrices=jnp.asarray(surface_mesh.element_b_matrices, dtype=jnp.float32),
        element_areas_m2=jnp.asarray(surface_mesh.element_areas_m2, dtype=jnp.float32),
        active_elements=jnp.ones(surface_mesh.element_nodes.shape[0], dtype=jnp.float32),
        active_nodes=jnp.ones(surface_mesh.node_uv.shape[0], dtype=jnp.float32),
        free_dofs=jnp.asarray(free_dofs, dtype=jnp.int32),
        fixed_dofs=jnp.asarray(fixed_dofs, dtype=jnp.int32),
        force_vector_n=jnp.asarray(force_vector, dtype=jnp.float32),
        load_direction_xy=jnp.asarray(load_xy, dtype=jnp.float32),
        loaded_node_indices=jnp.asarray(loaded_nodes, dtype=jnp.int32),
        matrix_constitutive_pa=matrix_q,
        orthotropic_constitutive_pa=ortho_q,
        reference_compliance_n_m=1.0,
        mesh_node_count=int(surface_mesh.node_uv.shape[0]),
        mesh_element_count=int(surface_mesh.element_nodes.shape[0]),
    )

    base_membrane = config.base_laminate_thickness_m * provisional_model.matrix_constitutive_pa[None, :, :]
    reference_displacement, _, _ = _solve_displacements(provisional_model, base_membrane, config.fem_regularization)
    reference_compliance = float(jnp.maximum(jnp.dot(provisional_model.force_vector_n, reference_displacement), 1.0e-8))

    return MembraneFEMModel(
        node_uv=provisional_model.node_uv,
        node_xy_m=provisional_model.node_xy_m,
        element_nodes=provisional_model.element_nodes,
        element_dof_indices=provisional_model.element_dof_indices,
        element_centers_uv=provisional_model.element_centers_uv,
        element_b_matrices=provisional_model.element_b_matrices,
        element_areas_m2=provisional_model.element_areas_m2,
        active_elements=provisional_model.active_elements,
        active_nodes=provisional_model.active_nodes,
        free_dofs=provisional_model.free_dofs,
        fixed_dofs=provisional_model.fixed_dofs,
        force_vector_n=provisional_model.force_vector_n,
        load_direction_xy=provisional_model.load_direction_xy,
        loaded_node_indices=provisional_model.loaded_node_indices,
        matrix_constitutive_pa=provisional_model.matrix_constitutive_pa,
        orthotropic_constitutive_pa=provisional_model.orthotropic_constitutive_pa,
        reference_compliance_n_m=reference_compliance,
        mesh_node_count=provisional_model.mesh_node_count,
        mesh_element_count=provisional_model.mesh_element_count,
    )


def solve_membrane_response(
    path_uv: jnp.ndarray,
    path_xy_m: jnp.ndarray,
    thickness_scale: jnp.ndarray,
    model: MembraneFEMModel,
    config: OptimizationConfig,
) -> dict[str, jnp.ndarray]:
    path_tangents_xy = _curve_tangents_xy(path_xy_m)
    distances = jnp.linalg.norm(model.element_centers_uv[:, None, :] - path_uv[None, :, :], axis=-1)
    kernels = jnp.exp(-0.5 * (distances / jnp.maximum(config.tow_half_width_uv, 1.0e-6)) ** 2)
    orientation_raw = jnp.einsum("ep,pd->ed", kernels, path_tangents_xy)
    fallback = jnp.broadcast_to(model.load_direction_xy, orientation_raw.shape)
    fiber_direction_xy = _normalize_xy(orientation_raw + 1.0e-4 * fallback)
    fiber_angle_rad = jnp.arctan2(fiber_direction_xy[:, 1], fiber_direction_xy[:, 0])

    softmin_temperature = jnp.maximum(0.35 * config.tow_half_width_uv, 1.0e-4)
    proximity_weights = jax.nn.softmax(-distances / softmin_temperature, axis=-1)
    minimum_distance = jnp.sum(proximity_weights * distances, axis=-1)
    fiber_presence = jnp.exp(-0.5 * (minimum_distance / jnp.maximum(config.tow_half_width_uv, 1.0e-6)) ** 2)
    fiber_thickness_m = config.fiber_layer_thickness_m * thickness_scale * fiber_presence
    membrane_matrix = (
        config.base_laminate_thickness_m * model.matrix_constitutive_pa[None, :, :]
        + fiber_thickness_m[:, None, None] * _rotate_plane_stress_q(model.orthotropic_constitutive_pa, fiber_direction_xy)
    )

    displacement_vector, element_stiffness, residual = _solve_displacements(model, membrane_matrix, config.fem_regularization)
    nodal_displacement_xy = displacement_vector.reshape(-1, 2)
    nodal_displacement_magnitude = jnp.linalg.norm(nodal_displacement_xy, axis=-1) * model.active_nodes
    compliance_n_m = jnp.dot(model.force_vector_n, displacement_vector)
    normalized_compliance = compliance_n_m / jnp.maximum(model.reference_compliance_n_m, 1.0e-8)

    element_displacements = displacement_vector[model.element_dof_indices]
    center_strain = jnp.einsum("eab,eb->ea", model.element_b_matrices, element_displacements)
    total_thickness_m = config.base_laminate_thickness_m + fiber_thickness_m
    effective_q = membrane_matrix / jnp.maximum(total_thickness_m[:, None, None], 1.0e-12)
    element_stress_pa = jnp.einsum("eab,eb->ea", effective_q, center_strain)
    von_mises_pa = jnp.sqrt(
        jnp.maximum(
            element_stress_pa[:, 0] ** 2
            - element_stress_pa[:, 0] * element_stress_pa[:, 1]
            + element_stress_pa[:, 1] ** 2
            + 3.0 * element_stress_pa[:, 2] ** 2,
            0.0,
        )
    )
    element_strain_energy_n_m = (
        0.5 * jnp.einsum("ea,eab,eb->e", center_strain, membrane_matrix, center_strain) * model.element_areas_m2
    )
    loaded_projection = jnp.dot(nodal_displacement_xy[model.loaded_node_indices], model.load_direction_xy)

    return {
        "mesh_node_count": jnp.asarray(model.mesh_node_count, dtype=jnp.int32),
        "mesh_element_count": jnp.asarray(model.mesh_element_count, dtype=jnp.int32),
        "compliance_n_m": compliance_n_m,
        "normalized_compliance": normalized_compliance,
        "reference_compliance_n_m": jnp.asarray(model.reference_compliance_n_m, dtype=jnp.float32),
        "force_vector_n": model.force_vector_n,
        "load_direction_xy": model.load_direction_xy,
        "node_uv": model.node_uv,
        "node_xy_m": model.node_xy_m,
        "node_displacement_xy_m": nodal_displacement_xy,
        "node_displacement_magnitude_m": nodal_displacement_magnitude,
        "element_nodes": model.element_nodes,
        "element_centers_uv": model.element_centers_uv,
        "element_areas_m2": model.element_areas_m2,
        "active_elements": model.active_elements,
        "active_nodes": model.active_nodes,
        "fiber_direction_xy": fiber_direction_xy,
        "fiber_angle_rad": fiber_angle_rad,
        "fiber_thickness_m": fiber_thickness_m,
        "total_thickness_m": total_thickness_m,
        "element_membrane_matrix_n_per_m": membrane_matrix,
        "element_stiffness_matrix_n_per_m": element_stiffness,
        "element_strain": center_strain,
        "element_stress_pa": element_stress_pa,
        "element_von_mises_pa": von_mises_pa,
        "element_strain_energy_n_m": element_strain_energy_n_m,
        "maximum_displacement_m": jnp.max(nodal_displacement_magnitude),
        "mean_loaded_edge_displacement_m": jnp.mean(loaded_projection),
        "solver_residual_norm": jnp.linalg.norm(residual),
        "free_dof_count": jnp.asarray(model.free_dofs.shape[0], dtype=jnp.int32),
        "fixed_dof_count": jnp.asarray(model.fixed_dofs.shape[0], dtype=jnp.int32),
    }
