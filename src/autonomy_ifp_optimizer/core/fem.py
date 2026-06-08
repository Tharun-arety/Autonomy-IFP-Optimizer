from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..config import LoadCase, OptimizationConfig
from .geometry import SurfaceDefinition, keepout_signed_distance, load_direction_in_plane, surface_plane_coordinates


@dataclass(frozen=True)
class MembraneFEMModel:
    mesh_shape: tuple[int, int]
    node_uv: jnp.ndarray
    node_xy_m: jnp.ndarray
    element_nodes: jnp.ndarray
    element_dof_indices: jnp.ndarray
    element_centers_uv: jnp.ndarray
    active_elements: jnp.ndarray
    active_nodes: jnp.ndarray
    free_dofs: jnp.ndarray
    fixed_dofs: jnp.ndarray
    force_vector_n: jnp.ndarray
    load_direction_xy: jnp.ndarray
    loaded_node_indices: jnp.ndarray
    gauss_b_matrices: jnp.ndarray
    gauss_detj_weights: jnp.ndarray
    center_b_matrix: jnp.ndarray
    element_area_m2: float
    matrix_constitutive_pa: jnp.ndarray
    orthotropic_constitutive_pa: jnp.ndarray
    reference_compliance_n_m: float


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


def _quad_b_matrix(node_xy_m: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    dshape_dxi = 0.25 * np.asarray(
        [
            -(1.0 - eta),
            1.0 - eta,
            1.0 + eta,
            -(1.0 + eta),
        ],
        dtype=np.float32,
    )
    dshape_deta = 0.25 * np.asarray(
        [
            -(1.0 - xi),
            -(1.0 + xi),
            1.0 + xi,
            1.0 - xi,
        ],
        dtype=np.float32,
    )
    jacobian = np.asarray(
        [
            [np.dot(dshape_dxi, node_xy_m[:, 0]), np.dot(dshape_deta, node_xy_m[:, 0])],
            [np.dot(dshape_dxi, node_xy_m[:, 1]), np.dot(dshape_deta, node_xy_m[:, 1])],
        ],
        dtype=np.float32,
    )
    inv_jacobian = np.linalg.inv(jacobian)
    gradients = inv_jacobian @ np.stack([dshape_dxi, dshape_deta], axis=0)
    dshape_dx = gradients[0]
    dshape_dy = gradients[1]

    b_matrix = np.zeros((3, 8), dtype=np.float32)
    b_matrix[0, 0::2] = dshape_dx
    b_matrix[1, 1::2] = dshape_dy
    b_matrix[2, 0::2] = dshape_dy
    b_matrix[2, 1::2] = dshape_dx
    return b_matrix, float(abs(np.linalg.det(jacobian)))


def _reference_element_matrices(span_x_m: float, span_y_m: float, mesh_u: int, mesh_v: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    width = span_x_m / mesh_u
    height = span_y_m / mesh_v
    node_xy = np.asarray(
        [
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height],
        ],
        dtype=np.float32,
    )
    gauss = 1.0 / np.sqrt(3.0)
    points = [(-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss)]
    b_matrices = []
    detj_weights = []
    for xi, eta in points:
        b_matrix, det_j = _quad_b_matrix(node_xy, xi, eta)
        b_matrices.append(b_matrix)
        detj_weights.append(det_j)
    center_b_matrix, _ = _quad_b_matrix(node_xy, 0.0, 0.0)
    return (
        jnp.asarray(np.stack(b_matrices, axis=0), dtype=jnp.float32),
        jnp.asarray(np.asarray(detj_weights, dtype=np.float32), dtype=jnp.float32),
        jnp.asarray(center_b_matrix, dtype=jnp.float32),
        float(width * height),
    )


def _surface_plane_spans(surface: SurfaceDefinition) -> tuple[float, float]:
    if surface.kind == "cylinder":
        radius = surface.params["radius_m"]
        return 2.0 * np.pi * radius, surface.params["height_m"]
    return surface.params["length_m"], surface.params["width_m"]


def _select_edges(load_xy: np.ndarray) -> tuple[str, str]:
    if abs(load_xy[0]) >= abs(load_xy[1]):
        return ("left", "right") if load_xy[0] >= 0.0 else ("right", "left")
    return ("bottom", "top") if load_xy[1] >= 0.0 else ("top", "bottom")


def _build_force_vector(
    node_xy_m: np.ndarray,
    boundary_nodes: np.ndarray,
    load_direction_xy: np.ndarray,
    load_magnitude_n: float,
    ndof: int,
) -> np.ndarray:
    force = np.zeros(ndof, dtype=np.float32)
    if boundary_nodes.size < 2:
        return force

    edge_points = node_xy_m[boundary_nodes]
    segment_lengths = np.linalg.norm(edge_points[1:] - edge_points[:-1], axis=1)
    total_edge_length = max(float(np.sum(segment_lengths)), 1.0e-8)
    traction = load_magnitude_n * load_direction_xy / total_edge_length

    for start, end, length in zip(boundary_nodes[:-1], boundary_nodes[1:], segment_lengths):
        nodal_force = 0.5 * traction * float(length)
        for node in (int(start), int(end)):
            force[2 * node : 2 * node + 2] += nodal_force
    return force


def _assemble_global_stiffness(
    model: MembraneFEMModel,
    element_membrane_matrix_n_per_m: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    bt = jnp.swapaxes(model.gauss_b_matrices, 1, 2)
    element_stiffness = jnp.einsum(
        "gia,eab,gbj,g->eij",
        bt,
        element_membrane_matrix_n_per_m,
        model.gauss_b_matrices,
        model.gauss_detj_weights,
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
    mesh_u = int(config.fem_elements_u)
    mesh_v = int(config.fem_elements_v)
    u_nodes = np.linspace(0.0, 1.0, mesh_u + 1, dtype=np.float32)
    v_nodes = np.linspace(0.0, 1.0, mesh_v + 1, dtype=np.float32)
    uu, vv = np.meshgrid(u_nodes, v_nodes, indexing="xy")
    node_uv = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1)
    node_xy = np.asarray(surface_plane_coordinates(surface, jnp.asarray(node_uv, dtype=jnp.float32)))

    elements = []
    for j in range(mesh_v):
        for i in range(mesh_u):
            lower_left = j * (mesh_u + 1) + i
            elements.append(
                [
                    lower_left,
                    lower_left + 1,
                    lower_left + mesh_u + 2,
                    lower_left + mesh_u + 1,
                ]
            )
    element_nodes = np.asarray(elements, dtype=np.int32)
    element_centers_uv = np.mean(node_uv[element_nodes], axis=1)
    active_elements = np.asarray(
        keepout_signed_distance(surface, jnp.asarray(element_centers_uv, dtype=jnp.float32)) >= 0.0,
        dtype=bool,
    )

    active_nodes = np.zeros(node_uv.shape[0], dtype=bool)
    active_nodes[np.unique(element_nodes[active_elements].reshape(-1))] = True

    grid = np.arange(node_uv.shape[0], dtype=np.int32).reshape(mesh_v + 1, mesh_u + 1)
    boundary_sets = {
        "left": grid[:, 0],
        "right": grid[:, -1],
        "bottom": grid[0, :],
        "top": grid[-1, :],
    }

    load_xy = np.asarray(load_direction_in_plane(surface, load_case.direction_xyz))
    fixed_side, loaded_side = _select_edges(load_xy)
    fixed_nodes = boundary_sets[fixed_side][active_nodes[boundary_sets[fixed_side]]]
    loaded_nodes = boundary_sets[loaded_side][active_nodes[boundary_sets[loaded_side]]]

    ndof = node_uv.shape[0] * 2
    force_vector = _build_force_vector(node_xy, loaded_nodes, load_xy, load_case.magnitude_n, ndof)

    fixed_dofs = np.sort(np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1]).astype(np.int32))
    active_dofs = np.sort(np.concatenate([2 * np.flatnonzero(active_nodes), 2 * np.flatnonzero(active_nodes) + 1]).astype(np.int32))
    free_dofs = np.asarray(sorted(set(active_dofs.tolist()) - set(fixed_dofs.tolist())), dtype=np.int32)

    dof_indices = np.zeros((element_nodes.shape[0], 8), dtype=np.int32)
    for idx, nodes in enumerate(element_nodes):
        dof_indices[idx] = np.asarray(
            [
                2 * nodes[0],
                2 * nodes[0] + 1,
                2 * nodes[1],
                2 * nodes[1] + 1,
                2 * nodes[2],
                2 * nodes[2] + 1,
                2 * nodes[3],
                2 * nodes[3] + 1,
            ],
            dtype=np.int32,
        )

    span_x_m, span_y_m = _surface_plane_spans(surface)
    b_matrices, detj_weights, center_b, element_area_m2 = _reference_element_matrices(span_x_m, span_y_m, mesh_u, mesh_v)
    matrix_q = _isotropic_plane_stress_q(config.matrix_modulus_pa, config.matrix_poisson)
    ortho_q = _orthotropic_plane_stress_q(
        config.fiber_modulus_longitudinal_pa,
        config.fiber_modulus_transverse_pa,
        config.fiber_shear_modulus_pa,
        config.fiber_poisson,
    )

    provisional_model = MembraneFEMModel(
        mesh_shape=(mesh_u, mesh_v),
        node_uv=jnp.asarray(node_uv, dtype=jnp.float32),
        node_xy_m=jnp.asarray(node_xy, dtype=jnp.float32),
        element_nodes=jnp.asarray(element_nodes, dtype=jnp.int32),
        element_dof_indices=jnp.asarray(dof_indices, dtype=jnp.int32),
        element_centers_uv=jnp.asarray(element_centers_uv, dtype=jnp.float32),
        active_elements=jnp.asarray(active_elements.astype(np.float32), dtype=jnp.float32),
        active_nodes=jnp.asarray(active_nodes.astype(np.float32), dtype=jnp.float32),
        free_dofs=jnp.asarray(free_dofs, dtype=jnp.int32),
        fixed_dofs=jnp.asarray(fixed_dofs, dtype=jnp.int32),
        force_vector_n=jnp.asarray(force_vector, dtype=jnp.float32),
        load_direction_xy=jnp.asarray(load_xy, dtype=jnp.float32),
        loaded_node_indices=jnp.asarray(loaded_nodes, dtype=jnp.int32),
        gauss_b_matrices=b_matrices,
        gauss_detj_weights=detj_weights,
        center_b_matrix=center_b,
        element_area_m2=element_area_m2,
        matrix_constitutive_pa=matrix_q,
        orthotropic_constitutive_pa=ortho_q,
        reference_compliance_n_m=1.0,
    )

    base_membrane = provisional_model.active_elements[:, None, None] * (
        config.base_laminate_thickness_m * provisional_model.matrix_constitutive_pa[None, :, :]
    )
    reference_displacement, _, _ = _solve_displacements(provisional_model, base_membrane, config.fem_regularization)
    reference_compliance = float(
        jnp.maximum(jnp.dot(provisional_model.force_vector_n, reference_displacement), 1.0e-8)
    )

    return MembraneFEMModel(
        mesh_shape=provisional_model.mesh_shape,
        node_uv=provisional_model.node_uv,
        node_xy_m=provisional_model.node_xy_m,
        element_nodes=provisional_model.element_nodes,
        element_dof_indices=provisional_model.element_dof_indices,
        element_centers_uv=provisional_model.element_centers_uv,
        active_elements=provisional_model.active_elements,
        active_nodes=provisional_model.active_nodes,
        free_dofs=provisional_model.free_dofs,
        fixed_dofs=provisional_model.fixed_dofs,
        force_vector_n=provisional_model.force_vector_n,
        load_direction_xy=provisional_model.load_direction_xy,
        loaded_node_indices=provisional_model.loaded_node_indices,
        gauss_b_matrices=provisional_model.gauss_b_matrices,
        gauss_detj_weights=provisional_model.gauss_detj_weights,
        center_b_matrix=provisional_model.center_b_matrix,
        element_area_m2=provisional_model.element_area_m2,
        matrix_constitutive_pa=provisional_model.matrix_constitutive_pa,
        orthotropic_constitutive_pa=provisional_model.orthotropic_constitutive_pa,
        reference_compliance_n_m=reference_compliance,
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
    fiber_thickness_m = config.fiber_layer_thickness_m * thickness_scale * fiber_presence * model.active_elements
    membrane_matrix = model.active_elements[:, None, None] * (
        config.base_laminate_thickness_m * model.matrix_constitutive_pa[None, :, :]
        + fiber_thickness_m[:, None, None] * _rotate_plane_stress_q(model.orthotropic_constitutive_pa, fiber_direction_xy)
    )

    displacement_vector, element_stiffness, residual = _solve_displacements(model, membrane_matrix, config.fem_regularization)
    nodal_displacement_xy = displacement_vector.reshape(-1, 2)
    nodal_displacement_magnitude = jnp.linalg.norm(nodal_displacement_xy, axis=-1) * model.active_nodes
    compliance_n_m = jnp.dot(model.force_vector_n, displacement_vector)
    normalized_compliance = compliance_n_m / jnp.maximum(model.reference_compliance_n_m, 1.0e-8)

    element_displacements = displacement_vector[model.element_dof_indices]
    center_strain = jnp.einsum("ab,eb->ea", model.center_b_matrix, element_displacements)
    total_thickness_m = model.active_elements * (config.base_laminate_thickness_m + fiber_thickness_m)
    effective_q = jnp.where(
        total_thickness_m[:, None, None] > 1.0e-12,
        membrane_matrix / jnp.maximum(total_thickness_m[:, None, None], 1.0e-12),
        0.0,
    )
    element_stress_pa = jnp.einsum("eab,eb->ea", effective_q, center_strain)
    von_mises_pa = jnp.sqrt(
        jnp.maximum(
            element_stress_pa[:, 0] ** 2
            - element_stress_pa[:, 0] * element_stress_pa[:, 1]
            + element_stress_pa[:, 1] ** 2
            + 3.0 * element_stress_pa[:, 2] ** 2,
            0.0,
        )
    ) * model.active_elements
    element_strain_energy_n_m = (
        0.5 * jnp.einsum("ea,eab,eb->e", center_strain, membrane_matrix, center_strain) * model.element_area_m2
    )
    loaded_projection = jnp.dot(nodal_displacement_xy[model.loaded_node_indices], model.load_direction_xy)

    return {
        "mesh_shape": jnp.asarray(model.mesh_shape, dtype=jnp.int32),
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
