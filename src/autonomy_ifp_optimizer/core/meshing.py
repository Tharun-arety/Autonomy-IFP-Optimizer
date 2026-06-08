from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import OptimizationConfig
from .geometry import SurfaceDefinition, surface_plane_bounds, surface_uv_from_plane


@dataclass(frozen=True)
class SurfaceMesh:
    node_uv: np.ndarray
    node_xy_m: np.ndarray
    element_nodes: np.ndarray
    element_centers_uv: np.ndarray
    element_areas_m2: np.ndarray
    element_b_matrices: np.ndarray
    boundary_segments: dict[str, np.ndarray]


def _classify_boundary_curves(surface: SurfaceDefinition, surface_tag: int) -> dict[str, list[int]]:
    import gmsh

    x_min, x_max, y_min, y_max = surface_plane_bounds(surface)
    span = max(abs(x_max - x_min), abs(y_max - y_min), 1.0)
    tolerance = 1.0e-4 * span
    groups = {"left": [], "right": [], "bottom": [], "top": [], "keepout": []}
    for dim, curve_tag in gmsh.model.getBoundary([(2, surface_tag)], oriented=False, recursive=False):
        bbox = gmsh.model.getBoundingBox(dim, curve_tag)
        curve_x_min, curve_y_min, _, curve_x_max, curve_y_max, _ = bbox
        if abs(curve_x_min - x_min) <= tolerance and abs(curve_x_max - x_min) <= tolerance:
            groups["left"].append(curve_tag)
        elif abs(curve_x_min - x_max) <= tolerance and abs(curve_x_max - x_max) <= tolerance:
            groups["right"].append(curve_tag)
        elif abs(curve_y_min - y_min) <= tolerance and abs(curve_y_max - y_min) <= tolerance:
            groups["bottom"].append(curve_tag)
        elif abs(curve_y_min - y_max) <= tolerance and abs(curve_y_max - y_max) <= tolerance:
            groups["top"].append(curve_tag)
        else:
            groups["keepout"].append(curve_tag)
    return groups


def _create_occ_surface(surface: SurfaceDefinition) -> tuple[int, dict[str, list[int]]]:
    import gmsh

    x_min, x_max, y_min, y_max = surface_plane_bounds(surface)
    occ = gmsh.model.occ
    width = x_max - x_min
    height = y_max - y_min
    outer_surface = occ.addRectangle(x_min, y_min, 0.0, width, height)

    tool_entities: list[tuple[int, int]] = []
    if surface.kind == "plate_with_hole":
        for zone in surface.keep_outs:
            center_u, center_v = zone.center_uv
            center_x = surface.params["length_m"] * (center_u - 0.5)
            center_y = surface.params["width_m"] * (center_v - 0.5)
            radius_x = zone.radius_uv * surface.params["length_m"]
            radius_y = zone.radius_uv * surface.params["width_m"]
            tool_entities.append((2, occ.addDisk(center_x, center_y, 0.0, radius_x, radius_y)))

    if tool_entities:
        cut_surfaces, _ = occ.cut([(2, outer_surface)], tool_entities, removeObject=True, removeTool=True)
        if len(cut_surfaces) != 1:
            raise ValueError(f"Expected one trimmed surface for {surface.name}, got {len(cut_surfaces)}")
        surface_tag = cut_surfaces[0][1]
    else:
        surface_tag = outer_surface

    occ.synchronize()
    boundary_groups = _classify_boundary_curves(surface, surface_tag)
    gmsh.model.addPhysicalGroup(2, [surface_tag], tag=1)
    gmsh.model.setPhysicalName(2, 1, "domain")
    for index, (name, curve_tags) in enumerate(boundary_groups.items(), start=100):
        if curve_tags:
            gmsh.model.addPhysicalGroup(1, curve_tags, tag=index)
            gmsh.model.setPhysicalName(1, index, name)
    return surface_tag, boundary_groups


def _configure_mesh_controls(surface: SurfaceDefinition, config: OptimizationConfig, boundary_groups: dict[str, list[int]]) -> None:
    import gmsh

    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.Algorithm", float(config.mesh_algorithm))
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMin", min(config.mesh_refined_size_m, config.mesh_target_size_m))
    gmsh.option.setNumber("Mesh.MeshSizeMax", config.mesh_target_size_m)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    point_entities = gmsh.model.getEntities(0)
    if point_entities:
        gmsh.model.mesh.setSize(point_entities, config.mesh_target_size_m)

    if surface.keep_outs and boundary_groups["keepout"]:
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", boundary_groups["keepout"])
        gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 200)

        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", min(config.mesh_refined_size_m, config.mesh_target_size_m))
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", config.mesh_target_size_m)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "DistMax",
            max(config.mesh_refinement_distance_m, config.mesh_target_size_m),
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)


def _map_tags_to_indices(sorted_node_tags: np.ndarray, entity_node_tags: np.ndarray) -> np.ndarray:
    mapped = np.searchsorted(sorted_node_tags, entity_node_tags)
    if np.any(mapped >= sorted_node_tags.shape[0]) or not np.array_equal(sorted_node_tags[mapped], entity_node_tags):
        raise ValueError("Failed to map Gmsh node tags onto the extracted surface node array.")
    return mapped.astype(np.int32)


def _extract_triangle_elements(surface_tag: int, sorted_node_tags: np.ndarray) -> np.ndarray:
    import gmsh

    element_types, _, element_node_tags = gmsh.model.mesh.getElements(2, surface_tag)
    triangle_sets: list[np.ndarray] = []
    for element_type, node_tags in zip(element_types, element_node_tags):
        _, _, _, node_count, _, _ = gmsh.model.mesh.getElementProperties(element_type)
        if node_count != 3:
            continue
        triangle_sets.append(_map_tags_to_indices(sorted_node_tags, np.asarray(node_tags, dtype=np.int64)).reshape(-1, 3))
    if not triangle_sets:
        raise ValueError("Expected a first-order triangular surface mesh from Gmsh.")
    return np.vstack(triangle_sets).astype(np.int32)


def _extract_boundary_segments(curve_tags: list[int], sorted_node_tags: np.ndarray) -> np.ndarray:
    import gmsh

    line_segments: list[np.ndarray] = []
    for curve_tag in curve_tags:
        element_types, _, element_node_tags = gmsh.model.mesh.getElements(1, curve_tag)
        for element_type, node_tags in zip(element_types, element_node_tags):
            _, _, _, node_count, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            if node_count != 2:
                continue
            mapped = _map_tags_to_indices(sorted_node_tags, np.asarray(node_tags, dtype=np.int64)).reshape(-1, 2)
            line_segments.append(mapped)
    if not line_segments:
        return np.zeros((0, 2), dtype=np.int32)

    segments = np.vstack(line_segments).astype(np.int32)
    canonical = np.sort(segments, axis=1)
    _, unique_indices = np.unique(canonical, axis=0, return_index=True)
    return segments[np.sort(unique_indices)]


def _triangle_b_matrices(node_xy_m: np.ndarray, element_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangle_xy = node_xy_m[element_nodes]
    x1 = triangle_xy[:, 0, 0]
    y1 = triangle_xy[:, 0, 1]
    x2 = triangle_xy[:, 1, 0]
    y2 = triangle_xy[:, 1, 1]
    x3 = triangle_xy[:, 2, 0]
    y3 = triangle_xy[:, 2, 1]

    signed_double_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    negative_orientation = signed_double_area < 0.0
    if np.any(negative_orientation):
        oriented_nodes = element_nodes.copy()
        oriented_nodes[negative_orientation, 1] = element_nodes[negative_orientation, 2]
        oriented_nodes[negative_orientation, 2] = element_nodes[negative_orientation, 1]
        element_nodes = oriented_nodes
        triangle_xy = node_xy_m[element_nodes]
        x1 = triangle_xy[:, 0, 0]
        y1 = triangle_xy[:, 0, 1]
        x2 = triangle_xy[:, 1, 0]
        y2 = triangle_xy[:, 1, 1]
        x3 = triangle_xy[:, 2, 0]
        y3 = triangle_xy[:, 2, 1]
        signed_double_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

    areas = 0.5 * signed_double_area
    if np.any(areas <= 1.0e-12):
        raise ValueError("Encountered a degenerate membrane triangle with zero area.")

    beta = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)
    gamma = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)
    scale = 1.0 / (2.0 * areas)

    b_matrices = np.zeros((element_nodes.shape[0], 3, 6), dtype=np.float32)
    b_matrices[:, 0, 0::2] = beta * scale[:, None]
    b_matrices[:, 1, 1::2] = gamma * scale[:, None]
    b_matrices[:, 2, 0::2] = gamma * scale[:, None]
    b_matrices[:, 2, 1::2] = beta * scale[:, None]
    return element_nodes.astype(np.int32), areas.astype(np.float32), b_matrices


def build_surface_mesh(surface: SurfaceDefinition, config: OptimizationConfig) -> SurfaceMesh:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(surface.name or surface.kind)
        surface_tag, boundary_groups = _create_occ_surface(surface)
        _configure_mesh_controls(surface, config, boundary_groups)
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes(2, surface_tag, includeBoundary=True)
        if len(node_tags) == 0:
            raise ValueError(f"Gmsh did not produce any nodes for {surface.name}")

        node_xy = np.asarray(node_coords, dtype=np.float32).reshape(-1, 3)[:, :2]
        sorted_indices = np.argsort(np.asarray(node_tags, dtype=np.int64))
        sorted_node_tags = np.asarray(node_tags, dtype=np.int64)[sorted_indices]
        node_xy = node_xy[sorted_indices]
        node_uv = np.asarray(surface_uv_from_plane(surface, node_xy), dtype=np.float32)

        element_nodes = _extract_triangle_elements(surface_tag, sorted_node_tags)
        element_nodes, element_areas_m2, element_b_matrices = _triangle_b_matrices(node_xy, element_nodes)
        element_centers_uv = np.mean(node_uv[element_nodes], axis=1, dtype=np.float32)

        boundary_segments = {
            name: _extract_boundary_segments(curve_tags, sorted_node_tags)
            for name, curve_tags in boundary_groups.items()
        }

        return SurfaceMesh(
            node_uv=node_uv.astype(np.float32),
            node_xy_m=node_xy.astype(np.float32),
            element_nodes=element_nodes.astype(np.int32),
            element_centers_uv=element_centers_uv.astype(np.float32),
            element_areas_m2=element_areas_m2.astype(np.float32),
            element_b_matrices=element_b_matrices.astype(np.float32),
            boundary_segments=boundary_segments,
        )
    finally:
        gmsh.finalize()
