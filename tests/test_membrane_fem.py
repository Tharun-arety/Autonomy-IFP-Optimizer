from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autonomy_ifp_optimizer import GeometryConfig, LoadCase, OptimizationConfig, load_surface
from autonomy_ifp_optimizer.core.fem import prepare_membrane_fem_model
from autonomy_ifp_optimizer.core.geometry import keepout_signed_distance
from autonomy_ifp_optimizer.core.physics import evaluate_raw_design, initial_raw_params, optimize_ifp_path


def test_evaluate_raw_design_exposes_fem_response() -> None:
    surface = load_surface(surface="plate_with_hole", geometry_config=GeometryConfig(surface="plate_with_hole"))
    load_case = LoadCase()
    config = OptimizationConfig(steps=5)
    fem_model = prepare_membrane_fem_model(surface, load_case, config)
    raw = initial_raw_params(surface, config)

    design = evaluate_raw_design(raw, surface, load_case, config, fem_model=fem_model)
    fem = design["fem"]

    assert float(design["compliance_n_m"]) > 0.0
    assert float(design["normalized_compliance"]) > 0.0
    assert float(fem["maximum_displacement_m"]) > 0.0
    assert np.asarray(fem["element_membrane_matrix_n_per_m"]).shape[-2:] == (3, 3)
    assert np.asarray(fem["node_displacement_xy_m"]).shape[-1] == 2
    assert np.asarray(fem["element_nodes"]).shape[-1] == 3


def test_plate_mesh_is_gmsh_trimmed_around_keepout() -> None:
    surface = load_surface(surface="plate_with_hole", geometry_config=GeometryConfig(surface="plate_with_hole"))
    fem_model = prepare_membrane_fem_model(surface, LoadCase(), OptimizationConfig(steps=1))

    clearance = np.asarray(keepout_signed_distance(surface, fem_model.element_centers_uv), dtype=float)
    assert float(np.min(clearance)) > 0.0
    assert int(fem_model.mesh_element_count) > 100
    assert int(fem_model.mesh_node_count) > 60


def test_plate_optimizer_reduces_fem_compliance_and_clears_keepout() -> None:
    surface = load_surface(surface="plate_with_hole", geometry_config=GeometryConfig(surface="plate_with_hole"))
    load_case = LoadCase(magnitude_n=500.0, direction_xyz=(1.0, 0.0, 0.0))
    config = OptimizationConfig(steps=60, history_stride=10)
    fem_model = prepare_membrane_fem_model(surface, load_case, config)
    baseline = evaluate_raw_design(initial_raw_params(surface, config), surface, load_case, config, fem_model=fem_model)

    result = optimize_ifp_path(surface, load_case=load_case, config=config)

    assert result["metrics"]["normalized_compliance"] < float(baseline["normalized_compliance"])
    assert result["metrics"]["minimum_keepout_clearance_uv"] > 0.0
    assert result["metrics"]["solver_residual_norm"] < 1.0e-2


def test_cylinder_optimizer_reduces_fem_compliance_without_steering_failure() -> None:
    surface = load_surface(surface="cylinder", geometry_config=GeometryConfig(surface="cylinder"))
    load_case = LoadCase(magnitude_n=650.0, direction_xyz=(0.0, 0.0, 1.0))
    config = OptimizationConfig(steps=60, history_stride=10)
    fem_model = prepare_membrane_fem_model(surface, load_case, config)
    baseline = evaluate_raw_design(initial_raw_params(surface, config), surface, load_case, config, fem_model=fem_model)

    result = optimize_ifp_path(surface, load_case=load_case, config=config)

    assert result["metrics"]["normalized_compliance"] < float(baseline["normalized_compliance"])
    assert result["metrics"]["min_steering_radius_m"] >= config.min_steering_radius_m
    assert result["metrics"]["solver_residual_norm"] < 1.0e-2
