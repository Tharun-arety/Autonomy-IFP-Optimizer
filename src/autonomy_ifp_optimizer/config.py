from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadCase:
    magnitude_n: float = 500.0
    direction_xyz: tuple[float, float, float] = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class GeometryConfig:
    surface: str = "plate_with_hole"
    length_m: float = 0.60
    width_m: float = 0.35
    crown_m: float = 0.018
    hole_center_uv: tuple[float, float] = (0.50, 0.50)
    hole_radius_uv: float = 0.14
    cylinder_radius_m: float = 0.09
    cylinder_height_m: float = 0.70


@dataclass(frozen=True)
class OptimizationConfig:
    steps: int = 350
    learning_rate: float = 0.025
    num_path_samples: int = 160
    coverage_grid: int = 48
    min_steering_radius_m: float = 0.05
    max_thickness: float = 1.20
    tow_half_width_uv: float = 0.028
    thickness_scale_bounds: tuple[float, float] = (0.35, 2.00)
    structural_weight: float = 1.00
    length_weight: float = 0.03
    steering_weight: float = 22.0
    thickness_weight: float = 7.0
    keepout_weight: float = 50.0
    boundary_weight: float = 15.0
    smoothness_weight: float = 0.20
    grad_clip_norm: float = 1.0
    history_stride: int = 10


@dataclass(frozen=True)
class ExportConfig:
    sample_count: int = 200
    placement_speed_mps: float = 0.5
    roving_linear_density_kg_per_m: float = 0.0008
    output_dir: Path = Path("outputs")


@dataclass(frozen=True)
class SurrogateConfig:
    samples: int = 512
    valid_fraction: float = 0.6
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    hidden_width: int = 64
    hidden_depth: int = 3
    seed: int = 0
