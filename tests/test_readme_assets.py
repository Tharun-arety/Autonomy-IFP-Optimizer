from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autonomy_ifp_optimizer.core.geometry import KeepOutZone, SurfaceDefinition


def _load_assets_module():
    script_path = ROOT / "tools" / "generate_readme_assets.py"
    spec = importlib.util.spec_from_file_location("generate_readme_assets", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_keepout_transition_uses_na_for_surfaces_without_keepouts() -> None:
    module = _load_assets_module()
    surface = SurfaceDefinition(
        name="robotic_limb",
        kind="cylinder",
        params={"radius_m": 0.09, "height_m": 0.7},
        start_uv=(0.08, 0.16),
        end_uv=(0.92, 0.84),
    )

    text = module._format_keepout_transition(
        {"surface": surface},
        {"minimum_keepout_clearance_uv": 1000.0},
        {"minimum_keepout_clearance_uv": 1000.0},
    )

    assert text == "Keep-out clearance: n/a (no keep-out zones)"


def test_keepout_transition_formats_real_clearance_values() -> None:
    module = _load_assets_module()
    surface = SurfaceDefinition(
        name="drone_plate",
        kind="plate_with_hole",
        params={"length_m": 1.0, "width_m": 1.0, "crown_m": 0.0},
        start_uv=(0.08, 0.24),
        end_uv=(0.92, 0.76),
        keep_outs=(KeepOutZone(center_uv=(0.5, 0.5), radius_uv=0.14),),
    )

    text = module._format_keepout_transition(
        {"surface": surface},
        {"minimum_keepout_clearance_uv": -0.137},
        {"minimum_keepout_clearance_uv": 0.0704},
    )

    assert text == "Keep-out clearance: -0.137 -> +0.070 uv"
