from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autonomy_ifp_optimizer import GeometryConfig, LoadCase, OptimizationConfig, SurrogateConfig, load_surface
from autonomy_ifp_optimizer.ai_surrogate.train_flax_model import train_surrogate
from autonomy_ifp_optimizer.core.physics import optimize_ifp_path
from autonomy_ifp_optimizer.export.toolpath import compute_metrics, write_interactive_toolpath_html


def _small_plate_result() -> dict[str, object]:
    surface = load_surface(surface="plate_with_hole", geometry_config=GeometryConfig(surface="plate_with_hole"))
    load_case = LoadCase(magnitude_n=500.0, direction_xyz=(1.0, 0.0, 0.0))
    config = OptimizationConfig(steps=12, history_stride=4)
    return optimize_ifp_path(surface, load_case=load_case, config=config)


def test_optimizer_result_contains_baseline_and_frames() -> None:
    result = _small_plate_result()

    assert "baseline" in result
    assert result["baseline"]["metrics"]["normalized_compliance"] > 0.0
    assert len(result["frames"]) >= len(result["history"])
    assert result["frames"][0]["step"] == 0.0
    assert len(result["frames"][0]["path_xyz"]) == 160
    assert len(result["frames"][0]["normals"]) == 160


def test_interactive_toolpath_export_writes_html(tmp_path: Path) -> None:
    result = _small_plate_result()
    result["metrics"] = compute_metrics(result)

    output_path = write_interactive_toolpath_html(result, tmp_path)

    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "plotly" in text.lower()
    assert "Interactive IFP Toolpath" in text
    assert 'src="https://cdn.plot.ly' not in text


def test_surrogate_training_writes_validation_artifact(tmp_path: Path) -> None:
    artifacts = train_surrogate(
        surface_name="plate_with_hole",
        config=SurrogateConfig(samples=16, epochs=2, batch_size=8),
        output_dir=tmp_path,
    )

    validation_path = Path(artifacts["validation_path"])
    assert validation_path.exists()
    payload = np.load(validation_path)
    assert payload["y_true"].shape == payload["y_pred"].shape
    assert payload["target_names"].shape[0] == 5
