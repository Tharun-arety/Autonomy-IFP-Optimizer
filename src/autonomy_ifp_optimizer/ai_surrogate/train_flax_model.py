from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..config import GeometryConfig, LoadCase, OptimizationConfig, SurrogateConfig
from ..core.fem import prepare_membrane_fem_model
from ..core.geometry import load_surface
from ..core.physics import control_points_from_raw, evaluate_raw_design

SURROGATE_TARGET_NAMES = [
    "total_loss",
    "normalized_compliance",
    "steering_penalty",
    "thickness_penalty",
    "keepout_penalty",
]


def generate_dataset(
    surface_name: str,
    config: SurrogateConfig,
    optimization_config: OptimizationConfig | None = None,
) -> dict[str, np.ndarray]:
    optimization_config = optimization_config or OptimizationConfig()
    surface = load_surface(mesh=surface_name, surface=surface_name, geometry_config=GeometryConfig(surface=surface_name))
    load_case = LoadCase()
    fem_model = prepare_membrane_fem_model(surface, load_case, optimization_config)

    key = jax.random.PRNGKey(config.seed)
    valid_count = int(config.samples * config.valid_fraction)
    invalid_count = config.samples - valid_count

    valid_key, invalid_key = jax.random.split(key)
    valid_raw = 0.8 * jax.random.normal(valid_key, (valid_count, 5))
    invalid_raw = 2.2 * jax.random.normal(invalid_key, (invalid_count, 5))
    invalid_raw = invalid_raw.at[:, 1].add(2.2)
    raw_params = jnp.concatenate([valid_raw, invalid_raw], axis=0)

    def feature_and_target(raw: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        design = evaluate_raw_design(raw, surface, load_case, optimization_config, fem_model=fem_model)
        control_points, thickness_scale = control_points_from_raw(raw, surface, optimization_config)
        features = jnp.concatenate([control_points[1:3].reshape(-1), jnp.asarray([thickness_scale])], axis=0)
        targets = jnp.asarray(
            [
                design["loss"],
                design["normalized_compliance"],
                design["steering_penalty"],
                design["thickness_penalty"],
                design["keepout_penalty"],
            ],
            dtype=jnp.float32,
        )
        return features, targets

    features, targets = jax.vmap(feature_and_target)(raw_params)
    permutation = jax.random.permutation(key, features.shape[0])
    return {
        "features": np.asarray(features[permutation]),
        "targets": np.asarray(targets[permutation]),
    }


def train_surrogate(
    surface_name: str = "plate_with_hole",
    config: SurrogateConfig | None = None,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    try:
        from flax import linen as nn
        from flax import serialization
        import optax
    except ImportError as exc:
        raise RuntimeError("Flax and Optax are required to train the surrogate model.") from exc

    config = config or SurrogateConfig()
    dataset = generate_dataset(surface_name, config)
    features = jnp.asarray(dataset["features"], dtype=jnp.float32)
    targets = jnp.asarray(dataset["targets"], dtype=jnp.float32)

    split_index = max(1, min(features.shape[0] - 1, int(0.8 * features.shape[0])))
    train_x, val_x = features[:split_index], features[split_index:]
    train_y, val_y = targets[:split_index], targets[split_index:]

    x_mean = jnp.mean(train_x, axis=0)
    x_std = jnp.std(train_x, axis=0) + 1.0e-6
    y_mean = jnp.mean(train_y, axis=0)
    y_std = jnp.std(train_y, axis=0) + 1.0e-6

    norm_train_x = (train_x - x_mean) / x_std
    norm_val_x = (val_x - x_mean) / x_std
    norm_train_y = (train_y - y_mean) / y_std
    norm_val_y = (val_y - y_mean) / y_std

    class SurrogateMLP(nn.Module):
        hidden_width: int
        hidden_depth: int
        output_dim: int

        @nn.compact
        def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
            x = inputs
            for _ in range(self.hidden_depth):
                x = nn.Dense(self.hidden_width)(x)
                x = nn.gelu(x)
            return nn.Dense(self.output_dim)(x)

    model = SurrogateMLP(
        hidden_width=config.hidden_width,
        hidden_depth=config.hidden_depth,
        output_dim=targets.shape[1],
    )
    rng = jax.random.PRNGKey(config.seed)
    params = model.init(rng, norm_train_x[:1])
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(current_params: Any, current_opt_state: Any, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
        def loss_fn(model_params: Any) -> jnp.ndarray:
            prediction = model.apply(model_params, batch_x)
            return jnp.mean((prediction - batch_y) ** 2)

        loss_value, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_opt_state, loss_value

    @jax.jit
    def eval_step(model_params: Any, batch_x: jnp.ndarray, batch_y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        prediction = model.apply(model_params, batch_x)
        mse = jnp.mean((prediction - batch_y) ** 2)
        return mse, prediction

    history = []
    for epoch in range(config.epochs):
        permutation = np.random.default_rng(config.seed + epoch).permutation(split_index)
        shuffled_x = norm_train_x[permutation]
        shuffled_y = norm_train_y[permutation]

        batch_losses = []
        for start in range(0, split_index, config.batch_size):
            end = min(start + config.batch_size, split_index)
            params, opt_state, batch_loss = train_step(params, opt_state, shuffled_x[start:end], shuffled_y[start:end])
            batch_losses.append(float(batch_loss))

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            val_mse, _ = eval_step(params, norm_val_x, norm_val_y)
            history.append(
                {
                    "epoch": epoch,
                    "train_mse": float(np.mean(batch_losses)),
                    "val_mse": float(val_mse),
                }
            )

    val_mse, val_prediction = eval_step(params, norm_val_x, norm_val_y)
    denormalized_prediction = val_prediction * y_std + y_mean
    validation_rmse = float(jnp.sqrt(jnp.mean((denormalized_prediction - val_y) ** 2)))

    preview_batch = norm_val_x[: min(64, norm_val_x.shape[0])]
    _ = model.apply(params, preview_batch)
    start_time = time.perf_counter()
    _ = model.apply(params, preview_batch).block_until_ready()
    inference_latency_ms = 1000.0 * (time.perf_counter() - start_time)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "surrogate_dataset.npz"
    params_path = output_dir / "surrogate_params.msgpack"
    metrics_path = output_dir / "surrogate_metrics.json"
    validation_path = output_dir / "surrogate_validation.npz"

    np.savez(dataset_path, features=np.asarray(features), targets=np.asarray(targets))
    np.savez(
        validation_path,
        target_names=np.asarray(SURROGATE_TARGET_NAMES),
        y_true=np.asarray(val_y),
        y_pred=np.asarray(denormalized_prediction),
        y_true_normalized=np.asarray(norm_val_y),
        y_pred_normalized=np.asarray(val_prediction),
    )
    params_path.write_bytes(serialization.to_bytes(params))
    metrics_path.write_text(
        json.dumps(
            {
                "surface": surface_name,
                "samples": config.samples,
                "epochs": config.epochs,
                "target_names": SURROGATE_TARGET_NAMES,
                "validation_mse_normalized": float(val_mse),
                "validation_rmse": validation_rmse,
                "inference_latency_ms": inference_latency_ms,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "dataset_path": str(dataset_path),
        "params_path": str(params_path),
        "metrics_path": str(metrics_path),
        "validation_path": str(validation_path),
        "validation_rmse": validation_rmse,
        "inference_latency_ms": inference_latency_ms,
    }
