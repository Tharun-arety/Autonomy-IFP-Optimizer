# Autonomy-IFP-Optimizer

Autonomy-IFP-Optimizer is a differentiable path-planning engine for Infinite Fiber Placement (IFP) robotics.

Traditional composite CAM workflows rely on forward-engineering loops: define a path, evaluate it, adjust parameters, and repeat. This project uses JAX-based automatic differentiation to optimize fiber paths directly against structural proxies and manufacturing constraints such as minimum steering radius, thickness buildup, and keep-out zones. The repository also includes data-generation and surrogate-model training utilities for learned approximations of the same physics pipeline.

## Highlights

- Differentiable path optimization on analytic surface families
- Manufacturing-aware objectives with steering, thickness, boundary, and keep-out penalties
- Robot-oriented toolpath export with kinematic records and process metrics
- Flax surrogate training on physics-generated samples

## Repository Components

This repository is organized around four main components:

- `core/geometry.py`
  Defines the analytic surface models used by the demos, including a plate-with-hole surface and a cylinder. Imported meshes are mapped onto these surface families so the optimization remains stable and differentiable.
- `core/physics.py` and `core/constraints.py`
  Optimize cubic Bezier IFP paths with JAX auto-differentiation. The objective combines a structural stiffness proxy with manufacturing penalties for curvature, thickness buildup, boundary escape, and cutout intrusion.
- `export/toolpath.py`
  Converts optimized paths into robot-ready kinematic records: XYZ points, surface normals, tangents, binormals, arc length, and local steering radius. It also computes cycle-time and material estimates.
- `ai_surrogate/train_flax_model.py`
  Generates physics-labeled path samples and trains a Flax MLP surrogate to approximate loss and manufacturability terms at inference speed.

## Optimization Model

The optimizer solves for:

- Two internal Bezier control points in surface parameter space
- A continuous deposition and thickness scale

The loss combines:

- `compliance_proxy`
  A stiffness-oriented proxy that rewards alignment of the path tangent with the local preferred load direction and stress concentration field.
- `steering_penalty`
  An exponential penalty when the local steering radius drops below the minimum IFP limit. The CLI default is `50 mm`.
- `thickness_penalty`
  A differentiable deposition-density map across the surface that penalizes unstable buildup and non-uniform laydown.
- `keepout_penalty`
  Smoothly discourages paths from entering holes and integrated cutouts.
- `boundary_penalty` and `smoothness_penalty`
  Keep the route on the valid surface and prevent erratic control polygons.

Manufacturability is part of the optimization state rather than a post-processing check.

## CLI Workflow

Run the full differentiable optimization on the demo plate:

```bash
python main.py optimize --mesh examples/drone_plate.obj --load 500 --min-radius 50
```

Export the optimized path to robot-facing kinematics:

```bash
python main.py export --input outputs/optimized_path.json --format json
```

Train the Flax surrogate model on physics-generated IFP samples:

```bash
python main.py train-surrogate --samples 1000 --epochs 250
```

Switch to a tubular demonstration surface:

```bash
python main.py optimize --surface cylinder --load 650 --direction 0,0,1
```

## Generated Artifacts

After `optimize`, the repository writes:

- `outputs/optimized_path.json`
  Full optimization result including control points, sampled path, normals, objective terms, and metrics
- `outputs/metrics.json`
  Manufacturability and process metrics including steering radius, estimated cycle time, and material estimate
- `outputs/ifp_kinematics.json` or `outputs/ifp_kinematics.csv`
  Robot-consumable kinematic array with XYZ positions plus tool-orientation vectors
- `outputs/ifp_preview.png`
  Preview showing the optimized path, thickness field, and steering-radius compliance

After `train-surrogate`, the repository writes:

- `outputs/surrogate_dataset.npz`
- `outputs/surrogate_params.msgpack`
- `outputs/surrogate_metrics.json`

## Example Notebooks

- `examples/drone_frame_cutout_avoidance.ipynb`
  Plate-with-hole workflow from geometry inspection through optimization, toolpath export, and process metrics.
- `examples/robotic_limb_optimization.ipynb`
  Cylindrical workflow from limb geometry through optimized routing, robotic kinematics export, and manufacturing metrics.
- `examples/drone_plate.obj`
  Lightweight mesh asset used by the cutout-avoidance demo.

## Scope and Limitations

This repository is a technical prototype, not a production planning system. The current implementation is intentionally scoped around analytic surfaces that keep the optimization differentiable and stable:

- 2.5D plate-with-hole parts for cutout-routing demonstrations
- Cylindrical parts for tubular IFP routing
- Single-path optimization with continuous deposition scaling

This scope is deliberate: it validates the core architecture before scaling into richer imported geometries, multi-course planning, collision-aware robot kinematics, and tighter manufacturing-software integration.

## Repository Layout

```text
Autonomy-IFP-Optimizer/
  examples/
    drone_frame_cutout_avoidance.ipynb
    robotic_limb_optimization.ipynb
    drone_plate.obj
  outputs/
  src/autonomy_ifp_optimizer/
    ai_surrogate/
      train_flax_model.py
    core/
      constraints.py
      geometry.py
      physics.py
    export/
      toolpath.py
    cli.py
    config.py
    visualize.py
  main.py
```

## Roadmap

- Replace the current stiffness proxy with a differentiable shell or reduced-order laminate response.
- Extend from one Bezier course to multi-course cooperative routing.
- Add robot singularity and head-clearance constraints alongside the current surface-normal export.
- Feed surrogate uncertainty back into the optimization loop for active learning.
