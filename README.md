# Autonomy-IFP-Optimizer

Autonomy-IFP-Optimizer is an open-source, mathematically differentiable path-planning engine built specifically for Infinite Fiber Placement (IFP) robotics.

Traditional composite CAM software relies on slow, forward-engineering loops (guess a path, run FEA, adjust, repeat). This repository leverages JAX auto-differentiation to achieve true inverse design at GPU speeds. It simultaneously optimizes structural load paths while mathematically enforcing strict dry-fiber manufacturing constraints—such as minimum steering radii (to prevent fiber buckling) and thickness buildup limits.

Built natively for modern deep learning frameworks (Flax/Equinox), this pipeline is designed to instantly generate training data for AI-surrogate models, serving as a blueprint for the backend of a fully autonomous factory operating system.

## Why This Matters For Holy Technologies

- It turns IFP path planning into an inverse-design problem instead of a downstream translation step.
- It keeps physical IFP limits inside the loss function: steering radius, thickness build-up, and hole avoidance are optimized together rather than checked after the fact.
- It produces robot-consumable kinematic arrays directly from the optimized path.
- It generates labeled design data natively from the same JAX physics pipeline, so Holy can train fast surrogates for scheduling, screening, or autonomous decision-making.

## What The Repository Demonstrates

This repository implements four modules that mirror the PRD:

- `core/geometry.py`
  Supports a 2.5D plate-with-hole demo surface and a tubular cylinder surface. Imported meshes are mapped onto these differentiable surface families so the optimization remains analytic and stable.
- `core/physics.py` and `core/constraints.py`
  Optimize cubic Bezier IFP paths with JAX auto-differentiation. The objective combines a structural stiffness proxy with manufacturing penalties for curvature, thickness build-up, boundary escape, and cutout intrusion.
- `export/toolpath.py`
  Converts optimized paths into robot-ready kinematic records: XYZ points, surface normals, tangents, binormals, arc length, and local steering radius. It also computes cycle time and material estimates.
- `ai_surrogate/train_flax_model.py`
  Generates physics-labeled path samples and trains a Flax MLP surrogate to approximate loss and manufacturability terms at inference speed. The implemented AI path is currently Flax-based, while the broader architecture is compatible with other JAX-native frameworks.

## Core Optimization Logic

The optimizer solves for:

- Two internal Bezier control points in surface parameter space
- A continuous deposition and thickness scale

The loss combines:

- `compliance_proxy`
  A stiffness-oriented proxy that rewards alignment of the path tangent with the local preferred load direction and stress concentration field.
- `steering_penalty`
  An exponential penalty when the local steering radius drops below the minimum IFP limit. The CLI default is `50 mm`.
- `thickness_penalty`
  A differentiable deposition-density map across the surface that penalizes unstable build-ups and non-uniform laydown.
- `keepout_penalty`
  Smoothly discourages paths from entering holes and integrated cutouts.
- `boundary_penalty` and `smoothness_penalty`
  Keep the route on the valid surface and prevent erratic control polygons.

This is the core claim of the repo: manufacturability is not a post-processing filter. It is part of the optimization state.

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

You can also switch to a tubular demonstration surface:

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
  Robot-consumable kinematic array with XYZ plus tool orientation vectors
- `outputs/ifp_preview.png`
  Preview showing the optimized path, thickness field, and steering-radius compliance

After `train-surrogate`, the repository writes:

- `outputs/surrogate_dataset.npz`
- `outputs/surrogate_params.msgpack`
- `outputs/surrogate_metrics.json`

## Example Notebooks

- `examples/drone_frame_cutout_avoidance.ipynb`
  Step-by-step plate-with-hole workflow from geometry inspection through optimization, toolpath export, and material/process metrics.
- `examples/robotic_limb_optimization.ipynb`
  Step-by-step tubular workflow from limb geometry through optimized routing, robotic kinematics export, and manufacturing metrics.
- `examples/drone_plate.obj`
  Lightweight mesh asset used by the cutout-avoidance demo.

## How This Fits An Autonomous Factory OS

This repository is best understood as an upstream planning service inside an autonomous composites software stack:

1. A component surface and load case enter the planner.
2. The differentiable JAX core solves a physically admissible IFP route.
3. The exporter emits robot-ready kinematic arrays and process metrics.
4. The same physics engine generates supervised data for Flax models that can later accelerate screening, sequencing, and autonomous planning decisions.

That makes the repo useful in two modes:

- `physics mode`
  High-fidelity differentiable optimization for new parts, load cases, or constraint studies
- `surrogate mode`
  Fast learned approximation once the design space has been sampled

## Current Scope

This is a pitch-ready technical prototype, not a finished production backend. The current implementation is intentionally scoped around analytic surfaces that keep the optimization differentiable and stable:

- 2.5D plate-with-hole parts for cutout-routing demonstrations
- Cylindrical parts for tubular IFP routing
- Single-path optimization with continuous deposition scaling

That scope is deliberate. It proves the right architecture for an autonomous-factory direction before scaling into richer imported geometries, multi-course planning, collision-aware robot kinematics, and tighter factory-software integration.

## Repo Layout

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

## Recommended Next Extensions

- Replace the current stiffness proxy with a differentiable shell or reduced-order laminate response.
- Extend from one Bezier course to multi-course cooperative routing.
- Add robot singularity and head-clearance constraints alongside the current surface-normal export.
- Feed surrogate uncertainty back into the optimization loop for active learning.

That is the story this repository is meant to tell: a differentiable planning layer that is robot-aware, manufacturability-aware, and AI-native from day one.
