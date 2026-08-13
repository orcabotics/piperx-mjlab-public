# PiPER-X MJLab

A minimal [MJLab](https://github.com/mujocolab/mjlab) environment for the
AgileX PiPER-X arm, with a physically feasible inertial model, two scripted
pick-and-place demos, and two PPO pick-cube tasks:

- `state`: reads the simulated cube position;
- `vision`: locates the cube from overhead RGB-D only. The vision observation
  does not contain the cube state.

The scripted demos use the same small Cartesian waypoint controller and
MJLab's damped-least-squares differential IK. The RL tasks share MJLab's
lift-cube reward and a seven-dimensional joint-position action.

## Requirements

- Linux, Python 3.10–3.13;
- NVIDIA GPU recommended (CPU evaluation is supported by MJLab but slower);
- MJLab 1.6.0, installed automatically.

```bash
git clone <this-repository>
cd piperx-mjlab
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The first run compiles and caches MuJoCo-Warp kernels, which can take a few
minutes.

## Run

```bash
# Ground-truth object state, native window when a display is available.
piperx-demo-state

# RGB-D object localization. Viser shows the camera image in a browser.
piperx-demo-vision --viewer viser

# One command with an explicit observation mode.
piperx-pick-place --observation vision --viewer none --steps 1200
```

Viewer choices are `auto`, `native`, `viser`, and `none`. Use `--device cpu`
for CPU execution.

The environment is also directly reusable:

```python
from mjlab.envs import ManagerBasedRlEnv
from piper_mjlab import make_env_cfg

env = ManagerBasedRlEnv(make_env_cfg("state"), device="cuda:0")
observations, _ = env.reset(seed=0)
```

## Reinforcement learning

The two PPO tasks differ only in the actor observation:

| Task | Actor observation | Critic observation |
|---|---|---|
| `state` | proprioception + cube/goal relative positions | same as actor |
| `vision` | proprioception + raw 64×64 RGB-D | privileged state |

The vision actor is an end-to-end Spatial-Softmax CNN. It does not call the red
cube estimator and does not receive simulated object state. Ground truth is
used only by the asymmetric critic and reward during training.

```bash
# Defaults: 1024 state environments or 64 camera environments, 3000 PPO updates.
piperx-rl-train state
piperx-rl-train vision

# Play a locally trained policy.
piperx-rl-play state --checkpoint-file /path/to/model_2900.pt
piperx-rl-play vision --checkpoint-file /path/to/model_2900.pt
```

The commands expose MJLab's normal CLI overrides; for example, use
`--env.scene.num-envs 16 --agent.max-iterations 2` for a short state smoke
run. Checkpoints are intentionally not bundled.

## Layout

```text
src/piper_mjlab/
├── assets/agilex_piper_x/   # URDF and meshes
├── robots/piper_x_constants.py
├── env.py                   # scene, observations, Cartesian actions
├── demo.py                  # RGB-D estimator and shared state machine
├── rl.py                    # state/vision lift task and PPO configs
└── scripts/convert_piper_x_urdf.py
```

## Model status

The per-link masses and centers of mass come from AgileX's official PiPER-X
URDF. Six upstream inertia tensors were physically impossible; they were
recomputed from the corresponding closed STL meshes at the official masses.
All tensors are now positive definite, satisfy the principal-moment triangle
inequalities, and fit their mesh bounds.

The arm plus flange is `4.347 kg`; the gripper is `0.500 kg`; the complete
model is `4.847 kg`. This is close to the `4.5 kg` PiPER-X catalog mass plus
the `0.5 kg` gripper. Hardware torque-log replay improved modestly after the
repair, but these are geometry-derived parameters rather than factory system
identification results. See [the inertial audit](docs/piper_x_inertial_audit.md).

Actuator PD gains are simulation gains, not identified hardware gains.

## Validation

```bash
pip install -e ".[dev]"
pytest -q
piperx-demo-state --viewer none --steps 1200
piperx-demo-vision --viewer none --steps 1200
piperx-rl-train state --env.scene.num-envs 16 --agent.max-iterations 1
piperx-rl-train vision --env.scene.num-envs 4 --agent.max-iterations 1
```

Validated on MJLab 1.6.0 / MuJoCo 3.11 / RTX 5090: final cube errors were
9 mm (`state`) and 14 mm (`vision`) in the final clean-tree run.
The default RL configurations also completed a full PPO update with 1024
state environments and 64 RGB-D environments respectively.

## License and provenance

Project code is MIT licensed. The PiPER-X URDF and meshes are derived from
[agilexrobotics/agx_arm_urdf](https://github.com/agilexrobotics/agx_arm_urdf)
and retain their upstream MIT license in
`src/piper_mjlab/assets/agilex_piper_x/LICENSE`.

MJLab is an Apache-2.0 dependency and is not vendored here.
