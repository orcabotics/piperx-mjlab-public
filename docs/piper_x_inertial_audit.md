# PiPER-X mass and inertia audit

Date: 2026-08-12

## Result

The upstream AgileX PiPER-X masses and centers of mass are retained. Six
upstream inertia tensors are replaced because they are mathematically
incompatible with the declared mass, center of mass, and complete STL
envelope: `link2`, `link3`, `link4`, `link6`, `gripper_link1`, and
`gripper_link2`.

The replacement is the homogeneous closed-STL inertia tensor scaled to the
upstream mass. This is a physically feasible and reproducible geometry prior;
it is not presented as a factory-identified rigid-body parameter. The
conversion script applies the same correction whenever the asset is rebuilt.

## Sources and mass check

- Upstream model: <https://github.com/agilexrobotics/agx_arm_urdf>,
  `piper_x/urdf/piper_x_description.urdf` and
  `piper_x_with_gripper_description.xacro`.
- AgileX catalog specification: PiPER-X mass 4.5 kg. The PiPER quick-start
  manual specifies 0.5 kg for the optional gripper.
- The local link decomposition is 4.347 kg for base through flange and
  0.500 kg for the three gripper links. The complete 4.847 kg model is 3.1%
  below the two rounded catalog numbers combined (5.0 kg), so there is no
  evidence for a broad mass rescaling.

## Why the upstream inertia is invalid

For a coordinate supported on `[a,b]` with mean `mu`, its variance cannot
exceed `(mu-a)(b-mu)`. Applying this to the two mesh coordinates perpendicular
to a rotation axis gives a hard upper bound on the corresponding diagonal
inertia. The largest upstream bound violations are:

| Link | Worst upstream inertia / geometric upper bound |
|---|---:|
| `link2` | 2.18 |
| `link3` | 2.07 |
| `link4` | 1.30 |
| `link6` | 555.38 |
| each gripper finger | 1.96 |

`link6` is the clearest failure: a 0.007 kg, 0.038 m-diameter part has an
upstream principal inertia of `1.521e-3 kg m^2`, equivalent to a 0.466 m
radius of gyration. Its mesh-derived replacement has principal inertias
`[6.397e-7, 6.397e-7, 1.203e-6] kg m^2`.

All replacement tensors are positive definite, satisfy the rigid-body
triangle inequalities, and fall below the mass/COM/envelope upper bound.

## Hardware-log comparison

Two local hardware captures were used:

- `logs/real_arm_yoyo_control/20260804T203416`: static prepared-pose and
  open-loop capture. Its measured-minus-model baseline was
  `[+0.146, +0.100, -0.908, -0.215, -0.022, +0.018] N m`. The mostly constant
  J3/J4 error cannot identify individual link masses from one pose and also
  includes drive friction and torque-sensor/current-estimator bias.
- `logs/calib_arm_only/20260805T195625`: 5,338-sample, 200 Hz no-payload
  excitation. Driven-joint acceleration was recovered from the torque stored
  by the old model, with non-driven acceleration fixed to zero. A constant
  per-joint residual was removed so motor bias does not masquerade as inertia.

| Model | J2-J4 centered pooled RMSE | J2 corr. | J3 corr. | J4 corr. |
|---|---:|---:|---:|---:|
| upstream tensors | 0.902 N m | 0.460 | 0.699 | 0.285 |
| physical mesh fallbacks | 0.885 N m | 0.471 | 0.707 | 0.335 |

The correction gives a modest improvement rather than a dramatic fit. That is
expected: this log excites only one coupled vertical path, measured torque
contains reducer friction and current-estimator bias, and commanded rather
than directly measured acceleration generated the original model torque.

For identification-grade inertias, collect multi-pose, multi-axis excitation
with measured joint acceleration and fit rigid-body inertial parameters jointly
with Coulomb/viscous friction and torque offsets. The present correction's
claim is narrower: it removes impossible tensors, preserves the supported
mass/COM data, and does not degrade the available hardware correspondence.
