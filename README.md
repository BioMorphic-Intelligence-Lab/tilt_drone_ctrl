# Wall-pinned tilted-rotor drone

A quadrotor with **fixed-tilt rotors** holds a rigid beam against a wall.  While
the tip does not slip, the contact behaves as a **spherical joint**, so the whole
beam+drone assembly is one rigid body rotating about a fixed point: a 6-state
nonlinear system whose states are the joint's three angles and their rates.

The simulation is:

* **plant** — Genesis 1.3.3, integrating the URDF's articulated dynamics.  The
  only thing the controller hands it is four rotor forces.
* **controller** — LQR, relinearised at every controller step and solved with
  python-control's `lqr`.
* **monitor** — the contact force at the joint is reconstructed each step and
  checked against the Coulomb friction cone, which is the condition under which
  the spherical-joint idealisation is valid in the first place.
* **visualisation** — Genesis's own 3D view (thrust arrows on each rotor, the
  contact-force arrow at the tip), plus matplotlib diagnostics.

```
conda env create -f environment.yml
conda activate tilt_drone

python scripts/run_sim.py --vis --plot          # 3D view + diagnostic plots
python scripts/run_sim.py --plot --mu 0.12      # tighten friction until it slips
python scripts/run_sim.py --traj trajectories/three_steps.traj.txt --plot
python scripts/run_sim.py --plot --disturb-t 2 --disturb-torque 0 0 2.5
python scripts/run_sim.py --vis --record run.mp4
pytest tests/ -q
```

## The model

Frames: the wall occupies `x < 0` with outward normal `n = +x`; the pivot is at
the robot's base; the body frame's origin *is* the pivot, `+x` runs along the
beam away from the wall, `+z` is up.

The three URDF revolute joints (axes x, y, z, all coincident at the pivot) stand
in for the spherical joint, so `R(q) = Rx(q0) Ry(q1) Rz(q2)` and the state is
exactly what Genesis reports for those dofs — no estimator, no conversion.

```
q_dot     = T(q) omega                      T(q) = E(q)^-1,  omega_body = E(q) q_dot
I_O w_dot = -omega x (I_O omega) + tau_g(q) + B_tau u
tau_g(q)  = r_cm x (m R(q)^T g)
B_tau[:,i] = p_i x d_i  -  spin_i * kappa * d_i
```

`I_O` is the inertia about the *pivot* (parallel-axis over every link below the
joint), and `p_i`, `d_i` are each rotor's position and thrust axis in the body
frame.  Because the tilt is fixed, `B_tau` is **constant in the body frame**; the
state dependence of the actuation onto the joint is the rotation of that wrench
into the world,

```
tau_joint(q, u) = R(q) B_tau u,     f_joint(q, u) = R(q) D u
```

which is what `DroneWallModel.joint_wrench_world` returns.

### Contact force and the friction cone

From linear momentum over the whole body,

```
f_c = m a_cm - m g - R D u,    a_cm = R (w_dot x r_cm + omega x (omega x r_cm))
```

`f_c` is the force the joint applies *to the drone*; the drone presses on the
wall with `-f_c`.  The cone test is `f_n = f_c.n > 0` (contact in compression,
not tension) and `|f_t| <= mu f_n`.  Genesis exposes no joint-constraint force,
so `f_c` is reconstructed from the measured state and the applied thrust; the
test suite checks it against the angular-momentum balance about the centre of
mass, an equation not used in deriving it.

## The rotor tilt

Every rotor sits at a **40 deg mount angle** — the total angle between its
thrust axis and the airframe z axis, which is the hardware limit.  Side tilt
spends the same budget as forward tilt (`mount = acos(cos(fwd) * cos(side))`), so
the two pairs split it differently:

| rotor pair | forward (into the wall) | side | mount |
|---|---|---|---|
| front (nearer the wall) | 40.00 deg | 0 | 40.00 deg |
| hind | 38.93 deg | +/-10 deg outboard | 40.00 deg |

The beam is **0.50 m** from the pivot to the drone body.  The tilt lives entirely
in the `rpy` of each `*_mount` joint in `urdf/tilt_drone.urdf` and is parsed back
out by `tilt_drone/urdf_model.py`, so the plant and the controller model can
never disagree about it.  Edit the URDF directly, or change the constants in
`scripts/gen_urdf.py` and regenerate — the generator prints each mount angle
and warns if one exceeds the cap.  Its CLI takes per-rotor angles and beam length:

```
python scripts/gen_urdf.py --out urdf/variant.urdf --beam 0.5 \
       --beta 40 40 38.93 38.93 --side 0 0 10 -10
```

Why those two tilts:

* **Forward tilt** turns the null space of the torque balance into a commandable
  push.  Holding the attitude only constrains torque about the pivot, which
  leaves a one-parameter family of rotor forces; the forward tilt makes the wall
  force vary along that family, so push becomes a design input.
* **Side tilt on the hind pair** buys authority about the pivot's z axis — the
  axis a flat quad has almost none of, since vertical thrusts at `p_i` produce
  torque `(y_i f, -x_i f, 0)` and only propeller drag touches z.  A sideways
  thrust component at lever arm `x_i` gives `tau_z = x_i f_y`, and the hind pair
  has the longest arm.  Tilting the two rotors to *opposite* sides makes a
  couple: z-torque from differential thrust, no net side force at equal thrust.
  Removing it costs a factor of 3.4 in the trim map's condition number (15.4 ->
  51.8), which `test_side_tilt_is_what_makes_the_map_well_conditioned` guards.
  At 40 deg the front rotors are no longer irrelevant to yaw — a thrust axis
  leaning that far makes z-torque through its own lateral offset `y_i*sin(beta)`.
  That coupling is why side tilt on *all four* rotors in an outboard pattern is
  worse than on the hind pair alone: it reinforces the same left/right pattern
  roll already uses, re-coupling the two axes.

## What the push force actually costs you

Two limits, both reported at startup and both strongly attitude-dependent:

1. **Torque-feasible range** — the push forces reachable while balancing gravity
   with `0 <= u <= u_max`.
2. **Cone-feasible range** — of those, the ones that do not slip.  Tilting the
   rotors into the wall couples push to lift: pressing harder means thrusting
   harder, so the tip carries a vertical shear of `lift - weight` that friction
   has to absorb.

For the shipped geometry at `mu = 0.4` (slip-free range, N):

| rotor limit | airframe level | pitched 12.5 deg nose-up |
|---|---|---|
| 8 N | [9.63, 13.61] | [13.25, **16.55**] |
| 10 N | [9.63, 14.79] | [13.25, 18.00] |
| 16 N | [9.63, 16.46] | [13.25, 19.79] |

Pitching the airframe *adds* to the fixed rotor tilt — 40 + 12.5 = 52.5 deg of
effective thrust angle — which is where most of the gain comes from.  Pitching
the other way subtracts from it and the achievable push collapses.  A push
command that is comfortable at one attitude is unreachable at another, which is
why `--f-push` defaults to `auto` (the middle of the slip-free range at the
current reference) and prints what it chose.  Pass a number to command one
explicitly; you will be warned if it is out of range.

Two things do *not* help: raising `u_max` past ~16 N (the hind rotors reach zero
thrust and the lever-arm geometry caps you), and raising `mu` (friction stops
binding once the tilt is this large — the cone then only stops you pressing too
*softly*).  What does help is a shorter beam: the balanced lift depends on
`x_cm / x_mean_rotor`, which barely moves, but a shorter beam widens the *spread*
of the rotor lever arms, so shifting thrust onto the front pair amplifies lift
much more.  Halving the beam from 0.8 to 0.4 m raises the ceiling from 17.0 to
20.7 N; 0.50 m is the shipped compromise.

The trim solve itself is a bounded weighted least-squares that prioritises torque
balance over push, so an impossible request tilts the beam correctly and presses
too softly, rather than falling over.

Running at the very top of a range puts the trim on a rotor bound, leaving the
LQR one-sided authority there; expect sustained saturation during transients.
Backing off a few percent restores headroom — but shrinks the cone margin, since
`mu*f_n` scales with the push.

## Step-reference trajectories

`--traj FILE` drives the joint through a sequence of step references.  One
waypoint per line — start time, then the three reference angles in degrees —
with `#` comments and blank lines ignored:

```
#  t     phi_x   theta_y   psi_z
   0.0     0.0      0.0      0.0
   2.5     0.0      2.0      5.0
   5.0    -4.0     -2.0     -5.0
```

`trajectories/three_steps.traj.txt` is that example.  Run it with

```
python scripts/run_sim.py --traj trajectories/three_steps.traj.txt --plot
```

The reference is a zero-order hold: each waypoint's angles apply until the next
waypoint's start time, so the controller sees real step inputs.  A waypoint at
`t <= 0` sets the starting reference (otherwise `--q-ref` does), start times must
strictly increase, and `--T` defaults to 3 s past the last waypoint so the final
step has time to settle.

Every waypoint re-solves the trim and re-reports both feasible push ranges,
because they move with attitude — in the example, the nose-up second step drops
the slip-free range to `[2.09, 2.32] N` while the nose-down third step raises it
to `[3.15, 3.38] N`.  With `--f-push auto` each step takes the middle of its own
range; with an explicit `--f-push` you get a warning at any step that cannot hold
it.  `--step-t` / `--q-ref2` remain for a single step and are ignored when
`--traj` is given.

## Layout

| file | role |
|---|---|
| `urdf/tilt_drone.urdf` | the robot; **source of truth** for geometry and tilt |
| `scripts/gen_urdf.py` | regenerates it (keeps inertia tensors consistent) |
| `tilt_drone/urdf_model.py` | URDF -> mass, `r_cm`, `I_O`, rotor geometry, `B_tau` |
| `tilt_drone/dynamics.py` | the 6-state model, its Jacobians, the contact force |
| `tilt_drone/trim.py` | feed-forward solve + the two feasible-push ranges |
| `tilt_drone/lqr_ctrl.py` | per-step relinearisation + `control.lqr` |
| `tilt_drone/contact.py` | friction-cone test and rate-limited warnings |
| `tilt_drone/trajectory.py` | step-reference trajectory files |
| `trajectories/` | example trajectory (three setpoints) |
| `tilt_drone/plant.py` | Genesis scene, force application, viewer/recording |
| `tilt_drone/plotting.py` | diagnostic figure |
| `scripts/run_sim.py` | the driver |

## Validation

`tests/test_model_vs_genesis.py` is the gate that matters: it compares the
analytic model's **acceleration** against Genesis's over a single 1e-5 s step, at
random states and inputs and one rotor at a time.  Comparing trajectories would
prove nothing here — the free system tumbles chaotically, so any two integrators
diverge whether or not the model is right.

Three Genesis 1.3.3 behaviours that silently break model/plant agreement, all
handled in `plant.py` and worth knowing if you extend this:

* **`armature` defaults to 0.1** on every dof — fictitious rotor inertia the
  analytic model knows nothing about.  Zeroed at build time.
* **`local=True` on `apply_links_external_force` rotates by the link's *inertial*
  frame** (a principal-axis frame), not the link frame.  A rotor disc has two
  equal principal moments, so that frame is arbitrary — here it mapped local `+z`
  onto world `-y`.  Thrust is therefore built in world coordinates from
  `get_links_quat` and applied with `local=False`.
* **`scene.add_camera(debug=True)`** is required for the thrust and contact-force
  arrows to appear in a recording; the viewer shows them either way.

Fixed links are merged by Genesis for speed, so the rotor links are preserved
explicitly via `links_to_keep` — they are the frames the thrust is applied at.

## Assumptions and limits

* The tip is *pinned*, not simulated as a contact: the cone test tells you when
  that assumption has broken, it does not model the slip that would follow.
* Rotor forces are commanded directly (no motor dynamics, no thrust lag), and
  `kappa` is a constant drag/thrust ratio.
* Aerodynamics near the wall (ground/wall effect) are not modelled.
* With every rotor canted 40 deg the airframe cannot hover level in free flight:
  held level it fights `mg*tan(40 deg)` = 13.2 N of horizontal bias, and hovering
  costs 31% more thrust.  This is a contact-phase geometry, not a flight trim.
* The x-y-z joint chain has a gimbal singularity at `q1 = +/- 90 deg`; the run
  warns below `|cos q1| = 0.2` and stops.
