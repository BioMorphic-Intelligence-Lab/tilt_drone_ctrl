# Wall-pinned tilted-rotor drone

A quadrotor with **fixed-tilt rotors** holds a rigid beam against a wall.  While
the tip does not slip, the contact behaves as a **spherical joint**, so the whole
beam+drone assembly is one rigid body rotating about a fixed point: a 6-state
nonlinear system whose states are the joint's three angles and their rates.

The simulation is:

* **plant** — Genesis 1.3.3, integrating the URDF's articulated dynamics.  The
  only thing the controller hands it is four rotor forces.
* **controller** — feed-forward trim plus an LQR, relinearised at every
  controller step and solved with python-control's `lqr`.  Explained from
  scratch in [The controller](#the-controller) below.
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
pytest tests/ -q                                # 31 tests, ~20 s
```

Every run writes `logs/run.npz` (all logged signals) and `logs/run.png` (the
six-panel diagnostic figure); `--record` adds an mp4.

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

For the shipped geometry the numbers `run_sim.py` prints at startup are

```
m = 1.600 kg,  r_cm = [0.4687, 0, 0.0038] m,  weight = 15.70 N
I_O = diag-ish [0.0086, 0.3752, 0.3825] kg m^2  (with I_xz = -0.0031)
rank(B_tau) = 3
```

Note how small `I_xx` is (0.0086 vs 0.375): rolling *about the beam* is nearly
free, while pitching and yawing swing the whole 0.5 m beam.  That 44:1 spread is
the single most important number for understanding the controller's behaviour —
it is why the roll channel gets the biggest feedback gains and the fastest
closed-loop pole.

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

## The controller

This section assumes you know what a differential equation and a matrix are, and
nothing else.  Everything here lives in `tilt_drone/trim.py` (feed-forward) and
`tilt_drone/lqr_ctrl.py` (feedback, 64 lines).

### What the controller is up against

Let go of this robot and it falls over.  Concretely, with all four rotors off and
the assembly level, the model reports an angular acceleration of **19.6 rad/s²**
about the pitch axis — gravity pulling the 1.6 kg mass, whose centre sits 0.47 m
out along the beam, straight down.  Worse, nothing in the system removes energy:
there is no damping term anywhere, so the beam does not settle, it swings.

Linearising the model about "level, at rest" and looking at the eigenvalues of
the resulting `A` matrix makes this precise:

```
open-loop eigenvalues:  0, 0, ±0.305j, +0.396, -0.396   [1/s]
```

Read that as four separate statements:

* the two zeros are **integrators** — the joint angles are the integral of the
  joint rates, and nothing pulls them back;
* `±0.305j` is a **purely oscillatory** mode: a pendulum with zero damping, which
  rings forever;
* `+0.396` is an **unstable** mode: any error along it grows like `e^(0.396 t)`,
  doubling every 1.75 s.  It is there because the centre of mass sits 3.8 mm
  *above* the pivot (the rotors are mounted 3 cm above the body centreline), so
  the assembly is a very flat inverted pendulum.

A positive eigenvalue means the plant cannot be run open-loop, at all, ever.
`tests/test_control.py::test_open_loop_is_unstable` asserts exactly this, so the
claim stays true if you change the geometry.

### The two halves: feed-forward and feedback

The controller is one line, in `LQRController.__call__`:

```python
u = u_ff - K @ (x - x_ref)
u = np.clip(u, 0.0, u_max)
```

* `u_ff` — the **feed-forward** (or *trim*) thrust: the four rotor forces that,
  if the robot were already exactly at the reference and perfectly still, would
  hold it there.  It does the bulk of the work: for the default reference it is
  `[5.65, 5.65, 4.00, 4.00] N`, against a 15.7 N weight.
* `K @ (x - x_ref)` — the **feedback** correction, proportional to how wrong the
  current state is.  It only ever has to handle the *error*, which is small.

Splitting the two is worth doing for a very practical reason: feedback that also
had to carry the 7.36 N·m of gravity torque would need a large steady error to
generate it (that is the classic proportional-control offset).  With the trim
carrying gravity, the feedback can be zero at the reference and still hold.

The state and reference are

```
x     = [q0, q1, q2, wx, wy, wz]        3 joint angles [rad] + 3 body rates [rad/s]
x_ref = [q_ref(3), 0, 0, 0]            an attitude, with all three rates zero
```

so "reference" here always means *stand at this attitude and stop moving*.

### Half one: the trim solve

The pivot carries the weight — that is what a spherical joint does — so the only
equilibrium condition on the 3-dof system is **zero net torque about the pivot**.
Three equations, four rotors: the solution is not unique, and the leftover
freedom is a whole line of thrust combinations that all hold the same attitude.

The forward rotor tilt is what turns that useless freedom into something you can
sell.  Moving along the line changes how hard the beam presses into the wall, so
the trim solve adds a fourth equation for the push force and gets a square
system:

```
[   B_tau    ] u = [ -tau_g(q_ref) ]     3 rows: torque about the pivot must cancel gravity
[ n^T R(q) D ]     [   -f_push     ]     1 row: press this hard into the wall
```

Two practical wrinkles, both handled in `solve_trim`:

* **Propellers only push.** The solve is bounded, `0 <= u <= u_max`, via
  `scipy.optimize.lsq_linear` rather than a plain matrix inverse.
* **Not every request is possible.** The four equations are weighted 100:1 in
  favour of the torque rows, so when the push force and the attitude cannot both
  be had, the solver keeps the robot upright and presses too softly, instead of
  pressing correctly and falling over.  `TrimResult.feasible` tells you which
  happened, and the run prints a warning.

`trim.report()` prints the achieved push, the torque residual (typically `1e-15`
N·m — exact, when the request is feasible) and the *condition number* of that
4×4 matrix, 15.4 for the shipped geometry.  Condition number is a "how invertible
is this, really" score: 1 is perfect, and large values mean some directions of
the requested wrench need enormous, cancelling rotor forces.  Section
[The rotor tilt](#the-rotor-tilt) is essentially the story of getting that number
down.

### Half two: linearising the plant

LQR is a method for *linear* systems, and this plant is not linear — `R(q)`,
`T(q)` and the `omega × I omega` Coriolis term all see to that.  The standard
move is to linearise: pick an operating point, and ask how the derivative of the
state changes for small departures from it.  That gives

```
d/dt (x - x0) ≈ A (x - x0) + B (u - u0)
```

with `A = ∂f/∂x` and `B = ∂f/∂u`, both evaluated at the operating point.
`DroneWallModel.linearize` computes `A` by central finite differences and `B` in
closed form (the dynamics are exactly affine in `u`, so `B` is simply
`[0; I_O^-1 B_tau]`, and `test_analytic_B_matches_finite_difference` checks it).

The matrices at the default reference, rounded:

```
        q0      q1      q2      wx      wy      wz
A =  [   0       0       0       1       0       0  ]   q0_dot
     [   0       0       0       0       1       0  ]   q1_dot
     [   0       0       0       0       0       1  ]   q2_dot
     [ -0.09     0       0       0       0       0  ]   wx_dot
     [   0     +0.16     0       0       0       0  ]   wy_dot
     [ -19.2     0       0       0       0       0  ]   wz_dot

        fl      fr      hl      hr
B =  [   0       0       0       0    ]
     [   0       0       0       0    ]
     [   0       0       0       0    ]
     [ 14.70  -14.70   11.86  -11.86  ]   wx_dot
     [ -0.77   -0.77   -1.37   -1.37  ]   wy_dot
     [  0.34   -0.34    0.67   -0.67  ]   wz_dot
```

Three things are worth reading straight off these:

1. **The top-right identity block in `A`** is just "angles are the integral of
   rates".  It is bookkeeping, not physics.
2. **The bottom-right block of `A` is exactly zero** — no damping.  The Coriolis
   term is quadratic in `omega`, so its derivative vanishes at `omega = 0`.
   Confirmation of the point above: this system will not calm itself down.
3. **The top three rows of `B` are zero.** You cannot push on an angle, only on
   an acceleration; the effect on position is two integrations away.  This is
   why any sensible controller feeds back *rates* as well as angles, and why
   `Q` below has entries for both.

The `-19.2` in `A` is the roll angle feeding into yaw acceleration.  It is large
because `I_O^-1` has a big `xz` off-diagonal term, which in turn is because
`I_xx` is tiny.  In plain terms: roll and yaw on this robot are not separate
problems, and a controller designed one axis at a time would fight itself.  LQR
handles the coupling for free — that is most of the reason to use it here.

### Half two, continued: what LQR actually computes

You want a matrix `K` such that `u = -K (x - x_ref)` drives the error to zero.
There are infinitely many stabilising `K`s.  LQR picks one by asking you to
score the outcomes instead of the gains: define a running cost

```
J = ∫ [ eᵀ Q e  +  vᵀ R v ] dt        e = x - x_ref,   v = u - u_ff
```

and take the `K` that minimises it.  `Q` prices *being wrong*; `R` prices
*working hard*.  The minimiser is found by solving the algebraic Riccati
equation, which `control.lqr(A, B, Q, R)` does for you in one call; you never
touch the Riccati equation by hand.

The defaults are diagonal:

```
Q = diag(40, 40, 40,  4, 4, 4)      --Q, angle errors then rate errors
R = 0.2 * I(4)                      --R, one number, same for every rotor
```

Only the *ratios* matter — scaling both by 10 gives a `K` identical to machine
precision.  Because the cost is quadratic, read them as: one radian of angle
error costs the same as `sqrt(40/4)` ≈ 3.2 rad/s of rate error, or `sqrt(40/0.2)`
≈ 14 N of thrust deviation.  Tuning rules of thumb:

| you want | do this | you get |
|---|---|---|
| faster response | raise `--Q`, or lower `--R` | bigger `K`, more thrust, earlier saturation |
| less overshoot / ringing | raise the *rate* entries of `--Q` (last three) | more damping |
| gentler actuation | raise `--R` | slower, softer, more error under disturbance |

For the shipped defaults the resulting gain is

```
        q0      q1      q2      wx      wy      wz
K =  [ 25.33  -4.91   -9.86    2.87   -1.98   -5.87 ]   rotor fl
     [-25.33  -4.91    9.86   -2.87   -1.98    5.87 ]   rotor fr
     [  8.06  -8.77    1.68    2.05   -3.54    0.24 ]   rotor hl
     [ -8.06  -8.77   -1.68   -2.05   -3.54   -0.24 ]   rotor hr
```

Units are N/rad for the first three columns and N/(rad/s) for the last three.
The structure is the physics, made visible without anyone designing it in:

* **Column `q0` (roll)** is antisymmetric left-to-right — roll is corrected with
  *differential* thrust across the airframe, mostly on the front pair.
* **Column `q1` (pitch)** is symmetric left-to-right, and the hind entries are
  nearly twice the front ones — pitch is corrected by adding thrust with the
  weight biased towards the long-lever hind pair, which nets out as a pitch
  torque.
* **Column `q2` (yaw)** is antisymmetric again, and its front and hind entries
  have *opposite* signs: yaw is corrected against roll, which is that `-19.2`
  coupling being cancelled.

Closing the loop with this `K` moves the eigenvalues from the open-loop list
above to

```
closed-loop eigenvalues:  -119.5,  -9.36,  -3.37 ± 2.97j,  -3.35,  -3.16   [1/s]
```

All real parts negative, so the error decays.  The slowest is `-3.16`, i.e. a
time constant of `1/3.16` = 0.32 s, and errors take roughly five of those to
disappear.  That matches what the simulation actually does: from the default
8.8 deg initial offset, the error is under 1 deg at t = 1.18 s and under 0.1 deg
at t = 1.83 s.  The very fast `-119.5` pole is the roll channel — cheap to move,
because `I_xx` is small.

`test_closed_loop_is_stable_at_the_reference` re-derives those eigenvalues and
asserts they are all in the left half-plane, and
`test_closed_loop_converges_in_genesis` runs the whole thing against the real
Genesis plant for 3 s and requires the attitude error to fall below 0.1 deg.

### Why the gain is recomputed every step

`A` was computed *at* the reference, so `K` is strictly only right near it.  How
wrong does it get?  Recomputing `K` at displaced attitudes and comparing:

| displacement from the reference | `‖K − K_ref‖ / ‖K_ref‖` |
|---|---|
| 10 deg on all three axes | 21 % |
| (20, −20, 30) deg | 75 % |

So for the few-degree excursions this robot is meant to do, a frozen gain would
be fine — but the cost of not freezing it is one small Riccati solve per step, so
the controller just does it.  `LQRController.__call__` relinearises at the
*current measured state* every control step and re-solves — the state-dependent
Riccati equation (SDRE) approach, i.e. gain scheduling where the schedule is the
state itself rather than a lookup table.  Two flags let you go back:

* `--freeze-gain` — linearise at the reference instead of the current state
  (still re-solved, but the answer stops changing).
* `--relin-every N` — re-solve only every `N`th control step.

If the Riccati solve ever fails — it can, if you push the geometry somewhere
non-stabilisable — the controller keeps the previous gain, counts the failure,
and prints a warning; failing on the *very first* call is fatal, because there is
no previous gain to fall back on.

### Saturation, and the things this controller does not do

The output is clipped to `0 <= u <= u_max`, because propellers cannot pull and
cannot exceed their limit.  LQR knows nothing about that clipping — the maths
assumed unbounded inputs — so if the clip is active for long, the guarantees are
void and the robot may not recover.  The controller counts saturated steps and
prints the total; a healthy run reports `LQR: 0 saturated steps`.  This is also
the reason to avoid trimming right at the edge of the feasible push range: a trim
sitting on a rotor bound leaves the feedback with authority in one direction only.

Three honest limitations:

* **No integral action.** There is no state that accumulates error, so a constant
  modelling error (a mis-measured mass, an unmodelled breeze) would leave a
  constant offset.  Here the model and the plant are the same URDF, so the final
  error is genuinely 0.000 deg — do not read that as evidence the design is
  robust.  On hardware you would add an integrator on the angle errors, which
  means extending the state to nine and letting LQR weight the new entries.
* **The push force is open-loop.** Feedback controls attitude only; the push
  comes entirely from `u_ff`.  Nothing measures the actual contact force and
  corrects it, because the plant does not expose one — `f_c` is reconstructed
  from the model, so feeding it back would be feeding back an assumption.
* **Full state, no noise, no delay.** The state is read exactly from the
  simulator at 250 Hz.  No estimator, no sensor noise, no actuator lag.

### The knobs

| flag | default | what it does |
|---|---|---|
| `--Q q0 q1 q2 wx wy wz` | `40 40 40 4 4 4` | LQR state weights |
| `--R` | `0.2` | LQR input weight, same for all four rotors |
| `--ctrl-hz` | `250` | controller rate; the sim runs at `1/--dt` = 500 Hz |
| `--relin-every` | `1` | re-solve the Riccati equation every N control steps |
| `--freeze-gain` | off | linearise at the reference, not the current state |
| `--u-max` | `10` | per-rotor thrust limit [N]; also bounds the trim solve |
| `--f-push` | `auto` | commanded push [N]; `auto` = middle of the slip-free range |
| `--q-ref`, `--q0` | `0 0 0`, `6 4 -5` | reference and initial joint angles [deg] |
| `--mu` | `0.4` | friction coefficient at the tip (monitoring only) |
| `--kappa` | `0.016` | propeller drag/thrust ratio [m] |
| `--dt`, `--T` | `0.002`, `6` | sim step [s] and duration [s] |
| `--disturb-t` / `--disturb-dur` / `--disturb-torque` | off / `0.05` / `0 0 2` | a brief world-frame torque on the beam |
| `--traj`, `--step-t`, `--q-ref2` | — | reference changes, see below |
| `--vis`, `--record`, `--plot` | off | 3D viewer, mp4, matplotlib figure |

## The rotor tilt

Every rotor sits at a **40 deg mount angle** — the total angle between its
thrust axis and the airframe z axis, which is the hardware limit.  Side tilt
spends the same budget as forward tilt (`mount = acos(cos(fwd) * cos(side))`), so
the two pairs split it differently:

| rotor pair | mount `rpy` pitch | side | resulting mount angle |
|---|---|---|---|
| front (nearer the wall) | 40.00 deg | 0 | 40.00 deg |
| hind | 38.93 deg | +/-10 deg outboard | 40.00 deg |

The URDF is generated from those `rpy` angles; the angle `urdf_model.py` reports
back for the hind pair is 38.23 deg, not 38.93, because it measures the thrust
axis's lean out of the y-z plane and the side tilt has already eaten part of it.
Both describe the same axis — `d = (-0.619, ±0.174, 0.766)` — and the mount angle
is 40.00 deg either way.

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
  worse than on the hind pair alone (cond 27.8 vs 15.4): it reinforces the same
  left/right pattern roll already uses, re-coupling the two axes.

## What the push force actually costs you

Two limits, both reported at startup and both strongly attitude-dependent:

1. **Torque-feasible range** — the push forces reachable while balancing gravity
   with `0 <= u <= u_max`.
2. **Cone-feasible range** — of those, the ones that do not slip.  Tilting the
   rotors into the wall couples push to lift: pressing harder means thrusting
   harder, so the tip carries a vertical shear of `lift - weight` that friction
   has to absorb.

For the shipped geometry at `mu = 0.4` (slip-free range, N):

| rotor limit | airframe level | pitched 12.5 deg nose-up | pitched 12.5 deg nose-down |
|---|---|---|---|
| 8 N | [9.63, 13.61] | [13.25, **16.55**] | [6.54, 9.60] |
| 10 N | [9.63, 14.79] | [13.25, 18.00] | [6.54, 10.23] |
| 16 N | [9.63, 16.46] | [13.25, 19.79] | [6.54, 10.23] |

Pitching the airframe nose-up *adds* to the fixed rotor tilt — 40 + 12.5 = 52.5
deg of effective thrust angle — which is where most of the gain comes from.
Pitching the other way subtracts from it and the achievable push collapses.  A
push command that is comfortable at one attitude is unreachable at another, which
is why `--f-push` defaults to `auto` (the middle of the slip-free range at the
current reference) and prints what it chose.  Pass a number to command one
explicitly; you will be warned if it is out of range.

Two things do *not* help: raising `u_max` past about 13 N (beyond that the hind
rotors have reached zero thrust and the lever-arm geometry caps the level ceiling
at 16.46 N no matter how large `u_max` gets), and raising `mu` (friction stops
binding once the tilt is this large — the cone then only stops you pressing too
*softly*).  What does help is a shorter beam: the balanced lift depends on
`x_cm / x_mean_rotor`, which barely moves, but a shorter beam widens the *spread*
of the rotor lever arms, so shifting thrust onto the front pair amplifies lift
much more.  Halving the beam from 0.8 to 0.4 m raises the level ceiling from
14.6 to 18.0 N; 0.50 m is the shipped compromise.  Reproduce any of it with

```
python scripts/gen_urdf.py --out urdf/variant.urdf --beam 0.4
python scripts/run_sim.py --urdf urdf/variant.urdf --plot
```

The three extra files in `urdf/` are **superseded** snapshots from earlier in
that study, kept only for reference: `tilt_drone_40deg.urdf` and
`tilt_drone_56deg.urdf` both have the old 0.8 m beam (the latter at a 56.6 deg
mount angle, above the current 40 deg hardware cap), and all three put side tilt
on *all four* rotors in an alternating pattern rather than on the hind pair.  Do
not treat them as alternative configurations of the shipped design.

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
   7.5     4.0      2.0      5.0
```

`trajectories/three_steps.traj.txt` is that example (four waypoints: the initial
reference plus three steps).  Run it with

```
python scripts/run_sim.py --traj trajectories/three_steps.traj.txt --plot
```

The reference is a zero-order hold: each waypoint's angles apply until the next
waypoint's start time, so the controller sees real step inputs.  A waypoint at
`t <= 0` sets the starting reference (otherwise `--q-ref` does), start times must
strictly increase, and `--T` defaults to 3 s past the last waypoint so the final
step has time to settle.

Every waypoint re-solves the trim and re-reports both feasible push ranges,
because they move with attitude.  In the shipped example the slip-free range
walks from `[9.63, 14.79] N` at the level start through `[9.14, 14.08]`,
`[10.31, 14.56]` and `[9.28, 13.33]`, and the auto push command tracks the middle
of whichever range is current: 12.21, 11.61, 12.44, 11.31 N.  With an explicit
`--f-push` you get a warning at any step that cannot hold it.  `--step-t` / `--q-ref2` remain for a
single step and are ignored when `--traj` is given.

Each step is a 5–13 deg jump, and the controller settles it to under 0.1 deg in
about 1.4–1.7 s with no rotor ever saturating and the cone margin never dropping
below +2.5 N.

## Layout

| file | role |
|---|---|
| `urdf/tilt_drone.urdf` | the robot; **source of truth** for geometry and tilt |
| `urdf/tilt_drone_{shortbeam,40deg,56deg}.urdf` | superseded snapshots from the geometry study |
| `scripts/gen_urdf.py` | regenerates them (keeps inertia tensors consistent) |
| `tilt_drone/urdf_model.py` | URDF -> mass, `r_cm`, `I_O`, rotor geometry, `B_tau` |
| `tilt_drone/dynamics.py` | the 6-state model, its Jacobians, the contact force |
| `tilt_drone/trim.py` | feed-forward solve + the two feasible-push ranges |
| `tilt_drone/lqr_ctrl.py` | per-step relinearisation + `control.lqr` |
| `tilt_drone/contact.py` | friction-cone test and rate-limited warnings |
| `tilt_drone/trajectory.py` | step-reference trajectory files |
| `trajectories/` | example trajectory (three steps) |
| `tilt_drone/plant.py` | Genesis scene, force application, viewer/recording |
| `tilt_drone/plotting.py` | diagnostic figure |
| `scripts/run_sim.py` | the driver |
| `logs/` | `run.npz`, `run.png` and any recording (gitignored) |

## Validation

`tests/test_model_vs_genesis.py` is the gate that matters: it compares the
analytic model's **acceleration** against Genesis's over a single 1e-5 s step, at
random states and inputs and one rotor at a time.  Comparing trajectories would
prove nothing here — the free system tumbles chaotically, so any two integrators
diverge whether or not the model is right.

`tests/test_control.py` covers the kinematics, the tilt geometry against its
design intent, the trim solve (exact inside the feasible range; torque-first
outside it), the friction cone, and the loop — analytically, and end to end
against Genesis.  `tests/test_trajectory.py` covers the trajectory file format
and checks every waypoint reaches the controller.  31 tests, about 20 s.

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
* The controller sees the exact state, with no noise, no delay and no estimator.
* Aerodynamics near the wall (ground/wall effect) are not modelled.
* With every rotor canted 40 deg the airframe cannot hover level in free flight:
  held level it fights `mg*tan(40 deg)` = 13.2 N of horizontal bias, and hovering
  costs 31% more thrust.  This is a contact-phase geometry, not a flight trim.
* The x-y-z joint chain has a gimbal singularity at `q1 = +/- 90 deg`; the setup
  warns below `|cos q1| = 0.5` and the run stops below 0.2.
