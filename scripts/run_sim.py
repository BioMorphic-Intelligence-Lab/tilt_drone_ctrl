#!/usr/bin/env python3
"""Simulate the wall-pinned tilted-rotor drone under per-timestep LQR.

    python scripts/run_sim.py --vis --plot
    python scripts/run_sim.py --plot --mu 0.12            # trip the friction cone
    python scripts/run_sim.py --traj trajectories/three_steps.traj.txt --plot
    python scripts/run_sim.py --plot --disturb-t 3 --disturb-torque 0 0 2.5
    python scripts/run_sim.py --vis --record run.mp4

Genesis is the plant: it integrates the URDF's articulated dynamics and we only
hand it rotor forces.  The analytic model in tilt_drone/dynamics.py is used to
linearise for LQR and to evaluate the contact force; tests/ checks the two agree.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tilt_drone.urdf_model import load_body, describe
from tilt_drone.dynamics import DroneWallModel
from tilt_drone.trim import solve_trim, push_force_range, cone_feasible_push_range
from tilt_drone.lqr_ctrl import LQRController
from tilt_drone.contact import ConeMonitor
from tilt_drone.plotting import SimLog, plot_run
from tilt_drone.trajectory import load_trajectory, describe as describe_traj

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--urdf", default=os.path.join(HERE, "urdf", "tilt_drone.urdf"))
    p.add_argument("--dt", type=float, default=0.002, help="simulation step [s]")
    p.add_argument("--T", type=float, default=None,
                   help="duration [s]; defaults to 6 s, or 3 s past the last "
                        "trajectory waypoint when --traj is given")
    p.add_argument("--ctrl-hz", type=float, default=250.0, help="controller rate [Hz]")
    p.add_argument("--kappa", type=float, default=0.016, help="propeller drag/thrust ratio [m]")
    p.add_argument("--u-max", type=float, default=10.0, help="per-rotor thrust limit [N]")
    p.add_argument("--mu", type=float, default=0.4, help="friction coefficient at the beam tip")
    p.add_argument("--f-push", default="auto",
                   help="commanded push into the wall [N], or 'auto' for the middle of the "
                        "range this attitude can hold without slipping")
    p.add_argument("--q-ref", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   help="reference joint angles [deg]")
    p.add_argument("--q0", type=float, nargs=3, default=[6.0, 4.0, -5.0],
                   help="initial joint angles [deg]")
    p.add_argument("--traj", default=None,
                   help="trajectory file of step references: one 't phi_x theta_y psi_z' "
                        "line per step, angles in degrees (see trajectories/)")
    p.add_argument("--step-t", type=float, default=None,
                   help="time of a single reference step [s]; --traj supersedes it")
    p.add_argument("--q-ref2", type=float, nargs=3, default=[0.0, 5.0, 8.0],
                   help="reference joint angles after --step-t [deg]")
    p.add_argument("--disturb-t", type=float, default=None, help="disturbance onset [s]")
    p.add_argument("--disturb-dur", type=float, default=0.05, help="disturbance duration [s]")
    p.add_argument("--disturb-torque", type=float, nargs=3, default=[0.0, 0.0, 2.0],
                   help="world-frame disturbance torque on the drone [N m]")
    p.add_argument("--Q", type=float, nargs=6, default=[40, 40, 40, 4, 4, 4],
                   help="LQR state weights (3 angles, 3 rates)")
    p.add_argument("--R", type=float, default=0.2, help="LQR input weight (per rotor)")
    p.add_argument("--relin-every", type=int, default=1,
                   help="re-solve the Riccati equation every N controller steps")
    p.add_argument("--freeze-gain", action="store_true",
                   help="linearise at the reference instead of the current state")
    p.add_argument("--vis", action="store_true", help="open the Genesis viewer")
    p.add_argument("--record", default=None, help="write an mp4 of the 3D view to this path")
    p.add_argument("--plot", action="store_true", help="show the diagnostic plots")
    p.add_argument("--no-save-plot", action="store_true")
    p.add_argument("--out", default=os.path.join(HERE, "logs"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def setup_reference(model, q_ref, a, indent=""):
    """Report what this reference attitude can hold, then solve the trim for it.

    Both limits are attitude-dependent, and strongly so: pitching the assembly up
    by 3 deg cuts the rotors' effective tilt into the wall from 10 to 7 deg, and
    with total thrust nearly pinned by the torque balance the achievable push
    drops by about the same 30%.  A push command that is fine at one attitude is
    therefore unreachable at another -- hence 'auto'.
    """
    def say(msg):
        print(indent + msg)

    rng = push_force_range(model, q_ref, a.u_max)
    cone_rng = cone_feasible_push_range(model, q_ref, a.u_max, a.mu)
    auto = str(a.f_push).lower() == "auto"

    if rng is None:
        say("  [WARN] no bounded rotor forces balance gravity at this reference attitude.")
    else:
        say(f"push reachable while balancing torque: [{rng[0]:.3f}, {rng[1]:.3f}] N")
    if cone_rng is None:
        say(f"  [WARN] at mu = {a.mu} no static trim sits inside the friction cone: every "
            "attitude-holding thrust slips the tip. Raise mu, or retune the rotor tilt "
            "so lift and push decouple better.")
    else:
        say(f"...of which the friction cone (mu = {a.mu}) admits: "
            f"[{cone_rng[0]:.3f}, {cone_rng[1]:.3f}] N")

    if auto:
        src = cone_rng or rng
        f_push = float(np.mean(src)) if src else 0.0
        say(f"push command (auto): {f_push:.3f} N"
            + ("  [middle of the slip-free range]" if cone_rng else
               "  [middle of the torque-feasible range; no slip-free trim exists]"))
    else:
        f_push = float(a.f_push)
        say(f"push command: {f_push:.3f} N")
        if rng is not None and not rng[0] - 1e-9 <= f_push <= rng[1] + 1e-9:
            say("  [WARN] outside the torque-feasible range; the trim solve will prioritise "
                "torque balance and the achieved push will differ. Widen it with more "
                "forward rotor tilt or a higher --u-max.")
        elif cone_rng is not None and not cone_rng[0] - 1e-9 <= f_push <= cone_rng[1] + 1e-9:
            say("  [WARN] this push slips even in steady state: pressing this hard needs a "
                "thrust whose lift differs from the weight by more than friction can hold. "
                "Expect persistent cone warnings.")

    trim = solve_trim(model, q_ref, f_push, a.u_max)
    for line in trim.report().splitlines():
        say(line)
    return trim, f_push


def main(argv=None):
    a = parse_args(argv)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    body = load_body(a.urdf)
    model = DroneWallModel(body, kappa=a.kappa)
    print(describe(body, a.kappa))

    waypoints = None
    if a.traj is not None:
        waypoints = load_trajectory(a.traj)
        print(describe_traj(waypoints, a.traj))
        if a.step_t is not None:
            print("  [WARN] --step-t is ignored while --traj is given.")
        if a.T is None:
            a.T = waypoints[-1].t + 3.0
        elif waypoints[-1].t >= a.T:
            print(f"  [WARN] the last waypoint starts at t = {waypoints[-1].t:.2f} s, at or "
                  f"after the end of the run (T = {a.T:.2f} s); it will never be applied.")
    if a.T is None:
        a.T = 6.0

    # a waypoint at t <= 0 sets the starting reference; otherwise --q-ref does
    next_wp = 0
    if waypoints is not None and waypoints[0].t <= 0.0:
        a.q_ref = list(waypoints[0].q_deg)
        next_wp = 1

    q_ref = np.deg2rad(a.q_ref)
    x_ref = np.concatenate([q_ref, np.zeros(3)])

    trim, f_push = setup_reference(model, q_ref, a)
    if model.gimbal_margin(q_ref) < 0.5:
        print(f"  [WARN] reference attitude is near gimbal lock "
              f"(|cos q_y| = {model.gimbal_margin(q_ref):.3f}); the x-y-z joint "
              "parametrisation is ill-conditioned there.")

    from tilt_drone.plant import GenesisPlant     # imported late: gs.init is global
    plant = GenesisPlant(model, a.urdf, dt=a.dt, show_viewer=a.vis,
                         record=a.record is not None)

    ctrl = LQRController(model, x_ref, trim.u, np.diag(a.Q), a.R * np.eye(model.n_u),
                         a.u_max, recompute_every=a.relin_every,
                         relinearize_at_state=not a.freeze_gain)
    rank = ctrl.controllability_rank()
    print(f"controllability rank at the reference: {rank}/6"
          + ("" if rank == 6 else "   [WARN] not controllable -- check the rotor tilt angles"))

    monitor = ConeMonitor(a.mu)
    log = SimLog()
    n_steps = int(round(a.T / a.dt))
    ctrl_decim = max(1, int(round(1.0 / (a.ctrl_hz * a.dt))))
    beam_idx = [plant.drone.get_link("beam").idx]

    plant.set_x(np.concatenate([np.deg2rad(a.q0), np.zeros(3)]))
    video_path = None
    if a.record:
        video_path = a.record if os.path.isabs(a.record) else os.path.join(a.out, a.record)
        plant.start_recording(video_path, fps=min(60, int(round(1.0 / a.dt))))

    print(f"\nsimulating {a.T:.1f} s at dt = {a.dt * 1e3:.1f} ms, "
          f"control @ {1.0 / (a.dt * ctrl_decim):.0f} Hz")
    u = trim.u.copy()
    for k in range(n_steps):
        t = k * a.dt
        x = plant.get_x()

        if waypoints is not None:
            while next_wp < len(waypoints) and t >= waypoints[next_wp].t:
                wp = waypoints[next_wp]
                next_wp += 1
                q_ref = wp.q
                print(f"  t={t:5.2f}s  step {next_wp}/{len(waypoints)} -> "
                      f"q_ref = {np.round(wp.q_deg, 2)} deg")
                trim_wp, f_push = setup_reference(model, q_ref, a, indent="    ")
                ctrl.x_ref = np.concatenate([q_ref, np.zeros(3)])
                ctrl.u_ff = trim_wp.u
        elif a.step_t is not None and t >= a.step_t and not np.allclose(ctrl.x_ref[:3], np.deg2rad(a.q_ref2)):
            q_ref = np.deg2rad(a.q_ref2)
            print(f"  t={t:5.2f}s  reference step to {np.round(a.q_ref2, 2)} deg")
            trim2, f_push = setup_reference(model, q_ref, a, indent="    ")
            ctrl.x_ref = np.concatenate([q_ref, np.zeros(3)])
            ctrl.u_ff = trim2.u

        if k % ctrl_decim == 0:
            u = ctrl(x)

        omega_dot = model.omega_dot(x, u)
        f_c = model.reaction_force(x, u, omega_dot)
        status = monitor.update(t, f_c)
        log.add(t, x[:3], x[3:], ctrl.x_ref[:3], u, f_c, status)

        plant.apply_rotor_forces(u)
        if a.disturb_t is not None and a.disturb_t <= t < a.disturb_t + a.disturb_dur:
            plant._solver.apply_links_external_torque(
                np.array([a.disturb_torque], float), beam_idx, ref="link_origin", local=False)
            if abs(t - a.disturb_t) < a.dt:
                print(f"  t={t:5.2f}s  disturbance torque {a.disturb_torque} N m for "
                      f"{a.disturb_dur * 1e3:.0f} ms")
        plant.draw(u, f_c, status.ok)
        plant.step()

        if model.gimbal_margin(x[:3]) < 0.2:
            print(f"  [WARN] t={t:.3f}s approaching gimbal lock; stopping.")
            break

    d = log.arrays()
    err = np.rad2deg(np.linalg.norm(d["q"] - d["q_ref"], axis=1))
    print(f"\n{ctrl.summary()}")
    print(monitor.summary())
    print(f"final attitude error: {err[-1]:.3f} deg   (peak {err.max():.3f} deg)")
    print(f"push force: mean {np.mean(d['f_n']):.3f} N, min {np.min(d['f_n']):.3f} N, "
          f"max {np.max(d['f_n']):.3f} N")
    print(f"cone margin: min {np.min(d['margin']):+.3f} N")

    if a.record:
        plant.save_video()
    np.savez(os.path.join(a.out, "run.npz"), **d)
    push_desc = ("auto (per reference)" if str(a.f_push).lower() == "auto"
                 else f"{float(a.f_push):.2f} N")
    plot_path = None if a.no_save_plot else os.path.join(a.out, "run.png")
    plot_run(log, [r.name for r in body.rotors], a.mu, a.u_max, path=plot_path, show=a.plot,
             title=f"Wall-pinned tilted-rotor drone  |  mu = {a.mu}, push cmd = {push_desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
