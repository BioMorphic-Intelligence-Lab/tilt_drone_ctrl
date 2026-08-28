"""Feed-forward rotor forces: hold an attitude *and* press with a given force.

At equilibrium the pivot carries the weight, so the only balance condition on
the 3-dof system is zero torque about the pivot.  That leaves a one-dimensional
family of rotor forces (the null space of B_tau), and the forward rotor tilt
turns that freedom into a commandable push against the wall:

    [   B_tau    ] u = [ -tau_g(q_ref) ]        3 torque equations
    [ n^T R(q) D ]     [   -f_push     ]        + the normal contact force

With four rotors that is square, but only well conditioned if the tilt angles
actually give independent authority -- hence the diagnostics here.  The solve is
bounded (0 <= u <= u_max, propellers push one way only) and weighted, so that
when the two cannot both be met the torque balance wins and the push force is
sacrificed rather than the drone falling over.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from .contact import check_cone


@dataclass
class TrimResult:
    u: np.ndarray
    tau_residual: np.ndarray   # B_tau u + tau_g, body frame [N m]
    f_push: float              # achieved normal push into the wall [N]
    f_push_cmd: float
    cond: float                # conditioning of the trim map
    saturated: bool
    feasible: bool

    def report(self) -> str:
        s = (f"trim: u = {np.round(self.u, 3)} N  "
             f"| push {self.f_push:.3f} N (cmd {self.f_push_cmd:.3f})  "
             f"| torque residual {np.linalg.norm(self.tau_residual):.2e} N m  "
             f"| cond {self.cond:.1f}")
        if not self.feasible:
            s += "\n  [WARN] trim infeasible under rotor limits: the attitude/push pair cannot " \
                 "be held exactly. Torque balance was prioritised; push force was relaxed."
        elif self.saturated:
            s += "\n  [WARN] a rotor sits on its limit at trim; little headroom left for feedback."
        return s


def trim_matrix(model, q_ref, normal=(1.0, 0.0, 0.0)):
    """The 4 x n_u map [torque about pivot; normal push] <- rotor forces."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    R = model.R(np.asarray(q_ref, float))
    return np.vstack([model.B_tau, n @ R @ model.D])


def solve_trim(model, q_ref, f_push, u_max, normal=(1.0, 0.0, 0.0),
               w_tau: float = 100.0, w_f: float = 1.0) -> TrimResult:
    q_ref = np.asarray(q_ref, float)
    A = trim_matrix(model, q_ref, normal)
    b = np.concatenate([-model.gravity_torque(q_ref), [-f_push]])
    W = np.diag([w_tau, w_tau, w_tau, w_f])
    sol = lsq_linear(W @ A, W @ b, bounds=(0.0, u_max))
    u = sol.x
    tau_res = model.B_tau @ u + model.gravity_torque(q_ref)
    achieved = -float(A[3] @ u)
    return TrimResult(
        u=u, tau_residual=tau_res, f_push=achieved, f_push_cmd=f_push,
        cond=float(np.linalg.cond(A)),
        saturated=bool(np.any(u > u_max - 1e-6) or np.any(u < 1e-9)),
        feasible=bool(np.linalg.norm(tau_res) < 1e-6 and abs(achieved - f_push) < 1e-4),
    )


def push_force_range(model, q_ref, u_max, normal=(1.0, 0.0, 0.0)):
    """Range of push force reachable while balancing torque exactly.

    Torque balance pins u to a line u(lam) = u_p + lam * v (v spans null(B_tau));
    the rotor limits clip lam to an interval, which maps to a push interval.
    Returns (f_min, f_max) or None if no bounded-feasible torque balance exists.
    """
    q_ref = np.asarray(q_ref, float)
    line = _trim_line(model, q_ref, u_max)
    if line is None:
        return None
    u_p, v, lo, hi = line
    A = trim_matrix(model, q_ref, normal)
    pushes = [-float(A[3] @ (u_p + lam * v)) for lam in (lo, hi)]
    return min(pushes), max(pushes)


def _trim_line(model, q_ref, u_max):
    """The bounded segment of rotor forces that balances torque: u(lam), lam in [lo, hi]."""
    b_tau = -model.gravity_torque(q_ref)
    u_p, *_ = np.linalg.lstsq(model.B_tau, b_tau, rcond=None)
    ns = np.linalg.svd(model.B_tau)[2][np.linalg.matrix_rank(model.B_tau):]
    if ns.shape[0] != 1:
        return None
    v = ns[0]
    lo, hi = -np.inf, np.inf
    for ui, vi in zip(u_p, v):
        if abs(vi) < 1e-12:
            if ui < -1e-9 or ui > u_max + 1e-9:
                return None
            continue
        a, c = (0.0 - ui) / vi, (u_max - ui) / vi
        lo, hi = max(lo, min(a, c)), min(hi, max(a, c))
    return (u_p, v, lo, hi) if lo <= hi else None


def cone_feasible_push_range(model, q_ref, u_max, mu, normal=(1.0, 0.0, 0.0), n=2001):
    """Push forces that can be held *without slipping*, at this reference attitude.

    Tilting the rotors into the wall couples push force to lift: pushing harder
    means thrusting harder, so the tip carries a vertical shear of (lift - weight)
    that the friction cone has to absorb.  Commanding a push outside this range
    gives a static equilibrium that slips, however well the LQR tracks attitude.

    Returns (f_min, f_max) or None if no static trim sits inside the cone.
    """
    q_ref = np.asarray(q_ref, float)
    line = _trim_line(model, q_ref, u_max)
    if line is None:
        return None
    u_p, v, lo, hi = line
    A = trim_matrix(model, q_ref, normal)
    x0 = np.concatenate([q_ref, np.zeros(3)])
    feasible = []
    for lam in np.linspace(lo, hi, n):
        u = u_p + lam * v
        f_c = model.reaction_force(x0, u, omega_dot=np.zeros(3))
        if check_cone(f_c, mu, normal).ok:
            feasible.append(-float(A[3] @ u))
    return (min(feasible), max(feasible)) if feasible else None
