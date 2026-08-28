"""Per-timestep LQR: relinearise the nonlinear model, resolve the Riccati
equation, apply the state feedback.

Uses python-control's ``lqr`` (continuous-time algebraic Riccati) on the
Jacobians of the 6-state model evaluated at the current state, so the gain
tracks the operating point instead of being frozen at the reference.
"""
from __future__ import annotations

import numpy as np
import control


class LQRController:
    def __init__(self, model, x_ref, u_ff, Q, R, u_max,
                 recompute_every: int = 1, relinearize_at_state: bool = True):
        self.model = model
        self.x_ref = np.asarray(x_ref, float)
        self.u_ff = np.asarray(u_ff, float)
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.u_max = u_max
        self.recompute_every = max(1, int(recompute_every))
        self.relinearize_at_state = relinearize_at_state
        self.K = None
        self._k = 0
        self.n_riccati_failures = 0
        self.n_saturated_steps = 0

    def controllability_rank(self, x=None) -> int:
        A, B = self.model.linearize(self.x_ref if x is None else x, self.u_ff)
        return int(np.linalg.matrix_rank(control.ctrb(A, B), tol=1e-9))

    def _gain(self, x):
        A, B = self.model.linearize(x, self.u_ff)
        try:
            K, _, _ = control.lqr(A, B, self.Q, self.R)
            return np.asarray(K, float)
        except Exception as exc:                      # non-stabilisable / near-singular
            self.n_riccati_failures += 1
            if self.K is None:
                raise RuntimeError(
                    f"LQR failed at the initial linearisation ({exc}). The system is not "
                    "stabilisable there -- check rotor tilt angles and gimbal margin."
                ) from exc
            print(f"  [WARN] LQR solve failed ({exc}); holding the previous gain.")
            return self.K

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        if self.K is None or self._k % self.recompute_every == 0:
            self.K = self._gain(x if self.relinearize_at_state else self.x_ref)
        self._k += 1
        u = self.u_ff - self.K @ (x - self.x_ref)
        u_clipped = np.clip(u, 0.0, self.u_max)
        if not np.allclose(u, u_clipped, atol=1e-9):
            self.n_saturated_steps += 1
        return u_clipped

    def summary(self) -> str:
        parts = [f"LQR: {self.n_saturated_steps} saturated steps"]
        if self.n_riccati_failures:
            parts.append(f"{self.n_riccati_failures} Riccati failures")
        return ", ".join(parts)
