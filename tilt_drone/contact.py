"""Coulomb friction-cone monitoring at the spherical joint.

The spherical-joint idealisation is only valid while the beam tip does not slip,
i.e. while the contact force stays inside the friction cone.  This module does
not change the dynamics; it watches them and complains.

Sign convention: the wall occupies x < 0 with outward normal n = +x.  ``f_c`` is
the force the joint applies *to the drone*, so the drone presses on the wall
with -f_c and

    f_n = f_c . n        must be > 0   (contact in compression, not tension)
    ||f_c - f_n n||      must be <= mu * f_n
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConeStatus:
    f_c: np.ndarray      # joint force on the drone, world frame [N]
    f_n: float           # normal component (>0 = pressing into the wall) [N]
    f_t: float           # tangential magnitude [N]
    margin: float        # mu*f_n - |f_t|; negative means slipping [N]
    ok: bool
    reason: str = ""


def check_cone(f_c: np.ndarray, mu: float, normal=(1.0, 0.0, 0.0)) -> ConeStatus:
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    f_n = float(f_c @ n)
    f_t = float(np.linalg.norm(f_c - f_n * n))
    margin = mu * f_n - f_t
    if f_n <= 0.0:
        return ConeStatus(f_c, f_n, f_t, margin, False, "contact lost (tension at the tip)")
    if margin < 0.0:
        return ConeStatus(f_c, f_n, f_t, margin, False, "outside friction cone (tip would slip)")
    return ConeStatus(f_c, f_n, f_t, margin, True)


class ConeMonitor:
    """Rate-limited warnings, so a long violation does not flood the console."""

    def __init__(self, mu: float, normal=(1.0, 0.0, 0.0), warn_every: float = 0.25):
        self.mu, self.normal, self.warn_every = mu, normal, warn_every
        self._last_warn = -np.inf
        self.n_violations = 0
        self.first_violation_t = None

    def update(self, t: float, f_c: np.ndarray) -> ConeStatus:
        st = check_cone(f_c, self.mu, self.normal)
        if not st.ok:
            self.n_violations += 1
            if self.first_violation_t is None:
                self.first_violation_t = t
            if t - self._last_warn >= self.warn_every:
                self._last_warn = t
                print(f"  [WARN] t={t:6.3f}s  {st.reason}: "
                      f"f_n={st.f_n:7.3f} N, |f_t|={st.f_t:6.3f} N, "
                      f"mu*f_n={self.mu * st.f_n:6.3f} N (margin {st.margin:+.3f} N)")
        return st

    def summary(self) -> str:
        if self.n_violations == 0:
            return "friction cone: never violated"
        return (f"friction cone: VIOLATED on {self.n_violations} steps, "
                f"first at t = {self.first_violation_t:.3f} s")
