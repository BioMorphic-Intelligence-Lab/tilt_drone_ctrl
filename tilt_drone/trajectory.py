"""Step-reference trajectories read from a text file.

A trajectory file lists step inputs for the spherical joint: when each step
starts, and the reference angles it holds until the next one.  One waypoint per
line, blank lines and ``#`` comments ignored:

    # t[s]   phi_x   theta_y   psi_z        (degrees)
    0.0      0.0     0.0       0.0
    2.5      0.0     3.0       5.0

The reference is a zero-order hold: each waypoint's angles apply from its start
time until the next waypoint's, so the controller sees genuine step inputs
rather than an interpolated path.  Whether a given step is actually holdable is a
separate question -- the push force a reference attitude can sustain varies a lot
with pitch -- so the driver re-solves the trim and re-reports the feasible ranges
at every waypoint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Waypoint:
    t: float                 # start time of this step [s]
    q_deg: np.ndarray        # reference joint angles [deg]
    line_no: int = 0

    @property
    def q(self) -> np.ndarray:
        return np.deg2rad(self.q_deg)


def load_trajectory(path: str) -> list[Waypoint]:
    """Parse a trajectory file into an ordered list of step waypoints."""
    waypoints: list[Waypoint] = []
    with open(path) as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.replace(",", " ").split()
            if len(fields) != 4:
                raise ValueError(
                    f"{path}:{line_no}: expected 4 values "
                    f"'t phi_x theta_y psi_z', found {len(fields)}: {raw.strip()!r}")
            try:
                values = [float(v) for v in fields]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from None
            waypoints.append(Waypoint(t=values[0], q_deg=np.array(values[1:]), line_no=line_no))

    if not waypoints:
        raise ValueError(f"{path}: no waypoints found")
    for prev, nxt in zip(waypoints, waypoints[1:]):
        if nxt.t <= prev.t:
            raise ValueError(
                f"{path}:{nxt.line_no}: start times must strictly increase "
                f"(t = {nxt.t} follows t = {prev.t} on line {prev.line_no})")
    if waypoints[0].t < 0.0:
        raise ValueError(f"{path}:{waypoints[0].line_no}: start times must be non-negative")
    return waypoints


def describe(waypoints: list[Waypoint], path: str) -> str:
    lines = [f"trajectory: {len(waypoints)} step reference(s) from {path}"]
    for w in waypoints:
        lines.append(f"  t = {w.t:6.2f} s -> q_ref = "
                     f"[{w.q_deg[0]:6.2f} {w.q_deg[1]:6.2f} {w.q_deg[2]:6.2f}] deg")
    return "\n".join(lines)
