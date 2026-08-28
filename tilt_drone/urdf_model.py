"""Parse the URDF into the rigid-body quantities the analytic model needs.

The URDF is the single source of truth: Genesis loads it as the plant and this
module derives the same body from the same file, so editing a rotor's ``rpy``
moves the plant and the controller model together.

The robot is expected to look like

    wall_anchor -(continuous, x)- pivot_x -(continuous, y)- pivot_y
                -(continuous, z)- beam -(fixed)- body -(fixed)- rotor_*

i.e. exactly three consecutive revolute dofs standing in for the spherical
joint, and a rigid subtree below them.  The frame after the third revolute
joint is the *body frame*; its origin is the pivot, which is what makes Euler's
equation about a fixed point the natural way to write the dynamics.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

BALL_AXES = (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))


def rpy_to_R(rpy) -> np.ndarray:
    """URDF fixed-axis convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def skew(v) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


@dataclass
class Rotor:
    name: str
    pos: np.ndarray      # rotor origin in the body (pivot) frame [m]
    axis: np.ndarray     # unit thrust direction in the body frame
    spin: int            # +1 = CCW about its own +z, -1 = CW
    tilt_fwd_deg: float  # tilt of the thrust axis towards the wall (-x)
    tilt_side_deg: float # tilt of the thrust axis towards +y


@dataclass
class BodyModel:
    """Rigid body formed by everything below the spherical joint."""
    mass: float
    r_cm: np.ndarray          # CoM in the body frame (origin = pivot)
    I_O: np.ndarray           # inertia about the pivot, body frame
    rotors: list[Rotor] = field(default_factory=list)
    joint_names: tuple = ()
    urdf_path: str = ""

    @property
    def n_rotors(self) -> int:
        return len(self.rotors)

    @property
    def D(self) -> np.ndarray:
        """(3, n) matrix of thrust directions in the body frame."""
        return np.column_stack([r.axis for r in self.rotors])

    def B_tau(self, kappa: float) -> np.ndarray:
        """(3, n) map from rotor forces to torque about the pivot, body frame.

        Column i is  p_i x d_i  (thrust lever arm)  -  spin_i * kappa * d_i
        (propeller drag reaction).  Constant, because the tilt is fixed.
        """
        return np.column_stack([
            np.cross(r.pos, r.axis) - r.spin * kappa * r.axis for r in self.rotors
        ])


def _origin(elem):
    o = elem.find("origin")
    if o is None:
        return np.zeros(3), np.eye(3)
    xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
    rpy = np.array([float(v) for v in o.get("rpy", "0 0 0").split()])
    return xyz, rpy_to_R(rpy)


def _link_inertial(link):
    """(mass, CoM offset in link frame, inertia about the CoM in link frame)."""
    ine = link.find("inertial")
    if ine is None:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    m = float(ine.find("mass").get("value"))
    c, R_c = _origin(ine)
    t = ine.find("inertia")
    ixx, iyy, izz = float(t.get("ixx")), float(t.get("iyy")), float(t.get("izz"))
    ixy, ixz, iyz = float(t.get("ixy", 0)), float(t.get("ixz", 0)), float(t.get("iyz", 0))
    I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    return m, c, R_c @ I @ R_c.T


def load_body(urdf_path: str, upstream_mass_tol: float = 1e-3) -> BodyModel:
    root = ET.parse(urdf_path).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = list(root.findall("joint"))
    children = {}
    for j in joints:
        children.setdefault(j.find("parent").get("link"), []).append(j)

    # -- locate the three revolute dofs standing in for the spherical joint ---
    parents = {j.find("child").get("link") for j in joints}
    base = next(n for n in links if n not in parents)
    chain, link_name = [], base
    while True:
        js = children.get(link_name, [])
        movable = [j for j in js if j.get("type") in ("continuous", "revolute")]
        if not movable:
            break
        if len(movable) > 1:
            raise ValueError(f"link '{link_name}' branches into several revolute joints")
        j = movable[0]
        chain.append(j)
        link_name = j.find("child").get("link")
    if len(chain) != 3:
        raise ValueError(f"expected 3 revolute joints for the spherical joint, found {len(chain)}")
    for i, j in enumerate(chain):
        xyz, R = _origin(j)
        if np.linalg.norm(xyz) > 1e-9 or not np.allclose(R, np.eye(3), atol=1e-9):
            raise ValueError(f"joint '{j.get('name')}' must be coincident with the pivot "
                             "(zero origin) for the spherical-joint model to hold")
        ax = np.array([float(v) for v in j.find("axis").get("xyz").split()])
        if not np.allclose(ax / np.linalg.norm(ax), BALL_AXES[i], atol=1e-9):
            raise ValueError(f"joint '{j.get('name')}' axis must be {BALL_AXES[i]}; "
                             "the model assumes an x-y-z chain")

    # links above the pivot must be negligible: they are not part of the rigid body
    for j in [None] + chain[:-1]:
        n = base if j is None else j.find("child").get("link")
        m, _, _ = _link_inertial(links[n])
        if m > upstream_mass_tol:
            raise ValueError(f"link '{n}' sits above the spherical joint and carries "
                             f"{m} kg; it must be (near-)massless")

    body_root = chain[-1].find("child").get("link")

    # ------- walk the rigid (fixed-joint) subtree, accumulating transforms ---
    mass, m_c, I_O = 0.0, np.zeros(3), np.zeros((3, 3))
    rotors: list[Rotor] = []
    stack = [(body_root, np.zeros(3), np.eye(3))]
    while stack:
        name, p, R = stack.pop()
        m, c_loc, I_c = _link_inertial(links[name])
        if m > 0:
            c = p + R @ c_loc                       # CoM in the body frame
            mass += m
            m_c += m * c
            # rotate to the body frame, then shift to the pivot (parallel axis)
            I_O += R @ I_c @ R.T + m * ((c @ c) * np.eye(3) - np.outer(c, c))
        if name.startswith("rotor"):
            axis = R @ np.array([0.0, 0.0, 1.0])
            rotors.append(Rotor(
                name=name, pos=p, axis=axis,
                spin=+1 if name.endswith("ccw") else -1,
                tilt_fwd_deg=np.degrees(np.arcsin(np.clip(-axis[0], -1, 1))),
                tilt_side_deg=np.degrees(np.arcsin(np.clip(axis[1], -1, 1))),
            ))
        for j in children.get(name, []):
            if j.get("type") != "fixed":
                raise ValueError(f"joint '{j.get('name')}' below the pivot is not fixed; "
                                 "the beam+drone must be a single rigid body")
            xyz, R_j = _origin(j)
            stack.append((j.find("child").get("link"), p + R @ xyz, R @ R_j))

    rotors.sort(key=lambda r: r.name)
    return BodyModel(mass=mass, r_cm=m_c / mass, I_O=I_O, rotors=rotors,
                     joint_names=tuple(j.get("name") for j in chain),
                     urdf_path=urdf_path)


def describe(body: BodyModel, kappa: float) -> str:
    B = body.B_tau(kappa)
    lines = [f"body: m = {body.mass:.4f} kg, r_cm = {np.round(body.r_cm, 4)} m",
             f"I_O (about pivot) =\n{np.array2string(body.I_O, precision=5)}",
             f"rotors ({body.n_rotors}):"]
    for r in body.rotors:
        lines.append(f"  {r.name:<14s} p = {np.round(r.pos, 3)}  d = {np.round(r.axis, 4)}  "
                     f"spin {'CCW' if r.spin > 0 else 'CW ':>3s}  "
                     f"tilt: {r.tilt_fwd_deg:5.2f} deg into wall, {r.tilt_side_deg:+5.2f} deg sideways")
    lines.append(f"B_tau =\n{np.array2string(B, precision=4)}")
    lines.append(f"rank(B_tau) = {np.linalg.matrix_rank(B)}")
    return "\n".join(lines)
