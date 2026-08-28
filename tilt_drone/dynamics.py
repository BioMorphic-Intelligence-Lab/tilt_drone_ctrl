"""6-state nonlinear model of the wall-pinned drone, and its linearisation.

With the beam tip pinned by a spherical joint, the beam+drone is one rigid body
rotating about a fixed point, so the dynamics are Euler's equation about that
point.  State is

    x = [q (3 joint angles), omega (3, body angular velocity)]

    q_dot     = T(q) omega
    I_O w_dot = -omega x (I_O omega) + tau_g(q) + B_tau u

The joint angles are the URDF's three revolute dofs in order (x, y, z), so
``x[:3]`` is exactly what Genesis reports for those dofs -- no conversion, no
estimator.  The rotation from body to world is R(q) = Rx(q0) Ry(q1) Rz(q2).

The state-dependent actuation onto the joint is  tau_joint(q, u) = R(q) B_tau u:
B_tau itself is constant in the body frame because the rotor tilt is fixed, and
all of the configuration dependence is the rotation of that wrench into the
world plus the gravity term.
"""
from __future__ import annotations

import numpy as np

from .urdf_model import BodyModel, skew

G_WORLD = np.array([0.0, 0.0, -9.81])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class DroneWallModel:
    """Analytic model derived from the URDF; used for LQR and for validation."""

    def __init__(self, body: BodyModel, kappa: float = 0.016, gravity=G_WORLD):
        self.body = body
        self.kappa = kappa
        self.g = np.asarray(gravity, float)
        self.I_O = body.I_O
        self.I_O_inv = np.linalg.inv(body.I_O)
        self.m = body.mass
        self.r_cm = body.r_cm
        self.B_tau = body.B_tau(kappa)   # (3, n) rotor force -> body torque about pivot
        self.D = body.D                  # (3, n) rotor force -> body force
        self.n_u = body.n_rotors
        self.n_x = 6

    # ------------------------------------------------------------ kinematics
    @staticmethod
    def R(q) -> np.ndarray:
        """Body -> world rotation for the x-y-z revolute chain."""
        return Rx(q[0]) @ Ry(q[1]) @ Rz(q[2])

    @staticmethod
    def E(q) -> np.ndarray:
        """omega_body = E(q) q_dot."""
        R2t, R3t = Ry(q[1]).T, Rz(q[2]).T
        return np.column_stack([R3t @ R2t @ np.array([1.0, 0, 0]),
                                R3t @ np.array([0, 1.0, 0]),
                                np.array([0, 0, 1.0])])

    @staticmethod
    def T(q) -> np.ndarray:
        """q_dot = T(q) omega_body.  Singular at q[1] = +/- pi/2 (gimbal lock)."""
        return np.linalg.inv(DroneWallModel.E(q))

    @staticmethod
    def gimbal_margin(q) -> float:
        """|det E(q)| = |cos(q1)|; 1 is nominal, 0 is gimbal lock."""
        return abs(np.cos(q[1]))

    # -------------------------------------------------------------- dynamics
    def gravity_torque(self, q) -> np.ndarray:
        """Torque about the pivot from weight, body frame."""
        return np.cross(self.r_cm, self.m * (self.R(q).T @ self.g))

    def actuation_torque(self, u) -> np.ndarray:
        """Rotor forces -> torque about the pivot, body frame."""
        return self.B_tau @ np.asarray(u, float)

    def joint_wrench_world(self, q, u):
        """The state-dependent actuation function, in world coordinates.

        Returns (force, torque) that the rotors exert on the spherical joint's
        rigid body, expressed in the world frame.
        """
        R = self.R(q)
        return R @ (self.D @ np.asarray(u, float)), R @ self.actuation_torque(u)

    def omega_dot(self, x, u) -> np.ndarray:
        q, w = x[:3], x[3:]
        tau = -np.cross(w, self.I_O @ w) + self.gravity_torque(q) + self.actuation_torque(u)
        return self.I_O_inv @ tau

    def f(self, x, u) -> np.ndarray:
        x = np.asarray(x, float)
        return np.concatenate([self.T(x[:3]) @ x[3:], self.omega_dot(x, u)])

    # --------------------------------------------------------- linearisation
    def linearize(self, x, u, eps: float = 1e-6):
        """(A, B) of dx/dt = f(x, u) about (x, u).

        A by central differences; B in closed form, since f is affine in u:
        d(omega_dot)/du = I_O^-1 B_tau and the kinematic rows do not see u.
        """
        x = np.asarray(x, float)
        A = np.zeros((self.n_x, self.n_x))
        for i in range(self.n_x):
            dx = np.zeros(self.n_x)
            dx[i] = eps
            A[:, i] = (self.f(x + dx, u) - self.f(x - dx, u)) / (2 * eps)
        B = np.vstack([np.zeros((3, self.n_u)), self.I_O_inv @ self.B_tau])
        return A, B

    # ------------------------------------------------------- contact reaction
    def reaction_force(self, x, u, omega_dot=None) -> np.ndarray:
        """Force the spherical joint applies to the drone, in world coordinates.

        Newton for the whole body about the fixed pivot:
            m a_cm = f_c + m g + R D u,
            a_cm   = R (w_dot x r_cm + w x (w x r_cm))
        The force the drone applies to the wall is -f_c.
        """
        q, w = np.asarray(x[:3], float), np.asarray(x[3:], float)
        if omega_dot is None:
            omega_dot = self.omega_dot(x, u)
        R = self.R(q)
        a_cm = R @ (np.cross(omega_dot, self.r_cm) + np.cross(w, np.cross(w, self.r_cm)))
        return self.m * a_cm - self.m * self.g - R @ (self.D @ np.asarray(u, float))

    def tip_state(self, x):
        """Body-frame axes of the drone in world coordinates (for plotting/debug)."""
        R = self.R(x[:3])
        return R @ np.array([1.0, 0, 0]), R @ np.array([0, 0, 1.0])
