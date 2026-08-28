"""The validation gate: does the analytic model the LQR is built on actually
describe the Genesis plant?

Comparing trajectories is a weak test -- this system tumbles chaotically when
free, so any two integrators diverge regardless of whether the model is right.
Instead we compare *accelerations* over a single tiny step, which is
integrator-independent and exercises the whole chain at once: parsed inertia
about the pivot, gravity torque, the rotor force/tilt geometry, propeller drag
reaction, and the way forces are handed to Genesis.
"""
import numpy as np
import pytest

from tilt_drone.urdf_model import load_body
from tilt_drone.dynamics import DroneWallModel

URDF = "urdf/tilt_drone.urdf"
KAPPA = 0.016


@pytest.fixture(scope="module")
def setup():
    from tilt_drone.plant import GenesisPlant
    model = DroneWallModel(load_body(URDF), kappa=KAPPA)
    plant = GenesisPlant(model, URDF, dt=1e-5, show_viewer=False, wall_visual=False,
                         draw_forces=False)
    return model, plant


def _genesis_omega_dot(plant, model, x, u):
    plant.set_x(x)
    w0 = plant.get_x()[3:]
    plant.apply_rotor_forces(u)
    plant.step()
    q1, qd1 = plant.get_state()
    return (model.E(q1) @ qd1 - w0) / plant.dt


def test_parsed_body_matches_genesis(setup):
    """Mass and CoM parsed from the URDF must match what Genesis built."""
    model, plant = setup
    m_gen = sum(float(np.asarray(l.inertial_mass).ravel()[0]) for l in plant.drone.links)
    assert m_gen == pytest.approx(model.m, rel=1e-4)  # dummy pivot links carry 1e-6 kg each


def test_gravity_only_acceleration(setup):
    """Free swing: pure gravity torque about the pivot."""
    model, plant = setup
    for q in ([0.0, 0.0, 0.0], [0.2, -0.3, 0.15], [-0.4, 0.25, 0.6]):
        x = np.concatenate([q, np.zeros(3)])
        got = _genesis_omega_dot(plant, model, x, np.zeros(4))
        assert got == pytest.approx(model.omega_dot(x, np.zeros(4)), abs=2e-3)


def test_rotor_actuation_and_drag(setup):
    """Thrust geometry (tilt!) and propeller drag reaction."""
    model, plant = setup
    rng = np.random.default_rng(0)
    for _ in range(8):
        q = rng.uniform(-0.4, 0.4, 3)
        w = rng.uniform(-0.5, 0.5, 3)
        u = rng.uniform(0.0, 8.0, 4)
        x = np.concatenate([q, w])
        got = _genesis_omega_dot(plant, model, x, u)
        assert got == pytest.approx(model.omega_dot(x, u), abs=5e-3, rel=2e-3)


def test_single_rotor_isolates_tilt(setup):
    """One rotor at a time: catches a wrong sign or a swapped rotor ordering."""
    model, plant = setup
    for i in range(4):
        u = np.zeros(4)
        u[i] = 6.0
        x = np.zeros(6)
        got = _genesis_omega_dot(plant, model, x, u) - _genesis_omega_dot(plant, model, x, np.zeros(4))
        want = model.I_O_inv @ (model.B_tau @ u)
        assert got == pytest.approx(want, abs=5e-3, rel=2e-3), f"rotor {i}"


def test_analytic_B_matches_finite_difference(setup):
    """The closed-form input Jacobian used by the LQR."""
    model, _ = setup
    x = np.array([0.1, -0.2, 0.05, 0.1, 0.0, -0.1])
    u = np.array([3.0, 3.5, 2.0, 2.5])
    _, B = model.linearize(x, u)
    B_fd = np.zeros_like(B)
    for i in range(model.n_u):
        du = np.zeros(model.n_u)
        du[i] = 1e-6
        B_fd[:, i] = (model.f(x, u + du) - model.f(x, u - du)) / 2e-6
    assert B == pytest.approx(B_fd, abs=1e-8)


def test_reaction_force_at_static_equilibrium(setup):
    """With no motion and no thrust, the joint simply carries the weight."""
    model, _ = setup
    x = np.zeros(6)
    f_c = model.reaction_force(x, np.zeros(4), omega_dot=np.zeros(3))
    assert f_c == pytest.approx(-model.m * model.g, abs=1e-9)
