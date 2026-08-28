"""Trim, friction cone, LQR, and a closed-loop run against the Genesis plant."""
import numpy as np
import pytest

from tilt_drone.urdf_model import load_body
from tilt_drone.dynamics import DroneWallModel
from tilt_drone.trim import solve_trim, push_force_range, cone_feasible_push_range
from tilt_drone.lqr_ctrl import LQRController
from tilt_drone.contact import check_cone

URDF = "urdf/tilt_drone.urdf"
U_MAX = 10.0


@pytest.fixture(scope="module")
def model():
    return DroneWallModel(load_body(URDF), kappa=0.016)


# ------------------------------------------------------------------ kinematics
def test_E_and_T_are_inverses(model):
    for q in (np.zeros(3), np.array([0.3, -0.4, 0.7])):
        assert model.T(q) @ model.E(q) == pytest.approx(np.eye(3), abs=1e-12)


def test_gimbal_margin_flags_the_singularity(model):
    assert model.gimbal_margin(np.zeros(3)) == pytest.approx(1.0)
    assert model.gimbal_margin([0.0, np.pi / 2, 0.0]) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------- the tilt
def test_tilt_matches_the_urdf_intent(model):
    """All four rotors lean into the wall; only the hind pair leans sideways."""
    rotors = {r.name: r for r in model.body.rotors}
    for r in rotors.values():
        assert r.tilt_fwd_deg > 5.0, f"{r.name} is not tilted into the wall"
        assert r.axis[0] < 0.0
    assert rotors["rotor_fl_ccw"].tilt_side_deg == pytest.approx(0.0, abs=1e-9)
    assert rotors["rotor_fr_cw"].tilt_side_deg == pytest.approx(0.0, abs=1e-9)
    # hind pair tilts outboard, each towards its own side -> a couple, no net side force
    assert rotors["rotor_hl_cw"].tilt_side_deg > 1.0
    assert rotors["rotor_hr_ccw"].tilt_side_deg < -1.0
    assert rotors["rotor_hl_cw"].tilt_side_deg == pytest.approx(
        -rotors["rotor_hr_ccw"].tilt_side_deg, abs=1e-9)


def test_side_tilt_carries_the_z_torque_authority(model):
    """The hind pair's side tilt is the main source of torque about the pivot's z.

    At the shipped 40 deg forward tilt the front rotors are no longer negligible
    here -- a thrust axis leaning that far into the wall makes yaw torque through
    the rotor's own lateral offset, y_i*sin(beta) -- so this is a ratio, not the
    order-of-magnitude gap a near-level quad would show.
    """
    z_row = np.abs(model.B_tau[2])
    hind = [i for i, r in enumerate(model.body.rotors) if abs(r.tilt_side_deg) > 1.0]
    front = [i for i in range(model.n_u) if i not in hind]
    assert z_row[hind].min() > 2.0 * z_row[front].max()


def test_side_tilt_is_what_makes_the_map_well_conditioned(tmp_path):
    """The design intent: without side tilt, yaw couples to roll and the trim
    solve degrades. Compare the shipped geometry against its no-side-tilt twin."""
    import io, contextlib, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "scripts"))
    import gen_urdf
    from tilt_drone.trim import trim_matrix

    def conditioning(side):
        out = str(tmp_path / f"twin_{side[2]}.urdf")
        with contextlib.redirect_stdout(io.StringIO()):
            gen_urdf.main(out_path=out, side_deg=side)
        m = DroneWallModel(load_body(out), kappa=0.016)
        sv = np.linalg.svd(m.B_tau, compute_uv=False)
        return np.linalg.cond(trim_matrix(m, np.zeros(3))), sv[0] / sv[-1]

    with_side = conditioning([0, 0, 10, -10])
    without = conditioning([0, 0, 0, 0])
    assert with_side[0] < 0.5 * without[0]      # trim map
    assert with_side[1] < 0.5 * without[1]      # torque map alone


def test_rotor_mount_angles_respect_the_hardware_cap(model):
    """Mount angle is the total tilt of the thrust axis from the airframe z axis;
    side tilt spends the same budget as forward tilt."""
    for r in model.body.rotors:
        mount = np.degrees(np.arccos(np.clip(r.axis[2], -1, 1)))
        assert mount <= 40.0 + 1e-3, f"{r.name} mount angle {mount:.2f} deg exceeds the cap"


# ----------------------------------------------------------------------- trim
def test_trim_is_exact_inside_the_feasible_range(model):
    for q_deg in ([0, 0, 0], [2, -2, 3], [-3, 1, -4]):
        q = np.deg2rad(q_deg)
        rng = push_force_range(model, q, U_MAX)
        assert rng is not None
        f_push = float(np.mean(rng))
        tr = solve_trim(model, q, f_push, U_MAX)
        assert tr.feasible
        assert np.linalg.norm(tr.tau_residual) < 1e-9
        assert tr.f_push == pytest.approx(f_push, abs=1e-6)
        assert np.all(tr.u >= -1e-9) and np.all(tr.u <= U_MAX + 1e-9)


def test_trim_prioritises_torque_when_push_is_impossible(model):
    q = np.zeros(3)
    rng = push_force_range(model, q, U_MAX)
    tr = solve_trim(model, q, rng[1] + 5.0, U_MAX)
    assert not tr.feasible
    assert np.linalg.norm(tr.tau_residual) < 1e-3     # attitude balance still held
    assert tr.f_push < rng[1] + 1e-3                  # push was the thing relaxed
    assert tr.f_push < tr.f_push_cmd - 1.0            # ...and relaxed a long way


def test_cone_range_is_inside_the_torque_range(model):
    q = np.zeros(3)
    box = push_force_range(model, q, U_MAX)
    cone = cone_feasible_push_range(model, q, U_MAX, mu=0.4)
    assert cone is not None
    assert box[0] - 1e-6 <= cone[0] <= cone[1] <= box[1] + 1e-6


def test_pitching_up_reduces_the_achievable_push(model):
    """Effective tilt into the wall shrinks with pitch, so the push range drops."""
    flat = push_force_range(model, np.zeros(3), U_MAX)
    pitched = push_force_range(model, np.deg2rad([0, 3, 0]), U_MAX)
    assert pitched[1] < flat[1]


# --------------------------------------------------------------- friction cone
def test_cone_geometry():
    n = np.array([1.0, 0, 0])
    assert check_cone(np.array([10.0, 1.0, 1.0]), mu=0.5, normal=n).ok
    slipping = check_cone(np.array([1.0, 2.0, 0.0]), mu=0.5, normal=n)
    assert not slipping.ok and "cone" in slipping.reason
    tension = check_cone(np.array([-1.0, 0.0, 0.0]), mu=0.5, normal=n)
    assert not tension.ok and "contact lost" in tension.reason
    st = check_cone(np.array([10.0, 3.0, 4.0]), mu=0.5, normal=n)
    assert st.f_n == pytest.approx(10.0) and st.f_t == pytest.approx(5.0)
    assert st.margin == pytest.approx(0.0)


def test_reaction_force_satisfies_the_com_momentum_balance(model):
    """f_c comes from linear momentum; check it against the *angular* balance
    about the centre of mass, which is not used in deriving it."""
    rng = np.random.default_rng(1)
    r = model.r_cm
    I_cm = model.I_O - model.m * ((r @ r) * np.eye(3) - np.outer(r, r))
    for _ in range(5):
        x = np.concatenate([rng.uniform(-0.4, 0.4, 3), rng.uniform(-0.6, 0.6, 3)])
        u = rng.uniform(0.0, 8.0, 4)
        wd = model.omega_dot(x, u)
        f_c = model.reaction_force(x, u, wd)
        R = model.R(x[:3])
        # thrust torque about the CoM + reaction torque about the CoM (body frame)
        tau_thrust_cm = sum(np.cross(rr.pos - r, ui * rr.axis)
                            for rr, ui in zip(model.body.rotors, u))
        tau_drag = -sum(rr.spin * model.kappa * ui * rr.axis
                        for rr, ui in zip(model.body.rotors, u))
        tau_reaction = np.cross(-r, R.T @ f_c)
        lhs = I_cm @ wd + np.cross(x[3:], I_cm @ x[3:])
        assert lhs == pytest.approx(tau_thrust_cm + tau_drag + tau_reaction, abs=1e-9)


# ------------------------------------------------------------------------ LQR
def test_closed_loop_is_stable_at_the_reference(model):
    q = np.zeros(3)
    tr = solve_trim(model, q, float(np.mean(push_force_range(model, q, U_MAX))), U_MAX)
    x_ref = np.zeros(6)
    ctrl = LQRController(model, x_ref, tr.u, np.diag([40, 40, 40, 4, 4, 4]),
                         0.2 * np.eye(4), U_MAX)
    assert ctrl.controllability_rank() == 6
    ctrl(x_ref)
    A, B = model.linearize(x_ref, tr.u)
    assert np.max(np.linalg.eigvals(A - B @ ctrl.K).real) < -1e-3


def test_open_loop_is_unstable(model):
    """Sanity: the plant genuinely needs the controller."""
    A, _ = model.linearize(np.zeros(6), np.zeros(4))
    assert np.max(np.linalg.eigvals(A).real) > 1e-3


def test_closed_loop_converges_in_genesis():
    """End-to-end: Genesis plant + per-step LQR drives a real offset to zero."""
    from tilt_drone.plant import GenesisPlant
    model = DroneWallModel(load_body(URDF), kappa=0.016)
    q_ref = np.zeros(3)
    tr = solve_trim(model, q_ref, float(np.mean(cone_feasible_push_range(model, q_ref, U_MAX, 0.4))), U_MAX)
    ctrl = LQRController(model, np.zeros(6), tr.u, np.diag([40, 40, 40, 4, 4, 4]),
                         0.2 * np.eye(4), U_MAX)
    plant = GenesisPlant(model, URDF, dt=0.002, show_viewer=False, wall_visual=False,
                         draw_forces=False)
    plant.set_x(np.concatenate([np.deg2rad([6.0, 4.0, -5.0]), np.zeros(3)]))
    for _ in range(1500):                      # 3 s
        u = ctrl(plant.get_x())
        plant.apply_rotor_forces(u)
        plant.step()
    x = plant.get_x()
    assert np.rad2deg(np.linalg.norm(x[:3])) < 0.1
    assert np.linalg.norm(x[3:]) < 0.02
