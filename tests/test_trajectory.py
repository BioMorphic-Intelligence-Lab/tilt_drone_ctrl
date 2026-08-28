"""Parsing of step-reference trajectory files, and applying one end to end."""
import numpy as np
import pytest

from tilt_drone.trajectory import load_trajectory

EXAMPLE = "trajectories/three_steps.traj.txt"


def test_example_file_parses():
    """Structural, not literal: the example is meant to be edited."""
    wps = load_trajectory(EXAMPLE)
    assert len(wps) >= 3
    assert wps[0].t == 0.0
    assert all(b.t > a.t for a, b in zip(wps, wps[1:]))
    assert all(w.q_deg.shape == (3,) for w in wps)
    assert wps[1].q == pytest.approx(np.deg2rad(wps[1].q_deg))


def test_comments_blank_lines_and_commas(tmp_path):
    f = tmp_path / "t.traj.txt"
    f.write_text("# header\n\n  0.0 1 2 3   # inline comment\n\n1.5, 4, 5, 6\n")
    wps = load_trajectory(str(f))
    assert len(wps) == 2
    assert wps[0].q_deg == pytest.approx([1, 2, 3])
    assert wps[1].t == 1.5 and wps[1].q_deg == pytest.approx([4, 5, 6])


@pytest.mark.parametrize("text, msg", [
    ("0.0 1 2\n", "expected 4 values"),
    ("0.0 1 2 3 4\n", "expected 4 values"),
    ("0.0 1 2 3\n0.0 4 5 6\n", "strictly increase"),
    ("2.0 1 2 3\n1.0 4 5 6\n", "strictly increase"),
    ("-1.0 1 2 3\n", "non-negative"),
    ("# only a comment\n", "no waypoints"),
    ("0.0 1 two 3\n", "could not convert"),
])
def test_bad_files_are_rejected_with_the_line(tmp_path, text, msg):
    f = tmp_path / "bad.traj.txt"
    f.write_text(text)
    with pytest.raises(ValueError, match=msg):
        load_trajectory(str(f))


def test_driver_applies_every_waypoint(tmp_path):
    """End to end: each step actually reaches the controller's reference."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import run_sim

    traj = tmp_path / "t.traj.txt"
    traj.write_text("0.0  0 0 0\n0.3  0 1 2\n0.6  1 0 -2\n")
    run_sim.main(["--traj", str(traj), "--T", "0.9", "--dt", "0.01", "--q0", "0", "0", "0",
                  "--out", str(tmp_path), "--no-save-plot"])
    d = np.load(tmp_path / "run.npz")
    applied = np.unique(np.round(np.rad2deg(d["q_ref"]), 6), axis=0)
    assert len(applied) == 3
    assert d["q_ref"][-1] == pytest.approx(np.deg2rad([1, 0, -2]))
