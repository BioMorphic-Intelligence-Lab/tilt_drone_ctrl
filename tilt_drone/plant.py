"""Genesis plant: the URDF is the simulated truth, forces go in as rotor thrusts.

The three URDF revolute joints are the spherical joint, so ``get_state`` returns
the model's state directly.  Each rotor's thrust acts along the +z axis of that
rotor's own link frame, taken from the *plant's* current link orientation rather
than from the model's geometry -- so the tilt encoded in the URDF's ``rpy`` is
what physically acts, and the tests comparing plant to model stay meaningful.

Note on ``local=True``: Genesis rotates a local force by the link's *inertial*
frame (``i_quat``), which is a principal-axis frame of the inertia tensor, not
the link frame.  A rotor disc has two equal principal moments, so that frame is
arbitrary -- here it mapped local +z onto world -y.  Forces are therefore built
in world coordinates from ``get_links_quat`` and applied with ``local=False``.
"""
from __future__ import annotations

import numpy as np
import genesis as gs


class GenesisPlant:
    def __init__(self, model, urdf_path, dt=0.005, pivot_height=1.2,
                 show_viewer=False, record=False, camera_res=(1280, 720),
                 draw_forces=True, wall_visual=True, substeps=1):
        self.model = model
        self.urdf_path = urdf_path
        self.dt = dt
        self.pivot_height = pivot_height
        self.show_viewer = show_viewer
        self.record = record
        self.draw_forces = draw_forces
        self.rotor_names = [r.name for r in model.body.rotors]

        # Frame the shot around the robot's actual reach, so a short beam does not
        # end up as a speck in a view composed for a long one. The reference pose
        # below was tuned at reach = 0.96 m.
        reach = max(float(np.linalg.norm(r.pos)) for r in model.body.rotors)
        s = reach / 0.96
        look = (0.62 * s, 0.0, pivot_height - 0.02)
        cam = (look[0] + 1.48 * s, -2.3 * s, look[2] + 0.77 * s)
        self._vis_scale = s          # force arrows shrink with the vehicle too

        if not gs._initialized:      # gs.init is process-global and raises if repeated
            gs.init(backend=gs.cpu, logging_level="warning")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, gravity=tuple(model.g)),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=cam, camera_lookat=look, camera_fov=40,
                # 'max_FPS' was renamed 'refresh_rate' in Genesis 1.x; no display
                # redraws faster than ~60 Hz, so don't ask for the full sim rate.
                refresh_rate=min(60, max(1, int(round(1.0 / dt)))),
            ),
            vis_options=gs.options.VisOptions(show_world_frame=True, world_frame_size=0.4),
            rigid_options=gs.options.RigidOptions(enable_collision=False, enable_self_collision=False),
            show_viewer=show_viewer,
        )
        if wall_visual:
            self.scene.add_entity(
                gs.morphs.Box(pos=(-0.06, 0.0, pivot_height), size=(0.12, 3.0, 3.0),
                              fixed=True, collision=False),
                surface=gs.surfaces.Default(color=(0.55, 0.57, 0.60)),
            )
        self.drone = self.scene.add_entity(gs.morphs.URDF(
            file=urdf_path, fixed=True, pos=(0.0, 0.0, pivot_height),
            collision=False, merge_fixed_links=True, links_to_keep=self.rotor_names,
        ))
        self.cam = None
        self._video_path = None
        if record:
            # debug=True is what makes the thrust/contact arrows appear in the
            # recording; without it the camera skips marker geometry.
            self.cam = self.scene.add_camera(res=camera_res, pos=cam, lookat=look,
                                             fov=40, GUI=False, debug=True)
        self.scene.build()

        # Genesis defaults to armature = 0.1 on every dof, which is fictitious
        # rotor inertia the analytic model knows nothing about.  Zero it, or the
        # plant and the LQR model quietly disagree.
        self.drone.set_dofs_armature(np.zeros(self.drone.n_dofs))
        self.drone.set_dofs_damping(np.zeros(self.drone.n_dofs))
        self.rotor_idx = [self.drone.get_link(n).idx for n in self.rotor_names]
        self.rotor_idx_local = [self.drone.get_link(n).idx_local for n in self.rotor_names]
        self._solver = self.scene.sim.rigid_solver

    # ------------------------------------------------------------------ state
    def get_state(self):
        """(q, q_dot) of the three spherical-joint dofs."""
        q = np.asarray(self.drone.get_dofs_position().cpu(), float).ravel()
        qd = np.asarray(self.drone.get_dofs_velocity().cpu(), float).ravel()
        return q, qd

    def get_x(self):
        """State in the model's coordinates: [q, omega_body]."""
        q, qd = self.get_state()
        return np.concatenate([q, self.model.E(q) @ qd])

    def set_state(self, q, qd=None):
        self.drone.set_dofs_position(np.asarray(q, float))
        self.drone.set_dofs_velocity(np.zeros(3) if qd is None else np.asarray(qd, float))

    def set_x(self, x):
        x = np.asarray(x, float)
        self.set_state(x[:3], self.model.T(x[:3]) @ x[3:])

    # --------------------------------------------------------------- actuation
    def rotor_axes_world(self) -> np.ndarray:
        """(n, 3) thrust directions, read from the plant's own link frames."""
        quats = np.asarray(self.drone.get_links_quat(self.rotor_idx_local).cpu(), float).reshape(-1, 4)
        return np.array([_quat_rotate(q, np.array([0.0, 0.0, 1.0])) for q in quats])

    def apply_rotor_forces(self, u):
        """Thrust along each rotor's own +z, plus its propeller drag reaction."""
        u = np.asarray(u, float)
        axes = self.rotor_axes_world()
        forces = axes * u[:, None]
        drag = np.array([-r.spin * self.model.kappa * ui
                         for r, ui in zip(self.model.body.rotors, u)])
        torques = axes * drag[:, None]
        self._solver.apply_links_external_force(forces, self.rotor_idx, ref="link_origin", local=False)
        self._solver.apply_links_external_torque(torques, self.rotor_idx, ref="link_origin", local=False)

    def step(self):
        self.scene.step()
        if self.cam is not None:
            self.cam.render()

    # ------------------------------------------------------------------- visuals
    def draw(self, u, f_c=None, cone_ok=True):
        """Thrust arrows on each rotor and the contact force at the pivot."""
        if not (self.draw_forces and (self.show_viewer or self.cam is not None)):
            return
        self.scene.clear_debug_objects()
        scale = 0.06 * self._vis_scale
        max_len = 0.55 * self._vis_scale
        pos_all = np.asarray(self.drone.get_links_pos(self.rotor_idx_local).cpu(), float).reshape(-1, 3)
        for pos, axis, ui in zip(pos_all, self.rotor_axes_world(), u):
            if ui > 1e-6:
                self.scene.draw_debug_arrow(pos=pos, vec=_clamp_arrow(axis * ui, scale, max_len),
                                            radius=0.006, color=(0.15, 0.75, 1.0, 1.0))
        if f_c is not None:
            colour = (0.2, 0.9, 0.3, 1.0) if cone_ok else (1.0, 0.25, 0.2, 1.0)
            self.scene.draw_debug_arrow(pos=(0.0, 0.0, self.pivot_height),
                                        vec=_clamp_arrow(-np.asarray(f_c), 0.12 * self._vis_scale,
                                                         max_len),
                                        radius=0.008, color=colour)

    def start_recording(self, path, fps=30):
        """Genesis 1.x names the output file when recording *starts*."""
        if self.cam is not None:
            self.cam.start_recording(save_to_filename=path, fps=fps)
            self._video_path = path

    def save_video(self):
        if self.cam is not None:
            self.cam.stop_recording()
            print(f"video written to {self._video_path}")


def _clamp_arrow(vec, scale, max_len=0.55):
    """Newtons -> metres, capped: a 20 N contact force at the 3 N scale would
    otherwise draw an arrow longer than the whole scene."""
    v = np.asarray(vec, float) * scale
    n = float(np.linalg.norm(v))
    return v * (max_len / n) if n > max_len else v


def _quat_rotate(q, v):
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return R @ v
