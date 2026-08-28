#!/usr/bin/env python3
"""Generate urdf/tilt_drone.urdf.

The URDF is the source of truth for the simulation: the analytic model used by
the LQR parses it, and Genesis loads the same file as the plant.  This script
only exists so that geometry and the inertia tensors stay consistent when you
change dimensions.  Editing the rotor ``rpy`` values directly in the generated
URDF is perfectly fine -- nothing here caches them.

Frame convention
----------------
World/base:  the wall occupies x < 0, its outward normal is +x, and the
spherical joint sits at the origin of the robot's base link.
Body frame (the frame after the third revolute joint) has its origin *at the
pivot*, +x pointing away from the wall along the beam, +z up.  Hence

    front rotors  (nearer the wall)  ->  x = L_beam - dx
    hind  rotors  (further away)     ->  x = L_beam + dx

Rotor tilt
----------
Each rotor's thrust axis is the +z axis of its own link frame, so the tilt is
encoded entirely in the fixed joint's ``rpy``.  With rpy = (r, p, 0) the thrust
direction in body coordinates is

    d = (sin(p)cos(r), -sin(r), cos(p)cos(r))

* all four rotors get pitch = -beta_fwd, which tilts the thrust into the wall (-x)
* the two hind rotors additionally get roll = -/+ gamma_side, tilting each one
  towards its own side (+y for hind-left, -y for hind-right).  That pair forms a
  couple: it produces torque about the pivot's z axis -- the axis a flat quad has
  almost no authority over -- without a net side force at equal thrust.
"""
import math
import os

# ----------------------------------------------------------------- geometry --
L_BEAM = 0.50        # m, pivot (wall) to drone body centre
R_BEAM = 0.012       # m
M_BEAM = 0.20        # kg

BODY_L, BODY_W, BODY_H = 0.18, 0.18, 0.06
M_BODY = 1.20        # kg (includes arms, which are visual-only)

DX, DY = 0.15, 0.15  # m, rotor offsets from the body centre
DZ = 0.03            # m, rotor plane above the body centre
R_ROTOR = 0.065      # m, propeller disc radius (visual)
M_ROTOR = 0.05       # kg, motor + prop

# ------------------------------------------------------------------- tilts ---
# The hardware limit is on each rotor's *mount angle* -- the total angle between
# its thrust axis and the airframe's z axis -- not on the forward tilt alone, so
# side tilt eats into the same budget:  mount = acos(cos(fwd) * cos(side)).
# The hind pair spends 10 deg of its budget sideways for yaw authority; the front
# pair has no side tilt and so can use the whole 40 deg going forward.
MOUNT_CAP_DEG = 40.0
BETA_FWD_DEG = [40.0, 40.0, 38.9298, 38.9298]   # front-left, front-right, hind-left, hind-right
GAMMA_SIDE_DEG = 10.0  # hind pair only, tilted towards their own side

M_DUMMY = 1e-6        # massless-in-spirit links carrying the 3 revolute dofs
I_DUMMY = 1e-9


def cylinder_inertia(m, r, h, axis="z"):
    """Solid cylinder about its centre of mass."""
    i_ax = 0.5 * m * r * r
    i_tr = m * (3.0 * r * r + h * h) / 12.0
    return {"x": (i_ax, i_tr, i_tr), "y": (i_tr, i_ax, i_tr), "z": (i_tr, i_tr, i_ax)}[axis]


def box_inertia(m, lx, ly, lz):
    return (m * (ly * ly + lz * lz) / 12.0,
            m * (lx * lx + lz * lz) / 12.0,
            m * (lx * lx + ly * ly) / 12.0)


def inertial(m, ixx, iyy, izz, xyz=(0, 0, 0), rpy=(0, 0, 0)):
    return f"""    <inertial>
      <origin xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>
      <mass value="{m:.8f}"/>
      <inertia ixx="{ixx:.8e}" ixy="0" ixz="0" iyy="{iyy:.8e}" iyz="0" izz="{izz:.8e}"/>
    </inertial>"""


def dummy_link(name):
    return f"""  <link name="{name}">
{inertial(M_DUMMY, I_DUMMY, I_DUMMY, I_DUMMY)}
  </link>"""


def main(out_path=None, beta_deg=None, gamma_deg=None, side_deg=None, beam_len=None):
    """Write the URDF. Angles default to the module constants above; passing them
    explicitly is how the tilt studies generate variants.

    beta_deg  : forward tilt into the wall; scalar, or 4 values in the rotor order
                (front-left, front-right, hind-left, hind-right).
    gamma_deg : side tilt applied to the hind pair only (outboard), the shorthand
                for the original layout.
    side_deg  : side tilt for all four rotors, 4 values in the same order,
                positive = towards +y. Overrides gamma_deg when given.
    beam_len  : pivot-to-body distance [m]; defaults to L_BEAM.
    """
    L = L_BEAM if beam_len is None else beam_len
    beta_in = BETA_FWD_DEG if beta_deg is None else beta_deg
    betas = [math.radians(b) for b in
             (beta_in if hasattr(beta_in, "__len__") else [beta_in] * 4)]
    gamma = math.radians(GAMMA_SIDE_DEG if gamma_deg is None else gamma_deg)
    if side_deg is not None:
        sides = [math.radians(g) for g in side_deg]
    else:
        sides = [0.0, 0.0, gamma, -gamma]      # hind pair only, outboard

    # rotor name -> (position relative to the *body* link, rpy of its mount joint)
    # names end in _ccw / _cw: the parser reads the propeller spin from there,
    # since URDF has nowhere standard to put it.
    positions = [(-DX, +DY, DZ), (-DX, -DY, DZ), (+DX, +DY, DZ), (+DX, -DY, DZ)]
    names = ["rotor_fl_ccw", "rotor_fr_cw", "rotor_hl_cw", "rotor_hr_ccw"]
    # rpy = (roll, pitch, 0): pitch tilts the thrust axis into the wall (-x),
    # roll tilts it sideways (roll = -g puts the thrust component at +y).
    rotors = {n: (p, (-g, -b, 0.0))
              for n, p, b, g in zip(names, positions, betas, sides)}

    bi = box_inertia(M_BODY, BODY_L, BODY_W, BODY_H)
    beam_i = cylinder_inertia(M_BEAM, R_BEAM, L, axis="x")
    rot_i = cylinder_inertia(M_ROTOR, R_ROTOR, 0.01, axis="z")

    arm_len = math.hypot(DX, DY)
    arms = []
    for sx, sy in ((+1, +1), (+1, -1), (-1, +1), (-1, -1)):
        yaw = math.atan2(sy * DY, sx * DX)
        arms.append(f"""    <visual>
      <origin xyz="{sx * DX / 2:.6f} {sy * DY / 2:.6f} 0" rpy="0 0 {yaw:.6f}"/>
      <geometry><box size="{arm_len:.6f} 0.015 0.015"/></geometry>
      <material name="grey"/>
    </visual>""")
    arms = "\n".join(arms)

    rotor_blocks = []
    for name, (pos, rpy) in rotors.items():
        rotor_blocks.append(f"""  <joint name="{name}_mount" type="fixed">
    <parent link="body"/>
    <child link="{name}"/>
    <!-- tilt lives here: rpy = (roll, pitch, yaw); thrust is this frame's +z -->
    <origin xyz="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>
  </joint>
  <link name="{name}">
{inertial(M_ROTOR, *rot_i)}
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="{R_ROTOR:.4f}" length="0.008"/></geometry>
      <material name="dark"/>
    </visual>
  </link>""")
    rotor_blocks = "\n".join(rotor_blocks)

    urdf = f"""<?xml version="1.0"?>
<!-- Generated by scripts/gen_urdf.py. Safe to hand-edit; nothing regenerates
     it automatically.  Rotor tilt = the rpy of each *_mount joint. -->
<robot name="tilt_drone">

  <material name="grey"><color rgba="0.35 0.35 0.38 1"/></material>
  <material name="dark"><color rgba="0.12 0.12 0.14 1"/></material>
  <material name="beam"><color rgba="0.85 0.55 0.15 1"/></material>
  <material name="blue"><color rgba="0.20 0.45 0.80 1"/></material>

  <!-- ============ spherical joint at the wall, as 3 revolute dofs ========= -->
  <link name="wall_anchor">
{inertial(M_DUMMY, I_DUMMY, I_DUMMY, I_DUMMY)}
  </link>

  <joint name="ball_x" type="continuous">
    <parent link="wall_anchor"/>
    <child link="pivot_x"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <dynamics damping="0.0" friction="0.0"/>
  </joint>
{dummy_link("pivot_x")}

  <joint name="ball_y" type="continuous">
    <parent link="pivot_x"/>
    <child link="pivot_y"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.0" friction="0.0"/>
  </joint>
{dummy_link("pivot_y")}

  <joint name="ball_z" type="continuous">
    <parent link="pivot_y"/>
    <child link="beam"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <dynamics damping="0.0" friction="0.0"/>
  </joint>

  <!-- ================= everything below here is one rigid body ============ -->
  <link name="beam">
{inertial(M_BEAM, *beam_i, xyz=(L / 2, 0, 0))}
    <visual>
      <origin xyz="{L / 2:.6f} 0 0" rpy="0 1.5707963 0"/>
      <geometry><cylinder radius="{R_BEAM:.4f}" length="{L:.4f}"/></geometry>
      <material name="beam"/>
    </visual>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.025"/></geometry>
      <material name="blue"/>
    </visual>
  </link>

  <joint name="beam_to_body" type="fixed">
    <parent link="beam"/>
    <child link="body"/>
    <origin xyz="{L:.6f} 0 0" rpy="0 0 0"/>
  </joint>

  <link name="body">
{inertial(M_BODY, *bi)}
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{BODY_L} {BODY_W} {BODY_H}"/></geometry>
      <material name="grey"/>
    </visual>
{arms}
  </link>

{rotor_blocks}

</robot>
"""
    out = out_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "urdf", "tilt_drone.urdf")
    with open(out, "w") as f:
        f.write(urdf)
    print(f"wrote {out}")
    print(f"  forward tilt : {[round(math.degrees(b), 1) for b in betas]} deg into the wall")
    print(f"  side tilt    : {[round(math.degrees(g), 1) for g in sides]} deg towards +y")
    print(f"  beam length  : {L:.3f} m")
    mounts = [math.degrees(math.acos(math.cos(b) * math.cos(g))) for b, g in zip(betas, sides)]
    print(f"  mount angle  : {[round(a, 2) for a in mounts]} deg"
          f"  (cap {MOUNT_CAP_DEG:.1f})")
    if max(mounts) > MOUNT_CAP_DEG + 1e-6:
        print(f"  [WARN] mount angle exceeds the {MOUNT_CAP_DEG:.1f} deg hardware cap")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate a tilt_drone URDF.")
    ap.add_argument("--out", default=None, help="output path (default: urdf/tilt_drone.urdf)")
    ap.add_argument("--beta", type=float, nargs="+", default=None,
                    help="forward tilt into the wall [deg]: one value, or four in the "
                         "order front-left front-right hind-left hind-right")
    ap.add_argument("--side", type=float, nargs=4, default=None,
                    help="side tilt [deg] for all four rotors, same order, +ve towards +y")
    ap.add_argument("--gamma", type=float, default=None,
                    help="shorthand: side tilt on the hind pair only, outboard [deg]")
    ap.add_argument("--beam", type=float, default=None,
                    help="pivot-to-body beam length [m] (default %.2f)" % L_BEAM)
    args = ap.parse_args()
    beta = args.beta if args.beta is None or len(args.beta) > 1 else args.beta[0]
    main(out_path=args.out, beta_deg=beta, gamma_deg=args.gamma, side_deg=args.side,
         beam_len=args.beam)
