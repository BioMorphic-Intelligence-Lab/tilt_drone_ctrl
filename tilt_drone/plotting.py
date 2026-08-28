"""Diagnostic plots for a run: states, actuation, contact.

Colour is assigned to the *entity*, fixed order, never cycled: the pivot axes
x/y/z keep the same three hues in every panel that shows them, and each rotor
keeps its hue between panels.  Rotor traces carry end-of-line labels as well as
a legend, which is also the relief for the aqua slot's sub-3:1 contrast.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# validated categorical slots (light surface): blue, orange, aqua, violet
AXIS_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
ROTOR_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


@dataclass
class SimLog:
    t: list = field(default_factory=list)
    q: list = field(default_factory=list)
    omega: list = field(default_factory=list)
    q_ref: list = field(default_factory=list)
    u: list = field(default_factory=list)
    f_c: list = field(default_factory=list)
    f_n: list = field(default_factory=list)
    f_t: list = field(default_factory=list)
    margin: list = field(default_factory=list)
    ok: list = field(default_factory=list)

    def add(self, t, q, omega, q_ref, u, f_c, status):
        self.t.append(t); self.q.append(np.copy(q)); self.omega.append(np.copy(omega))
        self.q_ref.append(np.copy(q_ref)); self.u.append(np.copy(u))
        self.f_c.append(np.copy(status.f_c)); self.f_n.append(status.f_n)
        self.f_t.append(status.f_t); self.margin.append(status.margin); self.ok.append(status.ok)

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


def _style(ax, ylabel, title=None):
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=6)
    ax.grid(True, color="#000000", alpha=0.08, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, labelsize=8)


def _direct_labels(ax, x, values, names, colors, y_range=None, min_gap_frac=0.062):
    """End-of-line labels, nudged apart so overlapping series stay readable.

    The gap is measured against the panel's full plotted range, not the spread of
    the final values -- symmetric rotor pairs converge to the same number, and a
    gap scaled to that spread would be far smaller than the text.
    """
    order = np.argsort(values)
    span = y_range if y_range else max(np.ptp(values), 1e-9)
    gap = min_gap_frac * max(span, 1e-9)
    placed = {}
    last = -np.inf
    for i in order:
        y = max(values[i], last + gap) if last > -np.inf else values[i]
        placed[i] = y
        last = y
    for i, name in enumerate(names):
        ax.annotate(name, (x, placed[i]), xytext=(5, 0), textcoords="offset points",
                    color=colors[i], fontsize=8, va="center", annotation_clip=False)


def _shade_violations(ax, t, ok):
    """Light red bands wherever the friction cone was violated."""
    ok = np.asarray(ok, bool)
    if ok.all():
        return
    edges = np.diff(np.concatenate([[True], ok, [True]]).astype(int))
    for s, e in zip(np.where(edges == -1)[0], np.where(edges == 1)[0]):
        ax.axvspan(t[max(s - 1, 0)], t[min(e, len(t) - 1)], color=CRITICAL, alpha=0.10, lw=0)


def plot_run(log: SimLog, rotor_names, mu, u_max, path=None, show=False, title=""):
    d = log.arrays()
    t, q, w, qr, u = d["t"], np.rad2deg(d["q"]), d["omega"], np.rad2deg(d["q_ref"]), d["u"]
    fn, ft, margin, ok = d["f_n"], d["f_t"], d["margin"], d["ok"]

    matplotlib.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                                "savefig.facecolor": SURFACE, "font.size": 9})
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 9.0), sharex=True)
    fig.suptitle(title or "Wall-pinned tilted-rotor drone", color=INK, fontsize=12,
                 x=0.012, ha="left", y=0.985)

    labels = ("phi_x", "theta_y", "psi_z")
    ax = axes[0, 0]
    for i, (lab, c) in enumerate(zip(labels, AXIS_COLORS)):
        ax.plot(t, qr[:, i], color=c, lw=1.0, ls=":", alpha=0.7)
        ax.plot(t, q[:, i], color=c, lw=1.8, label=lab)
    _style(ax, "joint angle [deg]", "Spherical-joint angles (dotted = reference)")
    ax.legend(frameon=False, ncol=3, fontsize=8, labelcolor=MUTED)

    ax = axes[1, 0]
    for i, (lab, c) in enumerate(zip(labels, AXIS_COLORS)):
        ax.plot(t, w[:, i], color=c, lw=1.8, label=f"omega_{lab[-1]}")
    ax.axhline(0, color="#c9c8c3", lw=0.8)
    _style(ax, "body rate [rad/s]", "Angular velocity (body frame)")
    ax.legend(frameon=False, ncol=3, fontsize=8, labelcolor=MUTED)

    ax = axes[2, 0]
    err = np.rad2deg(np.linalg.norm(d["q"] - d["q_ref"], axis=1))
    ax.plot(t, err, color=AXIS_COLORS[0], lw=1.8)
    _style(ax, "attitude error [deg]", "Tracking error  |q - q_ref|")
    ax.set_xlabel("time [s]", color=MUTED, fontsize=9)

    ax = axes[0, 1]
    for i, (name, c) in enumerate(zip(rotor_names, ROTOR_COLORS)):
        ax.plot(t, u[:, i], color=c, lw=1.8, label=name)
    _direct_labels(ax, t[-1], u[-1], [n.replace("rotor_", "") for n in rotor_names],
                   ROTOR_COLORS, y_range=float(np.ptp(u)))
    ax.axhline(u_max, color=MUTED, lw=0.9, ls="--")
    ax.annotate("u_max", (t[0], u_max), xytext=(2, 3), textcoords="offset points",
                color=MUTED, fontsize=8)
    ax.set_ylim(bottom=min(-0.2, u.min() - 0.2))
    _style(ax, "rotor force [N]", "Actuation")
    ax.legend(frameon=False, ncol=2, fontsize=8, labelcolor=MUTED)

    ax = axes[1, 1]
    ax.plot(t, fn, color=AXIS_COLORS[0], lw=1.8, label="f_n (push into wall)")
    ax.plot(t, ft, color=AXIS_COLORS[1], lw=1.8, label="|f_t| (tangential)")
    ax.plot(t, mu * fn, color=CRITICAL, lw=1.4, ls="--", label=f"mu*f_n  (mu = {mu})")
    ax.axhline(0, color="#c9c8c3", lw=0.8)
    _shade_violations(ax, t, ok)
    _style(ax, "force [N]", "Contact force at the spherical joint")
    ax.legend(frameon=False, ncol=1, fontsize=8, labelcolor=MUTED)

    ax = axes[2, 1]
    ax.plot(t, margin, color=GOOD, lw=1.8)
    ax.axhline(0, color=CRITICAL, lw=1.2, ls="--")
    ax.annotate("slip", (t[0], 0), xytext=(2, -11), textcoords="offset points",
                color=CRITICAL, fontsize=8)
    _shade_violations(ax, t, ok)
    _style(ax, "cone margin [N]", "Friction-cone margin  mu*f_n - |f_t|  (negative = slipping)")
    ax.set_xlabel("time [s]", color=MUTED, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if path:
        fig.savefig(path, dpi=140)
        print(f"plots written to {path}")
    if show:
        plt.show()
    return fig
