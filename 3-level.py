import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.linalg import expm

# ============================================================
# Import configuration
# ============================================================
import nv_config as cfg

gamma_e = cfg.gamma_e
D = cfg.D
two_pi = cfg.two_pi
Omega = cfg.Omega

# ============================================================
# Spin-1 operators  (|-1>, |0>, |+1>)
# ============================================================

Sx = (1 / np.sqrt(2)) * np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=complex)

Sz = np.diag([-1, 0, +1])

# ============================================================
# Time grid
# ============================================================

t = np.linspace(0, cfg.T_sim, cfg.nsteps)
dt = t[1] - t[0]

# ============================================================
# Hamiltonians
# ============================================================

def H_static(Bz):

    return two_pi * (
        D * (Sz @ Sz)
        + gamma_e * Bz * Sz
    )


def H_mw(ti, f_mw):

    return (
        two_pi
        * Omega
        * np.cos(two_pi * f_mw * ti)
        * Sx
    )


def H_RWA_3level(Bz, detuning):

    # transition frequencies
    w_m1 = D - gamma_e * Bz
    w_p1 = D + gamma_e * Bz

    # ========================================================
    # Microwave frequency fixed around D
    # User tunes with detuning slider
    # ========================================================

    f_mw = D + detuning

    H = np.zeros((3, 3), dtype=complex)

    # rotating-frame detunings
    H[0, 0] = two_pi * (w_m1 - f_mw)
    H[2, 2] = two_pi * (w_p1 - f_mw)

    # coupling strength
    g = two_pi * Omega / (2 * np.sqrt(2))

    H[0, 1] = g
    H[1, 0] = g

    H[1, 2] = g
    H[2, 1] = g

    return H

# ============================================================
# Simulations
# ============================================================

def simulate_lab(Bz, detuning):

    # microwave fixed around D
    f_mw = D + detuning

    psi = np.array([0, 1, 0], dtype=complex)

    P = np.zeros((cfg.nsteps, 3))

    for i, ti in enumerate(t):

        P[i] = np.abs(psi)**2

        if i < cfg.nsteps - 1:

            H = H_static(Bz) + H_mw(ti, f_mw)

            psi = expm(-1j * H * dt) @ psi
            psi /= np.linalg.norm(psi)

    return P


def simulate_rwa(Bz, detuning):

    psi = np.array([0, 1, 0], dtype=complex)

    P = np.zeros((cfg.nsteps, 3))

    H = H_RWA_3level(Bz, detuning)

    U = expm(-1j * H * dt)

    for i in range(cfg.nsteps):

        P[i] = np.abs(psi)**2

        if i < cfg.nsteps - 1:
            psi = U @ psi

    return P

# ============================================================
# π/2 detection
# ============================================================

def find_pi2_time(P):

    P0 = P[:, 1]

    for i in range(1, len(P0)):

        if P0[i] <= 0.5:
            return t[i], i

    return t[-1], len(t) - 1

# ============================================================
# Initial populations
# ============================================================

P_rwa = simulate_rwa(0.0, 0.0)

t_pi2_rwa, idx_pi2_rwa = find_pi2_time(P_rwa)

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

plt.subplots_adjust(bottom=0.28)

labels = [r"|-1⟩", r"|0⟩", r"|+1⟩"]
colors = ["red", "blue", "green"]

lines_rwa = []

for i in range(3):

    line, = ax.plot(
        t,
        P_rwa[:, i],
        color=colors[i],
        label=labels[i]
    )

    lines_rwa.append(line)

# π/2 marker
vline_rwa = ax.axvline(
    t_pi2_rwa,
    ls='--',
    lw=2,
    color='black'
)

dot_rwa_m1, = ax.plot([], [], 'o', color='red')

label_rwa_m1 = ax.text(0, 0, '')

# ============================================================
# Axes formatting
# ============================================================

ax.set_xlabel("Time (µs)")
ax.set_ylabel("Population")

ax.set_ylim(-0.02, 1.02)

ax.grid(alpha=0.3)

ax.legend()

title = ax.set_title("")

# ============================================================
# Sliders
# ============================================================

ax_Bz = plt.axes([0.15, 0.15, 0.7, 0.03])

ax_det = plt.axes([0.15, 0.10, 0.7, 0.03])

Bz_slider = Slider(
    ax_Bz,
    "Bz (G)",
    cfg.Bz_min,
    cfg.Bz_max,
    valinit=0.0
)

det_slider = Slider(
    ax_det,
    "Detuning Δ (MHz)",
    cfg.detuning_min,
    cfg.detuning_max,
    valinit=0.0
)

# ============================================================
# Update
# ============================================================

def update(val):

    Bz = Bz_slider.val
    det = det_slider.val

    # simulate
    P_rwa = simulate_rwa(Bz, det)

    # update curves
    for i in range(3):

        lines_rwa[i].set_ydata(P_rwa[:, i])

    # π/2 point
    t_pi2_rwa, idx_rwa = find_pi2_time(P_rwa)

    vline_rwa.set_xdata([t_pi2_rwa, t_pi2_rwa])

    dot_rwa_m1.set_data([t_pi2_rwa], [0.5])

    # label
    label_rwa_m1.set_position(
        (t_pi2_rwa + 0.03, 0.55)
    )

    label_rwa_m1.set_text(
        f"tπ/2 = {t_pi2_rwa:.3f} µs"
    )

    # ========================================================
    # Show frequencies
    # ========================================================

    w_m1 = D - gamma_e * Bz
    w_p1 = D + gamma_e * Bz

    f_mw = D + det

    title.set_text(
        f"MW = {f_mw:.3f} MHz   |   "
        f"|-1↔0| = {w_m1:.3f} MHz   |   "
        f"|+1↔0| = {w_p1:.3f} MHz"
    )

    fig.canvas.draw_idle()

# ============================================================
# Connect sliders
# ============================================================

Bz_slider.on_changed(update)

det_slider.on_changed(update)

# initial draw
update(None)

plt.show()