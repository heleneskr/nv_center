import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.linalg import expm

# ============================================================
# Units
# ============================================================

def MHz_to_rad(x):
    return 2 * np.pi * x

# ============================================================
# Basis
# |0>, |+1>, |-1>
# ============================================================

sx = np.array([
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 0]
], dtype=complex)

sz = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
], dtype=complex)

# ============================================================
# Parameters
# ============================================================

gamma_e = 2.8  # MHz/G

# microwave drive
Omega = MHz_to_rad(0.3)

# dephasing
gamma_phi = 5.0  # us^-1

# ============================================================
# Optical transition rates (us^-1)
# ============================================================

# optical excitation
laser_rate = 20.0

# radiative decay
gamma_rad_0 = 66.16
gamma_rad_pm = 66.16

# ISC
gamma_isc_0 = 11.1
gamma_isc_pm = 91.8

# singlet decay
gamma_s0 = 4.87
gamma_spm = 2.04

# timestep
dt = 0.0005

# ============================================================
# Hamiltonian
# ============================================================

def H(Delta, Bz):

    Delta_p = Delta - gamma_e * Bz
    Delta_m = Delta + gamma_e * Bz

    return np.array([
        [0,                     Omega/2,               Omega/2],
        [Omega/2,  MHz_to_rad(Delta_p),                      0],
        [Omega/2,                     0,  MHz_to_rad(Delta_m)]
    ], dtype=complex)

# ============================================================
# Optical dynamics
# ============================================================

def optical_step(g, e, s, laser_on=True):

    g = g.copy()
    e = e.copy()

    # ========================================================
    # Optical excitation
    # ========================================================

    if laser_on:

        exc0 = laser_rate * g[0] * dt
        excp = laser_rate * g[1] * dt
        excm = laser_rate * g[2] * dt

        g[0] -= exc0
        g[1] -= excp
        g[2] -= excm

        e[0] += exc0
        e[1] += excp
        e[2] += excm

    # ========================================================
    # Radiative decay
    # ========================================================

    rad0 = gamma_rad_0 * e[0] * dt
    radp = gamma_rad_pm * e[1] * dt
    radm = gamma_rad_pm * e[2] * dt

    e[0] -= rad0
    e[1] -= radp
    e[2] -= radm

    g[0] += rad0
    g[1] += radp
    g[2] += radm

    # ========================================================
    # ISC to singlet
    # ========================================================

    isc0 = gamma_isc_0 * e[0] * dt

    iscp = gamma_isc_pm * e[1] * dt
    iscm = gamma_isc_pm * e[2] * dt

    e[0] -= isc0
    e[1] -= iscp
    e[2] -= iscm

    s += isc0 + iscp + iscm

    # ========================================================
    # Singlet decay
    # ========================================================

    ds0 = gamma_s0 * s * dt

    dsp_each = 0.5 * gamma_spm * s * dt

    total_ds = ds0 + 2 * dsp_each

    s -= total_ds

    g[0] += ds0
    g[1] += dsp_each
    g[2] += dsp_each

    # ========================================================
    # Normalize
    # ========================================================

    total = np.sum(g) + np.sum(e) + s

    if total > 0:

        g /= total
        e /= total
        s /= total

    return g, e, s

# ============================================================
# Continuous-wave evolution
# ============================================================

def evolve_time_trace(Delta, Bz, tmax=5):

    # ground-state density matrix
    rho_g = np.zeros((3, 3), dtype=complex)
    rho_g[0, 0] = 1.0

    # excited-state populations
    e = np.zeros(3)

    # singlet population
    s = 0.0

    steps = int(tmax / dt)

    # propagator
    U = expm(-1j * H(Delta, Bz) * dt)

    # traces
    times = []

    g_trace = []
    e_trace = []
    s_trace = []

    pl_trace = []

    for n in range(steps):

        t = n * dt

        # ====================================================
        # Coherent MW evolution
        # ====================================================

        rho_g = U @ rho_g @ U.conj().T

        # ====================================================
        # Dephasing
        # ====================================================

        decay = np.exp(-gamma_phi * dt)

        for i in range(3):
            for j in range(3):

                if i != j:
                    rho_g[i, j] *= decay

        # ====================================================
        # Extract populations
        # ====================================================

        g = np.real(np.diag(rho_g)).copy()

        # ====================================================
        # Optical dynamics (laser always ON)
        # ====================================================

        g, e, s = optical_step(
            g,
            e,
            s,
            laser_on=True
        )

        # ====================================================
        # Update density matrix populations
        # ====================================================

        for i in range(3):
            rho_g[i, i] = g[i]

        # ====================================================
        # PL signal
        # ====================================================

        signal = (
            gamma_rad_0 * e[0]
            + gamma_rad_pm * e[1]
            + gamma_rad_pm * e[2]
        )

        # ====================================================
        # Store traces
        # ====================================================

        times.append(t)

        g_trace.append(g.copy())
        e_trace.append(e.copy())
        s_trace.append(s)

        pl_trace.append(signal)

    return (
        np.array(times),
        np.array(g_trace),
        np.array(e_trace),
        np.array(s_trace),
        np.array(pl_trace)
    )

# ============================================================
# CW ODMR spectrum
# ============================================================

def compute_odmr(freqs, Bz):

    PL = []

    for f in freqs:

        Delta = f - D0

        _, _, e_t, _, pl_t = evolve_time_trace(
            Delta=Delta,
            Bz=Bz,
            tmax=5
        )

        # steady-state PL
        signal = pl_t[-1]

        PL.append(signal)

    PL = np.array(PL)

    PL /= np.max(PL)

    return PL

# ============================================================
# Setup
# ============================================================

D0 = 2870  # MHz

freqs = np.linspace(D0 - 10, D0 + 10, 400)

Bz0 = 0

# ODMR
PL = compute_odmr(freqs, Bz0)

# Time trace at resonance
times, g_t, e_t, s_t, pl_t = evolve_time_trace(
    Delta=0,
    Bz=Bz0,
    tmax=5
)

# ============================================================
# Figure
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    gridspec_kw={'height_ratios': [1, 2]}
)

plt.subplots_adjust(bottom=0.18)

# ============================================================
# TOP: ODMR
# ============================================================

line_odmr, = ax1.plot(
    freqs,
    PL,
    color='black',
    linewidth=2
)

ax1.set_title("CW ODMR")

ax1.set_ylabel("Normalized PL")

ax1.set_ylim(0.7, 1.01)

ax1.grid(True, alpha=0.3)

# ============================================================
# BOTTOM: population dynamics
# ============================================================

labels = [
    "g0",
    "g+1",
    "g-1",
    "e0",
    "e+1",
    "e-1",
    "singlet"
]

lines = []

lines.append(ax2.plot(times, g_t[:, 0], lw=2)[0])
lines.append(ax2.plot(times, g_t[:, 1], '--', lw=2)[0])
lines.append(ax2.plot(times, g_t[:, 2], '--', lw=2)[0])

lines.append(ax2.plot(times, e_t[:, 0], lw=2)[0])
lines.append(ax2.plot(times, e_t[:, 1], '--', lw=2)[0])
lines.append(ax2.plot(times, e_t[:, 2], '--', lw=2)[0])

lines.append(ax2.plot(times, s_t, color='purple', lw=2)[0])

ax2.set_title("Population dynamics")

ax2.set_xlabel("Time (µs)")
ax2.set_ylabel("Population")

ax2.set_xlim(0, times[-1])
ax2.set_ylim(0, 1)

ax2.grid(True, alpha=0.3)

ax2.legend(labels, ncol=3)

# ============================================================
# Sliders
# ============================================================

ax_Bz = plt.axes([0.2, 0.08, 0.6, 0.03])

slider_Bz = Slider(
    ax=ax_Bz,
    label='Bz (G)',
    valmin=0,
    valmax=5,
    valinit=Bz0
)


ax_detuning = plt.axes([0.2, 0.03, 0.6, 0.03])

slider_detuning = Slider(
    ax=ax_detuning,
    label='MW detuning (MHz)',
    valmin=-10,
    valmax=10,
    valinit=0
)
# ============================================================
# Update
# ============================================================

def update(val):

    Bz = slider_Bz.val
    Delta = slider_detuning.val

    # ========================================================
    # Update ODMR
    # ========================================================

    PL = compute_odmr(freqs, Bz)

    line_odmr.set_ydata(PL)

    # ========================================================
    # Update time traces
    # ========================================================

    times, g_t, e_t, s_t, pl_t = evolve_time_trace(
        Delta=Delta,
        Bz=Bz,
        tmax=5
    )

    data = [
        g_t[:, 0],
        g_t[:, 1],
        g_t[:, 2],
        e_t[:, 0],
        e_t[:, 1],
        e_t[:, 2],
        s_t
    ]

    for line, d in zip(lines, data):
        line.set_data(times, d)

    ax2.set_xlim(0, times[-1])

    fig.canvas.draw_idle()

slider_Bz.on_changed(update)
slider_detuning.on_changed(update)

# Time-resolved fluorescence transient
# Compare m_s = 0 and m_s = ±1
# ============================================================

# total simulation time
t_total = 2.2  # us

times = np.arange(-0.2, t_total, dt)

# ============================================================
# Function to simulate transient
# ============================================================

def fluorescence_transient(initial_ground_state):

    g = initial_ground_state.copy()

    e = np.zeros(3)

    s = 0.0

    pl_trace = []

    for t in times:

        # laser OFF before t = 0
        laser_on = (t >= 0)

        # evolve populations
        g, e, s = optical_step(
            g,
            e,
            s,
            laser_on=laser_on
        )

        # instantaneous fluorescence
        signal = (
            gamma_rad_0 * e[0]
            + gamma_rad_pm * e[1]
            + gamma_rad_pm * e[2]
        )

        pl_trace.append(signal)

    return np.array(pl_trace)

# ============================================================
# Initial states
# ============================================================

# pure m_s = 0
g0_init = np.array([1.0, 0.0, 0.0])

# equal mixture of m_s = ±1
gpm_init = np.array([0.0, 0.5, 0.5])

# ============================================================
# Simulate
# ============================================================

pl_ms0 = fluorescence_transient(g0_init)

pl_mspm = fluorescence_transient(gpm_init)

# normalize together
norm = max(np.max(pl_ms0), np.max(pl_mspm))

pl_ms0 /= norm
pl_mspm /= norm

# ============================================================
# Plot
# ============================================================

fig5, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    times,
    pl_ms0,
    lw=3,
    label=r"$m_s = 0$"
)

ax.plot(
    times,
    pl_mspm,
    lw=3,
    label=r"$m_s = \pm1$"
)

# laser ON marker
ax.axvline(
    0,
    color='black',
    linestyle='--',
    alpha=0.7
)

ax.text(
    -0.14,
    0.65,
    "Laser\nON",
    fontsize=15
)

ax.set_xlabel("Time (µs)")
ax.set_ylabel("Luminescence intensity")

ax.set_title("Time-resolved luminescence")

ax.set_xlim(-0.2, 2.2)

ax.grid(True, alpha=0.3)

ax.legend()

plt.show()