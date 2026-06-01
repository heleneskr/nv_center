import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.linalg import expm
from scipy.optimize import curve_fit

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

# dephasing rate
gamma_phi = 2  # us^-1

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
# Pulse durations
# ============================================================

t_laser = 2.0
t_pause = 1.0      # NEW: dark time between init and MW
t_mw0 = 1

# ============================================================
# Hamiltonian
# ============================================================

def H_mw(Delta, Bz):

    Delta_p = Delta - gamma_e * Bz
    Delta_m = Delta + gamma_e * Bz

    Hmw = np.array([
        [0,                     Omega/2,               Omega/2],
        [Omega/2,  MHz_to_rad(Delta_p),                      0],
        [Omega/2,                     0,  MHz_to_rad(Delta_m)]
    ], dtype=complex)

    return Hmw

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
# Free evolution (NO laser, NO MW)
# ============================================================

def free_evolution_step(rho_g):

    # only dephasing during dark time

    decay = np.exp(-gamma_phi * dt)

    for i in range(3):
        for j in range(3):

            if i != j:
                rho_g[i, j] *= decay

    return rho_g

# ============================================================
# Pulsed ODMR sequence
# ============================================================

def pulsed_odmr_sequence(Delta, Bz, t_mw):

    rho_g = np.zeros((3, 3), dtype=complex)
    rho_g[0, 0] = 1.0

    e = np.zeros(3)
    s = 0.0

    times = []

    g_trace = []
    e_trace = []
    s_trace = []

    pl_trace = []

    t = 0

    # ========================================================
    # 1) Laser initialization
    # ========================================================

    steps = int(t_laser / dt)

    for _ in range(steps):

        g = np.real(np.diag(rho_g)).copy()

        g, e, s = optical_step(g, e, s, laser_on=True)

        for i in range(3):
            rho_g[i, i] = g[i]

        times.append(t)

        g_trace.append(g.copy())
        e_trace.append(e.copy())
        s_trace.append(s)

        pl_trace.append(np.sum(e))

        t += dt

    # ========================================================
    # 2) DARK PAUSE
    # ========================================================

    steps = int(t_pause / dt)

    for _ in range(steps):

        rho_g = free_evolution_step(rho_g)

        g = np.real(np.diag(rho_g)).copy()

        # no laser
        g, e, s = optical_step(g, e, s, laser_on=False)

        for i in range(3):
            rho_g[i, i] = g[i]

        times.append(t)

        g_trace.append(g.copy())
        e_trace.append(e.copy())
        s_trace.append(s)

        pl_trace.append(np.sum(e))

        t += dt

    # ========================================================
    # 3) Microwave pulse
    # ========================================================

    steps = int(t_mw / dt)

    U = expm(-1j * H_mw(Delta, Bz) * dt)

    for _ in range(steps):

        rho_g = U @ rho_g @ U.conj().T

        # dephasing
        decay = np.exp(-gamma_phi * dt)

        for i in range(3):
            for j in range(3):

                if i != j:
                    rho_g[i, j] *= decay

        g = np.real(np.diag(rho_g)).copy()

        # no laser during MW pulse
        g, e, s = optical_step(g, e, s, laser_on=False)

        for i in range(3):
            rho_g[i, i] = g[i]

        times.append(t)

        g_trace.append(g.copy())
        e_trace.append(e.copy())
        s_trace.append(s)

        pl_trace.append(np.sum(e))

        t += dt

    # ========================================================
    # 4) Readout laser pulse
    # ========================================================

    steps = int(t_laser / dt)

    readout_signal = 0

    for _ in range(steps):

        g = np.real(np.diag(rho_g)).copy()

        g, e, s = optical_step(g, e, s, laser_on=True)

        for i in range(3):
            rho_g[i, i] = g[i]

        signal = (
            gamma_rad_0 * e[0]
            + gamma_rad_pm * e[1]
            + gamma_rad_pm * e[2]
        )

        readout_signal += signal * dt

        times.append(t)

        g_trace.append(g.copy())
        e_trace.append(e.copy())
        s_trace.append(s)

        pl_trace.append(signal)

        t += dt

    return (
        readout_signal,
        np.array(times),
        np.array(g_trace),
        np.array(e_trace),
        np.array(s_trace),
        np.array(pl_trace)
    )

# ============================================================
# ODMR spectrum
# ============================================================

def compute_pulsed_odmr(freqs, Bz, t_mw):

    PL = []

    for f in freqs:

        Delta = f - D0

        signal, *_ = pulsed_odmr_sequence(
            Delta=Delta,
            Bz=Bz,
            t_mw=t_mw
        )

        PL.append(signal)

    PL = np.array(PL)

    PL /= np.max(PL)

    return PL

# ============================================================
# Setup
# ============================================================

D0 = 2870  # MHz

freqs = np.linspace(D0 - 10, D0 + 10, 1000)

Bz0 = 0

PL = compute_pulsed_odmr(freqs, Bz0, t_mw0)

signal, times, g_t, e_t, s_t, pl_t = pulsed_odmr_sequence(
    Delta=0,
    Bz=Bz0,
    t_mw=t_mw0
)

# ============================================================
# Figure
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(11, 8),
    gridspec_kw={'height_ratios': [1, 2]}
)

plt.subplots_adjust(bottom=0.22)

# ============================================================
# ODMR plot
# ============================================================

line_odmr, = ax1.plot(freqs, PL, color='black', lw=2)

ax1.set_title("Pulsed ODMR")
ax1.set_ylabel("Normalized PL")

ax1.set_ylim(0.7, 1.01)

ax1.grid(True, alpha=0.3)

# ============================================================
# Population dynamics
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

# ============================================================
# Pulse regions
# ============================================================

init_patch = ax2.axvspan(
    0,
    t_laser,
    color='green',
    alpha=0.15
)

pause_patch = ax2.axvspan(
    t_laser,
    t_laser + t_pause,
    color='blue',
    alpha=0.08
)

mw_patch = ax2.axvspan(
    t_laser + t_pause,
    t_laser + t_pause + t_mw0,
    color='gray',
    alpha=0.2
)

read_patch = ax2.axvspan(
    t_laser + t_pause + t_mw0,
    2 * t_laser + t_pause + t_mw0,
    color='red',
    alpha=0.15
)

# ============================================================
# Labels
# ============================================================

ax2.text(
    0.5 * t_laser,
    0.95,
    "Init",
    ha='center'
)

ax2.text(
    t_laser + 0.5 * t_pause,
    0.95,
    "pause",
    ha='center'
)

mw_text = ax2.text(
    t_laser + t_pause + 0.5 * t_mw0,
    0.95,
    "MW",
    ha='center'
)

read_text = ax2.text(
    t_laser + t_pause + t_mw0 + 0.5 * t_laser,
    0.95,
    "Readout",
    ha='center'
)

ax2.set_xlabel("Time (µs)")
ax2.set_ylabel("Population")

ax2.set_xlim(0, times[-1])
ax2.set_ylim(0, 1)

ax2.legend(labels, ncol=3)

ax2.grid(True, alpha=0.3)

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

ax_tmw = plt.axes([0.2, 0.03, 0.6, 0.03])

slider_tmw = Slider(
    ax=ax_tmw,
    label='MW pulse (µs)',
    valmin=0.1,
    valmax=5.0,
    valinit=t_mw0
)

# ============================================================
# Update
# ============================================================

def update(val):

    Bz = slider_Bz.val
    t_mw = slider_tmw.val

    # ========================================================
    # ODMR spectrum
    # ========================================================

    PL = compute_pulsed_odmr(freqs, Bz, t_mw)

    line_odmr.set_ydata(PL)

    # ========================================================
    # Population dynamics
    # ========================================================

    _, times, g_t, e_t, s_t, _ = pulsed_odmr_sequence(
        Delta=Bz * gamma_e,
        Bz=Bz,
        t_mw=t_mw
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

    # ========================================================
    # REMOVE OLD PATCHES
    # ========================================================

    global init_patch
    global pause_patch
    global mw_patch
    global read_patch

    init_patch.remove()
    pause_patch.remove()
    mw_patch.remove()
    read_patch.remove()

    # ========================================================
    # CREATE NEW PATCHES
    # ========================================================

    init_patch = ax2.axvspan(
        0,
        t_laser,
        color='green',
        alpha=0.15
    )

    pause_patch = ax2.axvspan(
        t_laser,
        t_laser + t_pause,
        color='blue',
        alpha=0.08
    )

    mw_patch = ax2.axvspan(
        t_laser + t_pause,
        t_laser + t_pause + t_mw,
        color='gray',
        alpha=0.2
    )

    read_patch = ax2.axvspan(
        t_laser + t_pause + t_mw,
        2 * t_laser + t_pause + t_mw,
        color='red',
        alpha=0.15
    )

    # ========================================================
    # UPDATE LABEL POSITIONS
    # ========================================================


    mw_text.set_position((
        t_laser + t_pause + 0.5 * t_mw,
        0.95
    ))

    read_text.set_position((
        t_laser + t_pause + t_mw + 0.5 * t_laser,
        0.95
    ))

    # ========================================================
    # AXES
    # ========================================================

    ax2.set_xlim(0, times[-1])

    fig.canvas.draw_idle()

slider_Bz.on_changed(update)
slider_tmw.on_changed(update)

def lorentzian_dip(f, baseline, depth, f0, gamma):

    return baseline - depth * (
        gamma**2 /
        ((f - f0)**2 + gamma**2)
    )
def fit_linewidth(freqs, PL):

    idx = np.argmin(PL)

    p0 = [
        np.max(PL),                    # baseline
        np.max(PL) - np.min(PL),       # depth
        freqs[idx],                    # center
        0.5                            # HWHM guess
    ]

    try:

        popt, _ = curve_fit(
            lorentzian_dip,
            freqs,
            PL,
            p0=p0
        )

        baseline, depth, f0, gamma = popt

        FWHM = 2 * abs(gamma)

        return FWHM, popt

    except RuntimeError:

        return np.nan, None
    
# ============================================================
# Linewidth vs MW pulse duration
# ============================================================

t_mw_values = np.linspace(0.1, 5.0, 25)

linewidth_values = []

freqs_fit = np.linspace(
    D0 - 5,
    D0 + 5,
    400
)

for t_mw_test in t_mw_values:

    PL = compute_pulsed_odmr(
        freqs_fit,
        Bz0,
        t_mw_test
    )

    linewidth, _ = fit_linewidth(
        freqs_fit,
        PL
    )

    linewidth_values.append(linewidth)

linewidth_values = np.array(linewidth_values)

fig6, ax = plt.subplots(figsize=(7,4))

ax.plot(
    t_mw_values,
    linewidth_values,
    'o-',
    lw=2
)

ax.set_xlabel("MW pulse duration (µs)")
ax.set_ylabel("Lorentzian FWHM (MHz)")
ax.set_title("Pulsed ODMR linewidth vs MW pulse duration")

ax.grid(True, alpha=0.3)

t_test = 1.0

PL = compute_pulsed_odmr(
    freqs_fit,
    Bz0,
    t_test
)

linewidth, popt = fit_linewidth(
    freqs_fit,
    PL
)

plt.figure(figsize=(7,4))

plt.plot(freqs_fit, PL, label="Simulation")

plt.plot(
    freqs_fit,
    lorentzian_dip(freqs_fit, *popt),
    '--',
    label=f"Fit (FWHM={linewidth:.3f} MHz)"
)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Normalized PL")
plt.legend()
plt.grid(True)

plt.show()
