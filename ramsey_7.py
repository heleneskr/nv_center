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
# 7-level basis
#
# 0 = g0
# 1 = g+1
# 2 = g-1
# 3 = e0
# 4 = e+1
# 5 = e-1
# 6 = singlet
# ============================================================

N = 7

# ============================================================
# Parameters
# ============================================================

gamma_e = 2.8  # MHz/G

Omega = MHz_to_rad(0.3)

gamma_phi = 0.1  # us^-1

dt = 0.001

T_free0 = 1.0

t_pi2 = np.pi / (2 * Omega)

# ============================================================
# Optical rates
# ============================================================

laser_rate = 20.0

gamma_rad_0 = 66.16
gamma_rad_pm = 66.16

gamma_isc_0 = 11.1
gamma_isc_pm = 91.8

gamma_s0 = 4.87
gamma_spm = 2.04

t_init = 2.0
t_read = 1.0

# ============================================================
# MW Hamiltonian
# KEEPING YOUR VERSION
# ============================================================

def H_mw(Delta, Bz, phi=0):

    Delta_p = Delta - gamma_e * Bz

    H = np.zeros((N, N), dtype=complex)

    phase = np.exp(1j * phi)

    H[0,1] = (Omega/2) * np.conj(phase)
    H[1,0] = (Omega/2) * phase

    H[1,1] = MHz_to_rad(Delta_p)

    return H

# ============================================================
# Free evolution Hamiltonian
# ============================================================

def H_free(Delta, Bz):

    Delta_p = Delta - gamma_e * Bz

    H = np.zeros((N, N), dtype=complex)

    H[1,1] = MHz_to_rad(Delta_p)

    return H

# ============================================================
# Ground-state dephasing
# ============================================================

def free_evolution_step(rho_g):

    rho_new = rho_g.copy()

    decay = np.exp(-gamma_phi * dt)

    for i in range(3):
        for j in range(3):

            if i != j:
                rho_new[i,j] *= decay

    return rho_new

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
        exc1 = laser_rate * g[1] * dt
        exc2 = laser_rate * g[2] * dt

        g[0] -= exc0
        g[1] -= exc1
        g[2] -= exc2

        e[0] += exc0
        e[1] += exc1
        e[2] += exc2

    # ========================================================
    # Radiative decay
    # ========================================================

    rad0 = gamma_rad_0 * e[0] * dt
    rad1 = gamma_rad_pm * e[1] * dt
    rad2 = gamma_rad_pm * e[2] * dt

    e[0] -= rad0
    e[1] -= rad1
    e[2] -= rad2

    g[0] += rad0
    g[1] += rad1
    g[2] += rad2

    # ========================================================
    # ISC
    # ========================================================

    isc0 = gamma_isc_0 * e[0] * dt

    isc1 = gamma_isc_pm * e[1] * dt
    isc2 = gamma_isc_pm * e[2] * dt

    e[0] -= isc0
    e[1] -= isc1
    e[2] -= isc2

    s += isc0 + isc1 + isc2

    # ========================================================
    # Singlet decay
    # ========================================================

    ds0 = gamma_s0 * s * dt

    dsp_each = 0.5 * gamma_spm * s * dt

    total_ds = ds0 + 2*dsp_each

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
# Time trace simulation
# ============================================================

def simulate_time_trace(Delta, Bz, T_free, phi2):

    total_time = t_init + 2*t_pi2 + T_free + t_read

    times = np.arange(0, total_time, dt)

    # ========================================================
    # Ground-state coherent density matrix
    # ========================================================

    rho_g = np.zeros((3,3), dtype=complex)

    rho_g[0,0] = 1/3
    rho_g[1,1] = 1/3
    rho_g[2,2] = 1/3

    # excited populations

    e = np.zeros(3)

    # singlet

    s = 0.0

    populations = []

    # ========================================================
    # Propagators
    # ========================================================

    U1 = expm(
        -1j * H_mw(Delta, Bz, phi=0)[0:3,0:3] * dt
    )

    U2 = expm(
        -1j * H_mw(Delta, Bz, phi=phi2)[0:3,0:3] * dt
    )

    Ufree = expm(
        -1j * H_free(Delta, Bz)[0:3,0:3] * dt
    )

    # ========================================================
    # Main loop
    # ========================================================

    for t in times:

        # save populations

        g = np.real(np.diag(rho_g))

        p = np.zeros(N)

        p[0:3] = g
        p[3:6] = e
        p[6] = s

        populations.append(p)

        # ====================================================
        # Initialization laser
        # ====================================================

        if t < t_init:

            g = np.real(np.diag(rho_g)).copy()

            g, e, s = optical_step(
                g,
                e,
                s,
                laser_on=True
            )

            rho_g = np.diag(g)

        # ====================================================
        # First pi/2 pulse
        # ====================================================

        elif t < t_init + t_pi2:

            rho_g = U1 @ rho_g @ U1.conj().T

            rho_g = free_evolution_step(rho_g)

            g = np.real(np.diag(rho_g)).copy()

            g, e, s = optical_step(
                g,
                e,
                s,
                laser_on=False
            )

            rho_g[0,0] = g[0]
            rho_g[1,1] = g[1]
            rho_g[2,2] = g[2]

        # ====================================================
        # Free evolution
        # ====================================================

        elif t < t_init + t_pi2 + T_free:

            rho_g = Ufree @ rho_g @ Ufree.conj().T

            rho_g = free_evolution_step(rho_g)

            g = np.real(np.diag(rho_g)).copy()

            g, e, s = optical_step(
                g,
                e,
                s,
                laser_on=False
            )

            rho_g[0,0] = g[0]
            rho_g[1,1] = g[1]
            rho_g[2,2] = g[2]

        # ====================================================
        # Second pi/2 pulse
        # ====================================================

        elif t < t_init + 2*t_pi2 + T_free:

            rho_g = U2 @ rho_g @ U2.conj().T

            rho_g = free_evolution_step(rho_g)

            g = np.real(np.diag(rho_g)).copy()

            g, e, s = optical_step(
                g,
                e,
                s,
                laser_on=False
            )

            rho_g[0,0] = g[0]
            rho_g[1,1] = g[1]
            rho_g[2,2] = g[2]

        # ====================================================
        # Readout laser
        # ====================================================

        else:

            g = np.real(np.diag(rho_g)).copy()

            g, e, s = optical_step(
                g,
                e,
                s,
                laser_on=True
            )

            rho_g = np.diag(g)

    return times, np.array(populations)

# ============================================================
# Ramsey scan
# ============================================================

# ============================================================
# Ramsey scan
# FIXED VERSION
# ============================================================

def compute_ramsey(Delta, Bz, phi2):

    taus = np.linspace(0, 10, 400)

    signal = []

    for tau in taus:

        # ====================================================
        # Run full sequence
        # ====================================================

        _, populations = simulate_time_trace(
            Delta,
            Bz,
            tau,
            phi2
        )

        # ====================================================
        # Readout window
        # ====================================================

        read_start = int(
            (t_init + 2*t_pi2 + tau) / dt
        )

        read_end = int(
            (t_init + 2*t_pi2 + tau + 0.25) / dt
        )

        read_pop = populations[read_start:read_end]

        # ====================================================
        # Instantaneous fluorescence
        # ====================================================

        fluorescence = (
            gamma_rad_0  * read_pop[:,3] +
            gamma_rad_pm * read_pop[:,4] +
            gamma_rad_pm * read_pop[:,5]
        )

        # ====================================================
        # Integrate early photons only
        # (this creates Ramsey contrast)
        # ====================================================

        signal.append(np.sum(fluorescence) * dt)

    signal = np.array(signal)

    # ========================================================
    # Convert to PL contrast centered at zero
    # ========================================================

    mean_signal = np.mean(signal)

    contrast = (signal - mean_signal) / mean_signal

    return taus, contrast

# ============================================================
# PL contrast vs magnetic field
# at fixed free precession time
# ============================================================

# ============================================================
# Ramsey contrast vs magnetic field
# ============================================================

def contrast_vs_B(T_free, Delta=2.0, phi2=0):

    # magnetic field in mT
    B_fields_mT = np.linspace(
        -0.01,
         0.01,
         400
    )

    contrast = []

    for BmT in B_fields_mT:

        # convert mT -> G
        Bz = 10 * BmT

        # ====================================================
        # Run sequence
        # ====================================================

        _, populations = simulate_time_trace(
            Delta,
            Bz,
            T_free,
            phi2
        )

        # ====================================================
        # Early readout window only
        # ====================================================

        read_start = int(
            (t_init + 2*t_pi2 + T_free) / dt
        )

        read_end = int(
            (t_init + 2*t_pi2 + T_free + 0.12) / dt
        )

        read_pop = populations[
            read_start:read_end
        ]

        # ====================================================
        # Fluorescence
        # ====================================================

        fluorescence = (
            gamma_rad_0  * read_pop[:,3]
            + gamma_rad_pm * read_pop[:,4]
            + gamma_rad_pm * read_pop[:,5]
        )

        signal = np.sum(fluorescence) * dt

        contrast.append(signal)

    contrast = np.array(contrast)

    # ========================================================
    # Normalize between 0 and 1
    # ========================================================

    contrast = (
        contrast - np.min(contrast)
    ) / (
        np.max(contrast) - np.min(contrast)
    )

    return B_fields_mT, contrast

# ============================================================
# Plot styling
# ============================================================

plt.style.use("default")

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2"
]

# ============================================================
# Figure
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(12,8)
)

plt.subplots_adjust(bottom=0.28, hspace=0.4)

Delta0 = 1.0
Bz0 = 0.5
phi20 = 0.0

times, populations = simulate_time_trace(
    Delta0,
    Bz0,
    T_free0,
    phi20
)

# ============================================================
# Population plot
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

for i in range(N):

    line, = ax1.plot(
        times,
        populations[:,i],
        lw=2,
        color=colors[i],
        label=labels[i]
    )

    lines.append(line)

ax1.set_title("7-Level NV Ramsey Sequence")

ax1.set_xlabel("Time (µs)")
ax1.set_ylabel("Population")

ax1.set_xlim(0, times[-1])
ax1.set_ylim(0, 1)

ax1.grid(alpha=0.3)

ax1.legend(ncol=4)

# ============================================================
# Pulse regions
# ============================================================

init_patch = ax1.axvspan(
    0,
    t_init,
    color='green',
    alpha=0.15
)

mw1_patch = ax1.axvspan(
    t_init,
    t_init + t_pi2,
    color='gray',
    alpha=0.2
)

mw2_patch = ax1.axvspan(
    t_init + t_pi2 + T_free0,
    t_init + 2*t_pi2 + T_free0,
    color='gray',
    alpha=0.2
)

read_patch = ax1.axvspan(
    t_init + 2*t_pi2 + T_free0,
    t_init + 2*t_pi2 + T_free0 + t_read,
    color='red',
    alpha=0.15
)

# ============================================================
# Ramsey plot
# ============================================================

taus, signal = compute_ramsey(
    Delta0,
    Bz0,
    phi20
)

ramsey_line, = ax2.plot(
    taus,
    signal,
    lw=3,
    color='red'
)

ax2.set_title("Ramsey PL Contrast")

ax2.set_xlabel(r"Free precession time $\tau$ (µs)")
ax2.set_ylabel("PL Contrast")

ax2.grid(alpha=0.3)

# center around zero like real Ramsey plots
ylim = 1.1 * np.max(np.abs(signal))

ax2.set_ylim(-ylim, ylim)

ax2.axhline(
    0,
    color='black',
    lw=1,
    alpha=0.5
)

freq_text = ax2.text(
    0.02,
    0.92,
    f"Δ = {Delta0:.2f} MHz",
    transform=ax2.transAxes,
    bbox=dict(facecolor='white', alpha=0.8)
)


# ============================================================
# Sliders
# ============================================================

ax_Delta = plt.axes([0.18,0.18,0.65,0.03])

slider_Delta = Slider(
    ax=ax_Delta,
    label='Detuning Δ (MHz)',
    valmin=0,
    valmax=10,
    valinit=Delta0
)

ax_Bz = plt.axes([0.18,0.12,0.65,0.03])

slider_Bz = Slider(
    ax=ax_Bz,
    label='Bz (G)',
    valmin=0,
    valmax=10,
    valinit=Bz0
)

ax_phi = plt.axes([0.18,0.06,0.65,0.03])

slider_phi = Slider(
    ax=ax_phi,
    label='Second pulse phase',
    valmin=0,
    valmax=2*np.pi,
    valinit=phi20
)

# ============================================================
# Update
# ============================================================

def update(val):

    global mw2_patch
    global read_patch

    Delta = slider_Delta.val
    Bz = slider_Bz.val
    phi2 = slider_phi.val

    # ========================================================
    # populations
    # ========================================================

    times, populations = simulate_time_trace(
        Delta,
        Bz,
        T_free0,
        phi2
    )

    for i in range(N):

        lines[i].set_data(
            times,
            populations[:,i]
        )

    # ========================================================
    # Ramsey
    # ========================================================

    taus, signal = compute_ramsey(
        Delta,
        Bz,
        phi2
    )

    ramsey_line.set_data(
        taus,
        signal
    )

    ylim = 1.1 * np.max(np.abs(signal))

    ax2.set_ylim(-ylim, ylim)

    # ========================================================
    # update pulse regions
    # ========================================================

    mw2_patch.remove()
    read_patch.remove()

    mw2_patch = ax1.axvspan(
        t_init + t_pi2 + T_free0,
        t_init + 2*t_pi2 + T_free0,
        color='gray',
        alpha=0.2
    )

    read_patch = ax1.axvspan(
        t_init + 2*t_pi2 + T_free0,
        t_init + 2*t_pi2 + T_free0 + t_read,
        color='red',
        alpha=0.15
    )

    ax1.set_xlim(0, times[-1])

    freq_text.set_text(
        f"Δ = {Delta:.2f} MHz"
    )

    fig.canvas.draw_idle()

# ============================================================
# Connect sliders
# ============================================================

slider_Delta.on_changed(update)
slider_Bz.on_changed(update)
slider_phi.on_changed(update)

# ============================================================
# Show
# ============================================================
# ============================================================
# FIXED-TAU MAGNETIC FIELD SCAN
# ============================================================

tau_fixed = 3.0  # us

Bvals = np.linspace(-0.1, 0.1, 500)  # Gauss

Delta = 1.0      # MHz
T2star = 6.0     # us

contrast = []

for Bz in Bvals:

    # effective detuning
    delta_eff = Delta - gamma_e * Bz

    # Ramsey phase
    phase = 2 * np.pi * delta_eff * tau_fixed

    # Ramsey envelope
    envelope = np.exp(-(tau_fixed / T2star)**2)

    # Ramsey contrast
    C = 0.5 * (
        1 + envelope * np.cos(phase)
    )

    contrast.append(C)

contrast = np.array(contrast)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(8,6))

plt.plot(
    Bvals,
    contrast,
    lw=4
)

plt.xlabel("B (G)", fontsize=18)
plt.ylabel("Normalized contrast", fontsize=18)


plt.grid(alpha=0.3)

plt.ylim(0, 1.05)

plt.tight_layout()

plt.show()