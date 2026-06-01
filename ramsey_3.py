import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.linalg import expm

import nv_config as cfg

gamma_e = cfg.gamma_e
D = cfg.D
two_pi = cfg.two_pi
Omega = cfg.Omega

# ============================================================
# Parameters
# ============================================================

T_free0 = 1.0
t_pi2 = 1/(4*np.sqrt(2)*Omega)

dt = 0.0001  # FIX: constant timestep (important)
nsteps_max = 20000

# ============================================================
# Microwave pulse control
# ============================================================

def mw_on(t, T_free):

    # first π/2
    if 0 <= t < t_pi2:
        return True

    # second π/2
    if t_pi2 + T_free <= t < 2*t_pi2 + T_free:
        return True

    return False


def mw_phase(t, T_free):

    # first pulse (X)
    if 0 <= t < t_pi2:
        return 0

    # second pulse (Y)
    if t_pi2 + T_free <= t < 2*t_pi2 + T_free:
        return np.pi/2

    return 0


# ============================================================
# Hamiltonian
# ============================================================

# ============================================================
# Hamiltonians used in the analytic derivation
# ============================================================

Omega_eff = np.sqrt(2) * two_pi * Omega

# π/2 pulse duration
t_pi2 = np.pi/(2*np.sqrt(2)*Omega_eff)


def H1():

    return (Omega_eff/2) * np.array([
        [0,1,0],
        [1,0,1],
        [0,1,0]
    ], dtype=complex)


def H2(phi=0):

    ep = np.exp(1j*phi)

    return (Omega_eff/2) * np.array([
        [0, ep, 0],
        [np.conj(ep), 0, np.conj(ep)],
        [0, ep, 0]
    ], dtype=complex)


def H_free(Bz, detuning):

    Delta_m = 2*np.pi*(-gamma_e*Bz - detuning)
    Delta_p = 2*np.pi*( gamma_e*Bz - detuning)

    return np.array([
        [Delta_m,0,0],
        [0,0,0],
        [0,0,Delta_p]
    ], dtype=complex)

# ============================================================
# Simulation
# ============================================================

def simulate(Bz, detuning, T_free):

    phi = 0

    T_total = 2*t_pi2 + T_free

    t = np.arange(0, T_total + dt, dt)

    psi = np.array([0,1,0], dtype=complex)

    P = np.zeros((len(t),3))

    for i, ti in enumerate(t[:-1]):

        # store current population
        P[i] = np.abs(psi)**2

        # first π/2 pulse
        if ti <= t_pi2:

            H = H1()

        # free evolution
        elif ti <= t_pi2 + T_free:

            H = H_free(Bz, detuning)

        # second π/2 pulse
        else:

            H = H2(phi)

        psi = expm(-1j * H * dt) @ psi
        psi /= np.linalg.norm(psi)

    # store final population AFTER last propagation step
    P[-1] = np.abs(psi)**2

    return t, P, psi

def ramsey_scan(Bz, detuning):

    taus = np.linspace(0, 5, 500)

    signal = []

    for tau in taus:

        _, _, psi_final = simulate(
            Bz,
            detuning,
            tau
        )

        signal.append(
            np.abs(psi_final[1])**2
        )

    return taus, np.array(signal)

# ============================================================
# Initial run
# ============================================================

t, P, _ = simulate(
    0.0,
    0.0,
    T_free0
)

fig, (ax_pop, ax_ramsey) = plt.subplots(
    2,
    1,
    figsize=(10,8)
)

plt.subplots_adjust(bottom=0.32, hspace=0.4)


lines = [ax_pop.plot(t, P[:,i], label=l)[0]
         for i,l in enumerate(["|-1⟩","|0⟩","|+1⟩"])]

ax_pop.set_xlabel("Time (µs)")
ax_pop.set_ylabel("Population")
ax_pop.grid(alpha=0.3)
ax_pop.legend()

tau_scan = np.linspace(0, 5, 500)

phi = np.pi/2

Delta_p = gamma_e*0.0 - 0.0
Delta_m = -gamma_e*0.0 - 0.0

theta_p = 2*np.pi*Delta_p*tau_scan
theta_m = 2*np.pi*Delta_m*tau_scan

P_theory = 0.25 * (
    1
    + np.cos((theta_p-theta_m)/2)**2
    - 2*np.cos((theta_p-theta_m)/2)
      * np.cos(
            phi
            + (theta_p+theta_m)/2
        )
)
theory_line, = ax_ramsey.plot(
    tau_scan,
    P_theory,
    '--k',
    lw=2,
    label='cos² model',
    zorder=10
)



ax_ramsey.legend()

pulse1 = None
pulse2 = None


# ============================================================
# Draw pulses
# ============================================================

def draw_pulses(T_free):
    global pulse1, pulse2

    if pulse1 is not None:
        pulse1.remove()
    if pulse2 is not None:
        pulse2.remove()

    pulse1 = ax_pop.axvspan(0, t_pi2, color="lightgray", alpha=0.3)

    start = t_pi2 + T_free
    pulse2 = ax_pop.axvspan(start, start + t_pi2,
                        color="lightgray", alpha=0.3)


draw_pulses(T_free0)
taus, signal = ramsey_scan(
    0.0,
    0.0
)

ramsey_line, = ax_ramsey.plot(
    taus,
    signal,
    color="red",
    lw=2,
    zorder=5)

ax_ramsey.set_title("Ramsey fringes")
ax_ramsey.set_xlabel(
    r"Free precession time $\tau$ ($\mu$s)"
)
ax_ramsey.set_ylabel(
    r"$P_{m_s=0}$"
)
ax_ramsey.grid(alpha=0.3)

# ============================================================
# Sliders
# ============================================================

ax_Bz = plt.axes([0.15,0.18,0.7,0.03])
ax_det = plt.axes([0.15,0.12,0.7,0.03])
ax_free = plt.axes([0.15,0.06,0.7,0.03])

Bz_slider = Slider(ax_Bz,"Bz (G)",
                   cfg.Bz_min, cfg.Bz_max, valinit=0.0)

det_slider = Slider(ax_det,"Detuning (MHz)",
                    cfg.detuning_min, cfg.detuning_max, valinit=0.0)

free_slider = Slider(ax_free,"Free time (µs)",
                     0.0, 5.0, valinit=T_free0)


# ============================================================
# Update
# ============================================================

def update(val):

    Bz = Bz_slider.val
    det = det_slider.val
    T_free = free_slider.val

    t, P, _ = simulate(
    Bz,
    det,
    T_free
)

    for i in range(3):
        lines[i].set_xdata(t)
        lines[i].set_ydata(P[:,i])

    ax_pop.set_xlim(0, t[-1])
    taus, signal = ramsey_scan(
    Bz,
    det
        )

    ramsey_line.set_data(
        taus,
        signal
        )

    ax_ramsey.set_xlim(
        taus[0],
        taus[-1]
        )

    ax_ramsey.set_ylim(
        np.min(signal)-0.05,
        np.max(signal)+0.05
        )
    tau_scan = np.linspace(0, 5, 500)

    phi = 0

    Delta_p = gamma_e*Bz - det
    Delta_m = -gamma_e*Bz - det

    theta_p = 2*np.pi*Delta_p*tau_scan
    theta_m = 2*np.pi*Delta_m*tau_scan

    P_theory = 0.25 * (
    1
    + np.cos((theta_p-theta_m)/2)**2
    - 2*np.cos((theta_p-theta_m)/2)
      * np.cos(
            phi
            + (theta_p+theta_m)/2
        )
    )
    theory_line.set_ydata(P_theory)
    draw_pulses(T_free)

    fig.canvas.draw_idle()


Bz_slider.on_changed(update)
det_slider.on_changed(update)
free_slider.on_changed(update)



plt.show()