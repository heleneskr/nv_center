import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -------- Constants --------
gamma_e = 2.8  # MHz/G

# Rabi frequency (drive strength)
Omega = 5.0  # MHz
tau_pi2 = np.pi / (2 * Omega)

# Time resolution
dt = 0.001
t_total = 1.5
t = np.arange(0, t_total, dt)

# Pauli matrices
sx = np.array([[0, 1],
               [1, 0]], dtype=complex)

sy = np.array([[0, -1j],
               [1j, 0]], dtype=complex)

sz = np.array([[1, 0],
               [0, -1]], dtype=complex)

I = np.eye(2)

# Initial state |0>
psi0 = np.array([1, 0], dtype=complex)


def evolve(psi, H, dt):
    U = np.eye(2) - 1j * H * dt
    return U @ psi


def simulate(Bz, detuning, free_time, theta):
    psi = psi0.copy()

    pop0 = []
    pop1 = []

    tau_free = free_time

    # Hamiltonians
    H_pulse1 = (Omega / 2) * sy
    H_free = (detuning / 2) * sz

    H_pulse2 = (Omega / 2) * (
        np.cos(theta) * sy - np.sin(theta) * sx
    )

    for ti in t:

        if ti < tau_pi2:
            H = H_pulse1

        elif ti < tau_pi2 + tau_free:
            H = H_free

        elif ti < tau_pi2 + tau_free + tau_pi2:
            H = H_pulse2

        else:
            H = np.zeros((2, 2))

        psi = evolve(psi, H, dt)
        psi = psi / np.linalg.norm(psi)

        pop0.append(np.abs(psi[0])**2)
        pop1.append(np.abs(psi[1])**2)

    return np.array(pop0), np.array(pop1)


# -------- Plot --------
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.3)

pop0, pop1 = simulate(0.03, 1.8, 0.8, 0)

l0, = ax.plot(t, pop0, label="|0⟩")
l1, = ax.plot(t, pop1, label="|+1⟩")

ax.set_title("Ramsey ODMR (matches analytic derivation)")
ax.set_xlabel("Time (µs)")
ax.set_ylabel("Population")
ax.legend()
ax.grid()

# -------- Sliders --------
ax_Bz = plt.axes([0.2, 0.18, 0.65, 0.03])
ax_det = plt.axes([0.2, 0.13, 0.65, 0.03])
ax_free = plt.axes([0.2, 0.08, 0.65, 0.03])
ax_theta = plt.axes([0.2, 0.03, 0.65, 0.03])

sBz = Slider(ax_Bz, "Bz (G)", 0, 1, valinit=0.03)
sDet = Slider(ax_det, "Detuning (MHz)", 0, 5, valinit=1.8)
sFree = Slider(ax_free, "Free time (µs)", 0, 1.2, valinit=0.8)
sTheta = Slider(ax_theta, "Pulse phase θ", 0, 2*np.pi, valinit=0)


def update(val):
    pop0, pop1 = simulate(
        sBz.val,
        sDet.val,
        sFree.val,
        sTheta.val
    )

    l0.set_ydata(pop0)
    l1.set_ydata(pop1)

    fig.canvas.draw_idle()


sBz.on_changed(update)
sDet.on_changed(update)
sFree.on_changed(update)
sTheta.on_changed(update)

plt.show()
