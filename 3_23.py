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
    return two_pi * (D * (Sz @ Sz) + gamma_e * Bz * Sz)


def H_mw(ti, f_mw):
    return two_pi * Omega * np.cos(two_pi * f_mw * ti) * Sx


def H_RWA_3level(Bz, detuning):

    w_m1 = D - gamma_e * Bz
    w_p1 = D + gamma_e * Bz

    f_mw = w_m1 - detuning

    H = np.zeros((3,3), dtype=complex)

    H[0,0] = two_pi * (w_m1 - f_mw)
    H[2,2] = two_pi * (w_p1 - f_mw)

    g = two_pi * Omega / (2*np.sqrt(2))

    H[0,1] = g
    H[1,0] = g
    H[1,2] = g
    H[2,1] = g

    return H

# ============================================================
# Simulations
# ============================================================

def simulate_lab(Bz, detuning):

    omega0 = D - gamma_e * Bz
    f_mw = omega0 - detuning

    psi = np.array([0,1,0], dtype=complex)
    P = np.zeros((cfg.nsteps,3))

    for i,ti in enumerate(t):

        P[i] = np.abs(psi)**2

        if i < cfg.nsteps-1:

            H = H_static(Bz) + H_mw(ti,f_mw)

            psi = expm(-1j*H*dt) @ psi
            psi /= np.linalg.norm(psi)

    return P


def simulate_rwa(Bz, detuning):

    psi = np.array([0,1,0], dtype=complex)
    P = np.zeros((cfg.nsteps,3))

    H = H_RWA_3level(Bz,detuning)
    U = expm(-1j*H*dt)

    for i in range(cfg.nsteps):

        P[i] = np.abs(psi)**2

        if i < cfg.nsteps-1:
            psi = U @ psi

    return P

# ============================================================
# π/2 detection
# ============================================================

def find_pi2_time(P):

    P0 = P[:,1]

    for i in range(1,len(P0)):
        if P0[i] <= 0.5:
            return t[i], i

    return t[-1], len(t)-1

# ============================================================
# Initial populations
# ============================================================

P_lab = simulate_lab(0.0,0.0)
P_rwa = simulate_rwa(0.0,0.0)

t_pi2_lab, idx_pi2_lab = find_pi2_time(P_lab)
t_pi2_rwa, idx_pi2_rwa = find_pi2_time(P_rwa)

# ============================================================
# Plot
# ============================================================

fig,axs = plt.subplots(1,2,figsize=(14,5),sharey=True)
plt.subplots_adjust(bottom=0.28)

# ---- Lab ----

lines_lab = [axs[0].plot(t,P_lab[:,i])[0] for i in range(3)]
#vline_lab = axs[0].axvline(t_pi2_lab,ls='--',lw=2)

#dot_lab_m1, = axs[0].plot([],[],'o',color='red')
#dot_lab_p1, = axs[0].plot([],[],'o',color='purple')

#label_lab_m1 = axs[0].text(0,0,'')
#label_lab_p1 = axs[0].text(0,0,'')

axs[0].set_title("Lab frame")
axs[0].set_xlabel("Time (µs)")
axs[0].set_ylabel("Population")
axs[0].grid(alpha=0.3)

# ---- RWA ----

lines_rwa = [axs[1].plot(t,P_rwa[:,i])[0] for i in range(3)]
#vline_rwa = axs[1].axvline(t_pi2_rwa,ls='--',lw=2)

#dot_rwa_m1, = axs[1].plot([],[],'o',color='red')
#dot_rwa_p1, = axs[1].plot([],[],'o',color='purple')

#label_rwa_m1 = axs[1].text(0,0,'')
#label_rwa_p1 = axs[1].text(0,0,'')

axs[1].set_title("3-level RWA")
axs[1].set_xlabel("Time (µs)")
axs[1].grid(alpha=0.3)

# ============================================================
# Sliders
# ============================================================

ax_Bz = plt.axes([0.15,0.15,0.7,0.03])
ax_det = plt.axes([0.15,0.10,0.7,0.03])

Bz_slider = Slider(ax_Bz,"Bz (G)",
                   cfg.Bz_min,cfg.Bz_max,valinit=0.0)

det_slider = Slider(ax_det,"Detuning Δ (MHz)",
                    cfg.detuning_min,cfg.detuning_max,valinit=0.0)

# ============================================================
# Update
# ============================================================

def update(val):

    Bz = Bz_slider.val
    det = det_slider.val

    P_lab = simulate_lab(Bz,det)
    P_rwa = simulate_rwa(Bz,det)

    for i in range(3):
        lines_lab[i].set_ydata(P_lab[:,i])
        lines_rwa[i].set_ydata(P_rwa[:,i])

    t_pi2_lab, idx_lab = find_pi2_time(P_lab)
    t_pi2_rwa, idx_rwa = find_pi2_time(P_rwa)

    #vline_lab.set_xdata([t_pi2_lab,t_pi2_lab])
    #vline_rwa.set_xdata([t_pi2_rwa,t_pi2_rwa])

    label_offset = 0.03

    #dot_lab_m1.set_data([t_pi2_lab],[P_lab[idx_lab,0]])
    #dot_lab_p1.set_data([t_pi2_lab],[P_lab[idx_lab,2]])

    #label_lab_m1.set_position((t_pi2_lab,P_lab[idx_lab,0]+label_offset))
    #label_lab_m1.set_text(f"{P_lab[idx_lab,0]:.5f}")

    #label_lab_p1.set_position((t_pi2_lab,P_lab[idx_lab,2]-label_offset))
    #label_lab_p1.set_text(f"{P_lab[idx_lab,2]:.5f}")

    #dot_rwa_m1.set_data([t_pi2_rwa],[P_rwa[idx_rwa,0]])
    #dot_rwa_p1.set_data([t_pi2_rwa],[P_rwa[idx_rwa,2]])

    #abel_rwa_m1.set_position((t_pi2_rwa,P_rwa[idx_rwa,0]+label_offset))
    #label_rwa_m1.set_text(f"{P_rwa[idx_rwa,0]:.5f}")

    #label_rwa_p1.set_position((t_pi2_rwa,P_rwa[idx_rwa,2]-label_offset))
    #label_rwa_p1.set_text(f"{P_rwa[idx_rwa,2]:.5f}")

    fig.canvas.draw_idle()

Bz_slider.on_changed(update)
det_slider.on_changed(update)

update(None)
plt.show()