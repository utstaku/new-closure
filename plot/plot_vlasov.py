import numpy as np
import matplotlib.pyplot as plt
import cmocean
import os

k =0.35
tmax = 40
L = 2 * np.pi / k

# === 改良版 spacetime_plot ===
def spacetime_plot(ax, data, title="", cmap="RdBu_r", vmin=None, vmax=None):
    im = ax.imshow(np.array(data).T, aspect='auto', origin='lower',
                   extent=[0, tmax, 0, L], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_title(title, fontsize=10, pad=4)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

"""
# ---- データのロード ----
"""
vlasov_data = np.load('../vlasov_single_data/A=0.1_k=0.35/moments_dt0.2.npz')

t_v = vlasov_data["t"]; n_v, u_v, p_v, dq_v, Eenergy_v = \
    vlasov_data["n"], vlasov_data["u"], vlasov_data["p"], vlasov_data["dq_dx"], vlasov_data["Energy"]

outdir = '../picture/vlasov_only/A=0.1_k=0.35/'
os.makedirs(outdir, exist_ok=True)

plt.figure()
plt.plot(t_v,Eenergy_v,'k-',label="Vlasov")
plt.yscale('log')
plt.xlabel('t')
plt.ylabel('Electric Field Energy')
plt.title('Vlasov vs ML closure')
plt.legend()
plt.savefig(outdir+'Eenergy_compare.png')

# === Fig.2: dq/dx 比較 ===
fig, axs = plt.subplots(1, 1, figsize=(8, 4))
spacetime_plot(axs, dq_v, "Vlasov: ∂q/∂x (Ground Truth)",cmap=cmocean.cm.balance)
fig.suptitle("Comparison of Heat-Flux Gradient ∂q/∂x", fontsize=12, y=0.98)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(outdir+'dq_dx.png', dpi=300)
plt.close()


# === Fig.5: n,u,p の比較 ===
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
spacetime_plot(axs[0], n_v, "Vlasov: Density n",cmap=cmocean.cm.thermal)
spacetime_plot(axs[1], u_v, "Vlasov: Velocity u",cmap=cmocean.cm.thermal)
spacetime_plot(axs[2], p_v, "Vlasov: Pressure p",cmap=cmocean.cm.thermal)
fig.suptitle("Comparison of Fluid Moments", fontsize=12, y=0.98)
plt.tight_layout(rect=[0,0,1,0.97]) 
plt.savefig(outdir+'moments.png', dpi=300)
plt.close()