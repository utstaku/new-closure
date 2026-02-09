import numpy as np
import matplotlib.pyplot as plt
import cmocean
import os

k =0.35
tmax = 30
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
vlasov_data = np.load('../vlasov_single_data/A=0.05_k=0.35/moments_dt0.002.npz')
mlc_data    = np.load('../fluid_simulation_results/windowAE_closure_A=0.05_k=0.35/moments.npz')

t_v = vlasov_data["t"]; n_v, u_v, p_v, dq_v, Eenergy_v = \
    vlasov_data["n"], vlasov_data["u"], vlasov_data["p"], vlasov_data["dq_dx"], vlasov_data["Energy"]
t_m = mlc_data["t"]; n_m, u_m, p_m, dq_m, Eenergy_m = \
    mlc_data["n"], mlc_data["u"], mlc_data["p"], mlc_data["dq_dx"], mlc_data["Energy"]

t_v = t_v[:len(t_m)]
n_v = n_v[:len(t_m)]
u_v = u_v[:len(t_m)]
p_v = p_v[:len(t_m)]    
dq_v = dq_v[:len(t_m)]
Eenergy_v = Eenergy_v[:len(t_m)]

outdir = '../picture/comparing/A=0.05_k=0.35_simpleAE/'
os.makedirs(outdir, exist_ok=True)

plt.figure()
plt.plot(t_v,Eenergy_v,'k-',label="Vlasov")
plt.plot(t_m,Eenergy_m,'r--',label='Fluid+ML(FNO)')
plt.yscale('log')
plt.xlabel('t')
plt.ylabel('Electric Field Energy')
plt.title('Vlasov vs ML closure')
plt.legend()
plt.savefig(outdir + 'Eenergy_compare.png')

# === Fig.2: dq/dx 比較 ===
fig, axs = plt.subplots(3, 1, figsize=(8, 8))
spacetime_plot(axs[0], dq_v, "Vlasov: ∂q/∂x (Ground Truth)",cmap=cmocean.cm.balance)
spacetime_plot(axs[1], dq_m, "Fluid + ML (FNO): ∂q/∂x Prediction",cmap=cmocean.cm.balance)
spacetime_plot(axs[2], np.abs(dq_v - dq_m), "Absolute Error |Δ(∂q/∂x)|", cmap=cmocean.cm.thermal)
fig.suptitle("Comparison of Heat-Flux Gradient ∂q/∂x", fontsize=12, y=0.98)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(outdir + 'dq_dx.png', dpi=300)
plt.close()


# === Fig.5: n,u,p の比較 ===
fig, axs = plt.subplots(3, 2, figsize=(9, 8))
titles = [["Vlasov", "Fluid + ML"],
          ["Vlasov", "Fluid + ML"],
          ["Vlasov", "Fluid + ML"]]
labels = ["Density n", "Velocity u", "Pressure p"]
data_sets = [(n_v, n_m), (u_v, u_m), (p_v, p_m)]

for i in range(3):
    for j in range(2):
        spacetime_plot(axs[i,j], data_sets[i][j], f"{titles[i][j]}\n{labels[i]}",cmap=cmocean.cm.balance)
fig.suptitle("Comparison of Fluid Moments (n, u, p)", fontsize=12, y=0.98)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(outdir + 'moments_compare.png', dpi=300)
plt.close()



# === Fig.6: 絶対誤差 ===
fig, axs = plt.subplots(3, 1, figsize=(8, 7))
quantities = ['Density n', 'Velocity u', 'Pressure p']
for i, (v, m, qname) in enumerate(zip([n_v,u_v,p_v],[n_m,u_m,p_m], quantities)):
    spacetime_plot(axs[i], np.abs(v-m), f'ML Error: {qname}', cmap=cmocean.cm.balance)
fig.suptitle("Absolute Errors of Fluid Moments relative to Vlasov", fontsize=12, y=0.98)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(outdir + 'abserror.png', dpi=300)
plt.close()
