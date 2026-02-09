import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from collections import deque

# ✅ あなたの学習済みモデル
import sys
sys.path.append("..")  # <- ここはあなたの環境に合わせてOK
from temporal_encoder.encoder import ClosureModel  # <- ここはあなたの環境に合わせてOK

# ============================================================
# Parameters
# ============================================================
k = 0.35
A = 0.1
n0 = 1.0
T0 = 1.0
me = 1.0
qe = -1.0
eps0 = 1.0

dt = 0.2
tmax = 30.0

L_x = 2 * np.pi / k
N_x = 64
dx = L_x / N_x
x = np.linspace(0, L_x, N_x, endpoint=False)

# Fourier derivative operator
kvec = 2 * np.pi * np.fft.fftfreq(N_x, d=dx)

def d_dx(f):
    return np.fft.ifft(1j * kvec * np.fft.fft(f)).real

# ============================================================
# Initial conditions
# ============================================================
n = n0 * (1.0 + A * np.cos(k * x))
u = np.zeros_like(x)
p = n * T0
Ex = (qe * n0 * A / (eps0 * k)) * np.sin(k * x)

# ============================================================
# Load trained window model (L=32, inchannel=3)
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = "../temporal_encoder/simpleencode_closure_neuralop_fno_dt0.2.pth"  # <- ここはあなたのckptに合わせて
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

# ---- instantiate the same model arch used in training ----
# ここはあなたの ClosureModel の __init__ と一致させてください
# 例：ClosureModel(C_in=3, L=32, ...) など
# 学習コードで arch を保存しているならそれを使うのが安全です
arch = ckpt.get("arch", {})

# できるだけ arch から復元（無ければデフォルトを仮定）
# ※あなたの ClosureModel の引数名に合わせて必要なら修正
model = ClosureModel(
    C_in=3,
    L=int(ckpt.get("L", 64)),
    d_m=int(arch.get("d_m", 16)),
    t_hidden=int(arch.get("t_hidden", 64)),
    t_layers=int(arch.get("t_layers", 2)),
    t_kernel=int(arch.get("t_kernel", 5)),
    fno_modes=int(arch.get("fno_modes", 16)),
    fno_hidden=int(arch.get("fno_hidden", 64)),
    fno_layers=int(arch.get("fno_layers", 4)),
).to(device)

model.load_state_dict(ckpt["model"], strict=False)
model.eval()

stats = ckpt["stats"]
L = int(ckpt.get("L", 64))  # ✅ L=32

# stats: input (C,), output scalar
mu = np.array(stats["mu"], dtype=np.float32)    # (3,)
sig = np.array(stats["sig"], dtype=np.float32)  # (3,)
y_mu = float(stats["y_mu"])
y_sig = float(stats["y_sig"])

def normalize_X_seq(X_seq_raw):
    # X_seq_raw: (L,N,3) -> normalized
    return (X_seq_raw - mu[None, None, :]) / sig[None, None, :]

def denorm_dqdx(dqdx_norm):
    return dqdx_norm * y_sig + y_mu

# ============================================================
# History buffer (store frames of shape (N,3): [n,u,p])
# ============================================================
hist = deque(maxlen=L)

def make_frame(n, u, p):
    return np.stack([n, u, p], axis=-1).astype(np.float32)  # (N,3)

def make_window_from_list(frames_list, L, fallback_frame):
    """
    frames_list: list of frames (each (N,3)), length <= L
    fallback_frame: (N,3) used when list is empty and for padding
    return: (L,N,3)
    """
    if len(frames_list) == 0:
        frames_list = [fallback_frame]

    if len(frames_list) >= L:
        return np.stack(frames_list[-L:], axis=0)

    pad_len = L - len(frames_list)
    first = frames_list[0][None, ...]  # (1,N,3)
    pad = np.repeat(first, pad_len, axis=0)
    return np.concatenate([pad, np.stack(frames_list, axis=0)], axis=0)

@torch.no_grad()
def predict_dqdx_from_window(X_seq_raw):
    """
    X_seq_raw: (L,N,3) physical
    return dqdx: (N,) physical
    """
    X_seq_norm = normalize_X_seq(X_seq_raw)                      # (L,N,3)
    X_t = torch.from_numpy(X_seq_norm[None, ...]).to(device)     # (1,L,N,3)
    y = model(X_t)

    # output shape handling: (1,N) or (1,1,N)
    if y.ndim == 3:
        dqdx_norm = y[0, 0].detach().cpu().numpy()
    else:
        dqdx_norm = y[0].detach().cpu().numpy()

    dqdx = denorm_dqdx(dqdx_norm).astype(np.float32)
    return dqdx

def predict_dqdx_stage(n_s, u_s, p_s, hist_deque):
    """
    RK stages用：履歴dequeは更新せず、仮の状態(n_s,u_s,p_s)を末尾に付けた窓で予測する
    """
    frame_s = make_frame(n_s, u_s, p_s)
    frames = list(hist_deque) + [frame_s]     # “今”を入れる
    X_seq_raw = make_window_from_list(frames, L=L, fallback_frame=frame_s)
    return predict_dqdx_from_window(X_seq_raw)

# ============================================================
# RHS (uses dqdx predicted with time-window model)
# ============================================================
alpha = 1.0

def rhs_with_dqdx(n, u, p, Ex, dqdx):
    dn = -d_dx(n * u)
    du = -u * d_dx(u) - (1.0 / (me * n)) * d_dx(p) + (qe / me) * Ex
    dp = -u * d_dx(p) - 3.0 * p * d_dx(u) - alpha * dqdx
    dEx = -(qe / eps0) * n * u
    return dn, du, dp, dEx

def rk4_step_window(n, u, p, Ex, dt, hist_deque):
    """
    重要：
    - 履歴dequeはこの関数内では「仮に使うだけ」
    - 最後に (n_new,u_new,p_new) のframeだけを1回 append する
    - RK4の各ステージは “hist + provisional_state” でdqdxを計算
    """
    # k1
    dq1 = predict_dqdx_stage(n, u, p, hist_deque)
    k1 = rhs_with_dqdx(n, u, p, Ex, dq1)

    # k2
    n2  = n  + 0.5 * dt * k1[0]
    u2  = u  + 0.5 * dt * k1[1]
    p2  = p  + 0.5 * dt * k1[2]
    Ex2 = Ex + 0.5 * dt * k1[3]
    dq2 = predict_dqdx_stage(n2, u2, p2, hist_deque)
    k2 = rhs_with_dqdx(n2, u2, p2, Ex2, dq2)

    # k3
    n3  = n  + 0.5 * dt * k2[0]
    u3  = u  + 0.5 * dt * k2[1]
    p3  = p  + 0.5 * dt * k2[2]
    Ex3 = Ex + 0.5 * dt * k2[3]
    dq3 = predict_dqdx_stage(n3, u3, p3, hist_deque)
    k3 = rhs_with_dqdx(n3, u3, p3, Ex3, dq3)

    # k4
    n4  = n  + dt * k3[0]
    u4  = u  + dt * k3[1]
    p4  = p  + dt * k3[2]
    Ex4 = Ex + dt * k3[3]
    dq4 = predict_dqdx_stage(n4, u4, p4, hist_deque)
    k4 = rhs_with_dqdx(n4, u4, p4, Ex4, dq4)

    n_new  = n  + (dt / 6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    u_new  = u  + (dt / 6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    p_new  = p  + (dt / 6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    Ex_new = Ex + (dt / 6.0) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

    # ログ用に “現在ステップのdqdx” は dq1 を採用（代表値）
    dqdx_cur = dq1

    # ✅ 1ステップ進んだ後に履歴を1回だけ更新
    hist_deque.append(make_frame(n_new, u_new, p_new))

    return n_new, u_new, p_new, Ex_new, dqdx_cur

# ============================================================
# Time evolution
# ============================================================
ts = []
Eenergy = []
n_history = []
u_history = []
p_history = []
dqdx_history = []
Ex_history = []

t = 0.0
step = 0

# 履歴初期化：最初の状態を入れておく（これで step=0 でも窓が作れる）
hist.clear()
hist.append(make_frame(n, u, p))

while t < tmax:
    ts.append(t)
    Eenergy.append(0.5 * L_x * np.mean(Ex**2))
    n_history.append(n.copy())
    u_history.append(u.copy())
    p_history.append(p.copy())
    Ex_history.append(Ex.copy())

    n, u, p, Ex, dqdx_cur = rk4_step_window(n, u, p, Ex, dt, hist)
    dqdx_history.append(dqdx_cur.copy())

    t += dt
    step += 1

    if step % 100 == 0:
        print(
            f"t={t:.3f}",
            " n[min,max]=", float(np.min(n)), float(np.max(n)),
            " u[max]=", float(np.max(np.abs(u))),
            " p[max]=", float(np.max(np.abs(p))),
            " dqdx[max]=", float(np.max(np.abs(dqdx_cur)))
        )

# ============================================================
# Save results
# ============================================================
Eenergy = np.array(Eenergy, dtype=np.float32)
Eenergy = Eenergy / Eenergy[0]

t_history = np.array(ts, dtype=np.float32)
n_history = np.array(n_history, dtype=np.float32)
u_history = np.array(u_history, dtype=np.float32)
p_history = np.array(p_history, dtype=np.float32)
dq_dx_history = np.array(dqdx_history, dtype=np.float32)
Energy_history = np.array(Eenergy, dtype=np.float32)

outdir = f"../fluid_simulation_results/windowAE_closure_A={A}_k={k}/"
os.makedirs(outdir, exist_ok=True)

np.savez(
    f"{outdir}/moments_dt0.2.npz",
    t=t_history,
    x=x,
    n=n_history,
    u=u_history,
    p=p_history,
    dq_dx=dq_dx_history,
    Energy=Energy_history,
)

print("saved:", f"{outdir}/moments.npz")

#energyのプロット
plt.figure()
plt.plot(t_history,Energy_history,'r-')
plt.yscale('log')   
plt.xlabel('t')
plt.ylabel('Electric Field Energy')
plt.title('Fluid + ML (FNO) Energy Evolution')
plt.show()