import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline


#parameters
Amax = 0.15
k0 = 0.1
N = 64         # x方向のgrid数
M = 32         # v方向のgrid数(半分)
Vmax = 6.0     # vの打ち切り速度
dt = 0.01       # time step (ω_pe^{-1} units)
tmax = 30.0    # end time (enough to see at least one recurrence
save_root = "../vlasov_random_data4"
Nmodes = 5
T0=1.0

# v方向の分割
dv = 2*Vmax/(2*M - 1)
## v方向のindex
v_index = np.arange(-M, M, 1, dtype = float)
v = v_index*dv

# x方向の分割
L = 2*np.pi / k0
dx = L/N
x = np.arange(N)*dx

# フーリエ空間における波数ベクトル
k_vec = 2*np.pi*np.fft.fftfreq(N, d=dx)

def generate_initial_density(x, n0, Nmodes, k0, Amax, seed):
    rng = np.random.default_rng(seed)

    # k_j = j * k0 とする（Wei et al. も "multiple modes" のみ言及）
    k_list = np.arange(1, Nmodes+1) * k0

    # ランダム振幅と位相
    A = rng.uniform(0.0, Amax, size=Nmodes)
    phi = rng.uniform(0.0, 2*np.pi, size=Nmodes)

    # 摂動を合成
    perturb = np.zeros_like(x)
    for Aj, kj, phij in zip(A, k_list, phi):
        perturb += Aj * np.cos(kj * x + phij)

    ne0 = n0 * (1.0 + perturb)
    ni0 = n0 * np.ones_like(x)  # イオンは静止一様背景

    return ne0, ni0, A, phi, k_list

# x方向への半ステップシフト
def shift_x_semi_lagrangian(f_in, v_arr, dt_half):
    F = np.fft.fft(f_in,axis=0)
    phase = np.exp(-1j * k_vec[:,None] * v_arr[None,:] * dt_half)
    F_shift = F*phase
    return np.fft.ifft(F_shift,axis=0).real

# Eの計算
def poisson_E_caluc(f_in):
    #fをvについて台形積分することでnを求める
    n = np.trapezoid(f_in,v,axis=1)
    #poisson方程式の右辺
    dn = 1-n
    dn_hat = np.fft.fft(dn)
    E_hat = np.zeros_like(dn_hat, dtype=complex)
    for i, kk in enumerate(k_vec):
        if kk != 0:
            E_hat[i] = dn_hat[i]/(1j*kk)
        else:
            E_hat[i] = 0
    E = np.fft.ifft(E_hat).real
    return E,dn,dn_hat

# v方向への1ステップシフト
def shift_v_lagrangian(f_in, E_x, dt_full):
    v_shift = E_x * dt_full
    f_out = np.zeros_like(f_in)

    for ix in range(f_in.shape[0]):
        cs = CubicSpline(v, f_in[ix,:],bc_type='natural',extrapolate=False)
        vv = v+v_shift[ix]
        fout = cs(vv)

        vmin, vmax = v[0], v[-1] + (v[1]-v[0]) #有効な速度範囲
        fout[~np.isfinite(fout)] = 0.0 #補間で出てきたNan,infを０に置き換える
        mask = (vv < vmin) | (vv > vmax)
        fout[mask] = 0.0 #補間点が有効な速度範囲を出ていたら0

        f_out[ix,:] = fout
    return f_out

# densityの計算
def density(f):
    return np.sum(f, axis=1) * dv

def dn_dx(n):
    n_hat = np.fft.fft(n)
    dn_dx = np.fft.ifft(1j * k_vec * n_hat).real
    return dn_dx

#速度uの計算
def velocity(f,n):
    j1=np.sum(f*v[None, :], axis=1) * dv
    return j1/n

def du_dx(u):
    u_hat = np.fft.fft(u)
    du_dx = np.fft.ifft(1j * k_vec * u_hat).real
    return du_dx

#圧力pの計算
def pressure(f,n):
    vc = v[None, :] - velocity(f,n)[:, None]
    p = np.sum(f * (vc**2), axis=1)*dv
    return p

def dp_dx(p):
    p_hat = np.fft.fft(p)
    dp_dx = np.fft.ifft(1j * k_vec * p_hat).real
    return dp_dx

#熱流速勾配の計算
def dq_dx(f,u):
    vc = v[None, :] - u[:, None]
    q = np.sum(f * (vc**3), axis=1)*dv
    q_hat = np.fft.fft(q)
    dq_dx = np.fft.ifft(1j * k_vec *q_hat).real
    return dq_dx

def run_vlasov_case(seed, save_dir, Nmodes, k0, Amax, T0):
    """
    seed → 初期条件の乱数
    save_dir → 保存フォルダ (例: "raw_data/vlasov_multi/data_0001")
    """

    
    os.makedirs(save_dir, exist_ok=True)

    # --------- 初期密度（Wei et al.方式） ----------
    ne0, ni0, A_list, phi_list, k_list = generate_initial_density(
        x,
        n0=1.0,
        Nmodes=Nmodes,
        k0=k0,
        Amax=Amax,
        seed=seed
    )

    # --------- 初期 f(x,v) (Maxwellian × 密度) ----------
    vt = np.sqrt(T0)
    f0_v = (1.0/np.sqrt(2*np.pi*vt**2)) * np.exp(-v**2/(2*vt**2))
    f = ne0[:, None] * f0_v[None, :]

    # --------- 時間発展 ----------
    t_list = []
    n_list = []
    u_list = []
    p_list = []
    dqdx_list = []
    dndx_list = []
    dudx_list = []
    dpdx_list = []

    t = 0.0
    while t < tmax:
        # モーメントを計算（あなたの関数に差し替え）
        n = density(f)
        u = velocity(f,n)
        p = pressure(f,n)
        dqdx = dq_dx(f,u)
        dndx = dn_dx(n)
        dudx = du_dx(u)
        dpdx = dp_dx(p)

        # 保存
        t_list.append(t)
        n_list.append(n)
        u_list.append(u)
        p_list.append(p)
        dqdx_list.append(dqdx)
        dndx_list.append(dndx)
        dudx_list.append(dudx)
        dpdx_list.append(dpdx)

        # 時間ステップ進める（あなたのコードのステップ関数に置き換え）
        #xを半分
        f = shift_x_semi_lagrangian(f, v, dt*0.5)

        #Eの計算
        E_half,_,_ = poisson_E_caluc(f)

        #v方向に進める
        f = shift_v_lagrangian(f, E_half, dt)

        #x方向に半分進める
        f = shift_x_semi_lagrangian(f, v, dt*0.5)

        t += dt

    # --------- 保存（moments.npz） ----------
    np.savez(
        os.path.join(save_dir, "moments.npz"),
        t=np.array(t_list),
        n=np.array(n_list),
        u=np.array(u_list),
        p=np.array(p_list),
        dq_dx=np.array(dqdx_list),
        dn_dx=np.array(dndx_list),
        du_dx=np.array(dudx_list),
        dp_dx=np.array(dpdx_list)
    )

    # 初期条件も保存（再現用）
    np.savez(
        os.path.join(save_dir, "init_info.npz"),
        A=A_list,
        phi=phi_list,
        k_list=k_list,
        ne0=ne0
    )

    print(f"[OK] seed={seed} saved to {save_dir}")
    print(len(t_list))

def generate_many_vlasov_cases(Ncases, out_root,start_seed=0):
    for i in range(Ncases):
        seed = start_seed + i
        save_dir = os.path.join(out_root, f"data_{i:04d}")

        run_vlasov_case(seed, save_dir,Nmodes=Nmodes,k0=k0,Amax=Amax,T0=T0)

    print("==== All datasets generated ====")

generate_many_vlasov_cases(Ncases=300, out_root=save_root)