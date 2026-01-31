import os
import json
import numpy as np
from scipy.interpolate import CubicSpline

# ============================================================
# Utils
# ============================================================

def log_uniform(rng, low, high):
    """sample from log-uniform in [low, high]"""
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))

def fourier_resample_periodic(y, N_new, axis=-1):
    """
    Periodic Fourier resampling from N_old -> N_new along axis.
    Works for real-valued signals; uses rFFT zero-pad/truncate.
    """
    y = np.asarray(y)
    N_old = y.shape[axis]
    Y = np.fft.rfft(y, axis=axis)

    n_old = N_old // 2 + 1
    n_new = N_new // 2 + 1
    n_copy = min(n_old, n_new)

    # Create Y_new with adjusted rfft length
    new_shape = list(Y.shape)
    new_shape[axis] = n_new
    Y_new = np.zeros(new_shape, dtype=np.complex128)

    # Copy low modes
    sl_old = [slice(None)] * Y.ndim
    sl_new = [slice(None)] * Y.ndim
    sl_old[axis] = slice(0, n_copy)
    sl_new[axis] = slice(0, n_copy)
    Y_new[tuple(sl_new)] = Y[tuple(sl_old)]

    # Scale to keep amplitude consistent under numpy FFT conventions
    Y_new *= (N_new / N_old)

    y_new = np.fft.irfft(Y_new, n=N_new, axis=axis)
    return y_new

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ============================================================
# Vlasov-Poisson 1D1V (electron) solver (semi-Lagrangian)
# ============================================================

class Vlasov1D1V:
    """
    Electron Vlasov-Poisson with fixed ion background (ni=1),
    periodic in x, truncated v in [-Vmax, Vmax).
    Strang splitting:
      x half-step (Fourier shift), compute E, v full-step (CubicSpline), x half-step.
    """
    def __init__(self, N, M, L, Vmax, dt, T0):
        # grids
        self.N = int(N)          # x points
        self.M = int(M)          # half of v points, total Nv = 2M
        self.L = float(L)
        self.Vmax = float(Vmax)
        self.dt = float(dt)
        self.T0 = float(T0)

        # x grid
        self.dx = self.L / self.N
        self.x = np.arange(self.N) * self.dx

        # v grid (2M points: -M,...,M-1)
        self.dv = 2 * self.Vmax / (2 * self.M - 1)
        v_index = np.arange(-self.M, self.M, 1, dtype=float)
        self.v = v_index * self.dv

        # Fourier k vector for x (consistent with FFT grid)
        self.k_vec = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)

    # -----------------------
    # Moments
    # -----------------------
    def density(self, f):
        return np.sum(f, axis=1) * self.dv

    def velocity(self, f):
        n = self.density(f)
        j1 = np.sum(f * self.v[None, :], axis=1) * self.dv
        return j1 / (n + 1e-30)

    def pressure(self, f):
        u = self.velocity(f)
        vc = self.v[None, :] - u[:, None]
        return np.sum(f * (vc ** 2), axis=1) * self.dv

    def dq_dx(self, f):
        u = self.velocity(f)
        vc = self.v[None, :] - u[:, None]
        q = np.sum(f * (vc ** 3), axis=1) * self.dv
        q_hat = np.fft.fft(q)
        return np.fft.ifft(1j * self.k_vec * q_hat).real

    # -----------------------
    # Field solve (Poisson)
    # -----------------------
    def poisson_E(self, f):
        # n(x) = ∫ f dv
        n = np.trapezoid(f, self.v, axis=1)
        dn = 1.0 - n  # rhs: 1 - ne
        dn_hat = np.fft.fft(dn)

        E_hat = np.zeros_like(dn_hat, dtype=np.complex128)
        mask = (self.k_vec != 0)
        E_hat[mask] = dn_hat[mask] / (1j * self.k_vec[mask])
        E_hat[~mask] = 0.0

        E = np.fft.ifft(E_hat).real
        return E

    # -----------------------
    # Semi-Lagrangian shifts
    # -----------------------
    def shift_x_half(self, f, dt_half):
        # Fourier shift in x: f(x,v) -> f(x - v*dt_half, v)
        F = np.fft.fft(f, axis=0)
        phase = np.exp(-1j * self.k_vec[:, None] * self.v[None, :] * dt_half)
        return np.fft.ifft(F * phase, axis=0).real

    def shift_v_full(self, f, E_x, dt_full):
        # Lagrangian shift in v: f(x,v) -> f(x, v + E*dt_full)
        v_shift = E_x * dt_full
        f_out = np.zeros_like(f)

        vmin = self.v[0]
        vmax_eff = self.v[-1] + (self.v[1] - self.v[0])

        for ix in range(f.shape[0]):
            cs = CubicSpline(self.v, f[ix, :], bc_type="natural", extrapolate=False)
            vv = self.v + v_shift[ix]
            fout = cs(vv)

            fout[~np.isfinite(fout)] = 0.0
            mask = (vv < vmin) | (vv > vmax_eff)
            fout[mask] = 0.0
            f_out[ix, :] = fout

        return f_out

    # -----------------------
    # One timestep (Strang)
    # -----------------------
    def step(self, f):
        dt = self.dt
        f = self.shift_x_half(f, 0.5 * dt)
        E = self.poisson_E(f)
        f = self.shift_v_full(f, E, dt)
        f = self.shift_x_half(f, 0.5 * dt)
        return f

# ============================================================
# Dataset generation
# ============================================================

def make_single_mode_ic(x, k0, A, phi, T0, v):
    """
    f(x,v) = n(x) * Maxwellian(v; T0)
    n(x) = 1 + A cos(k0 x + phi)
    """
    n0 = 1.0 + A * np.cos(k0 * x + phi)
    vt = np.sqrt(T0)
    f0_v = (1.0 / np.sqrt(2 * np.pi * vt**2)) * np.exp(-v**2 / (2 * vt**2))
    f = n0[:, None] * f0_v[None, :]
    return f, n0

def run_case_and_save(
    case_id: int,
    seed: int,
    out_root: str,
    # ---- random ranges ----
    k_range=(0.2, 1.0),        # continuous sampling for k
    A_range=(1e-4, 0.15),      # amplitude (log-uniform)
    m_choices=(1, 2, 3, 4, 5), # L = 2π m / k
    # ---- resolution policy ----
    N_lambda=64,               # points per wavelength (in physical x)
    N_ref=256,                 # common grid for ML (x_hat grid size)
    # ---- solver ----
    M=32,
    Vmax=6.0,
    dt=0.01,
    tmax=30.0,
    T0=1.0,
):
    rng = np.random.default_rng(seed)

    # Sample k (continuous), choose integer m, define L for periodic consistency
    k0 = float(rng.uniform(*k_range))
    m = int(rng.choice(m_choices))
    L = 2 * np.pi * m / k0

    # Choose physical Nx based on points-per-wavelength policy
    Nx = int(m * N_lambda)

    # Sample amplitude (log-uniform) and random phase
    A = log_uniform(rng, *A_range)
    phi = float(rng.uniform(0.0, 2 * np.pi))

    # Create solver
    solver = Vlasov1D1V(N=Nx, M=M, L=L, Vmax=Vmax, dt=dt, T0=T0)

    # Initial condition (single mode)
    f, ne0 = make_single_mode_ic(solver.x, k0, A, phi, T0, solver.v)

    # Time loop
    Nt = int(np.floor(tmax / dt)) + 1
    t_arr = np.arange(Nt) * dt

    n_hist   = np.zeros((Nt, Nx), dtype=np.float64)
    u_hist   = np.zeros((Nt, Nx), dtype=np.float64)
    p_hist   = np.zeros((Nt, Nx), dtype=np.float64)
    dqdx_hist= np.zeros((Nt, Nx), dtype=np.float64)

    for it, t in enumerate(t_arr):
        n_hist[it]    = solver.density(f)
        u_hist[it]    = solver.velocity(f)
        p_hist[it]    = solver.pressure(f)
        dqdx_hist[it] = solver.dq_dx(f)

        if it != Nt - 1:
            f = solver.step(f)

    # ---- Save raw (case grid) ----
    case_dir = os.path.join(out_root, f"data_{case_id:04d}")
    ensure_dir(case_dir)

    np.savez(
        os.path.join(case_dir, "moments_raw.npz"),
        t=t_arr,
        x=solver.x,
        v=solver.v,
        n=n_hist,
        u=u_hist,
        p=p_hist,
        dq_dx=dqdx_hist,
        # meta
        seed=seed, case_id=case_id,
        k0=k0, A=A, phi=phi, m=m,
        L=L, Nx=Nx, dx=solver.dx,
        M=M, Vmax=Vmax, dv=solver.dv,
        dt=dt, tmax=tmax, T0=T0,
    )

    np.savez(
        os.path.join(case_dir, "init_info.npz"),
        seed=seed, case_id=case_id,
        k0=k0, A=A, phi=phi, m=m,
        L=L, Nx=Nx, dx=solver.dx,
        ne0=ne0
    )

    # ---- Resample to common x_hat grid (ML grid) ----
    # Use x_hat in [0,1). Note: physical mapping is x = L * x_hat.
    x_hat = np.linspace(0.0, 1.0, N_ref, endpoint=False)

    n_ml    = fourier_resample_periodic(n_hist,    N_ref, axis=1)
    u_ml    = fourier_resample_periodic(u_hist,    N_ref, axis=1)
    p_ml    = fourier_resample_periodic(p_hist,    N_ref, axis=1)
    dqdx_ml = fourier_resample_periodic(dqdx_hist, N_ref, axis=1)

    np.savez(
        os.path.join(case_dir, "moments_ml.npz"),
        t=t_arr,
        x_hat=x_hat,
        n=n_ml, u=u_ml, p=p_ml, dq_dx=dqdx_ml,
        # meta kept for normalization / conditioning
        seed=seed, case_id=case_id,
        k0=k0, A=A, phi=phi, m=m,
        L=L, Nx_raw=Nx, N_ref=N_ref,
        dt=dt, tmax=tmax, T0=T0,
        Vmax=Vmax, M=M,
    )

    # Also dump a human-readable json summary
    summary = dict(
        seed=seed, case_id=case_id,
        k0=k0, A=A, phi=phi, m=m,
        L=L, Nx_raw=Nx, dx=float(solver.dx),
        N_ref=N_ref,
        dt=dt, tmax=tmax, T0=T0,
        Vmax=Vmax, M=M,
        k_range=list(k_range), A_range=list(A_range),
        m_choices=list(m_choices), N_lambda=N_lambda
    )
    with open(os.path.join(case_dir, "summary.json"), "w") as fjson:
        json.dump(summary, fjson, indent=2)

    print(f"[OK] data_{case_id:04d} seed={seed} | k0={k0:.4f} A={A:.3e} m={m} L={L:.3f} Nx={Nx} -> N_ref={N_ref}")


def generate_many_cases(
    Ncases: int,
    out_root: str,
    start_seed: int = 0,
    **kwargs
):
    ensure_dir(out_root)
    # save global config
    cfg = dict(Ncases=Ncases, start_seed=start_seed, **kwargs)
    with open(os.path.join(out_root, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    for i in range(Ncases):
        seed = start_seed + i
        run_case_and_save(case_id=i, seed=seed, out_root=out_root, **kwargs)

    print("==== All datasets generated ====")


# ============================================================
# Example execution
# ============================================================
if __name__ == "__main__":
    generate_many_cases(
        Ncases=300,
        out_root="../vlasov_random_xhat_dataset",
        start_seed=0,

        # random
        k_range=(0.2, 1.0),
        A_range=(1e-4, 0.15),
        m_choices=(1, 2, 3, 4, 5),

        # resolution
        N_lambda=64,   # Nx_raw = m * N_lambda
        N_ref=256,     # ML grid length (x_hat grid)

        # solver
        M=32,
        Vmax=6.0,
        dt=0.01,
        tmax=30.0,
        T0=1.0,
    )
