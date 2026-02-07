# train_closure_neuralop_fno.py
# Minimal template: (n,u,p) history -> Temporal Encoder (per-x) -> neuralop FNO (space nonlocal) -> dq_dx(x,t)
#
# Requirements:
#   pip install neuraloperator  (provides `neuralop`)
#   pip install torch numpy
#
# Data:
#   raw_data/vlasov_A=0.1/moments_train.npz
#   raw_data/vlasov_A=0.1/moments_test.npz
# containing keys: t, x, n, u, p, dq_dx

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from neuralop.models import FNO
except Exception as e:
    raise ImportError(
        "Could not import `neuralop`. Install NeuralOperator:\n"
        "  pip install neuraloperator\n"
        f"Original error: {e}"
    )

# ============================================================
# Dataset: sliding window
#   input  X_seq: (L, N, C_in)
#   target y:     (N,)   (dq_dx at time t)
# ============================================================
class MomentsWindowDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        L: int = 32,
        stride: int = 1,
        use_channels=("n", "u", "p"),
        normalize: bool = True,
        stats: dict | None = None,
    ):
        data = np.load(npz_path)

        self.x = data["x"].astype(np.float32)     # (N,)
        self.t = data["t"].astype(np.float32)     # (Nt,)
        self.N = self.x.shape[0]
        self.Nt = self.t.shape[0]

        # load fields (Nt, N)
        fields = {k: data[k].astype(np.float32) for k in ["n", "u", "p", "dq_dx"]}

        # stack inputs (Nt, N, C_in)
        X = np.concatenate([fields[ch][..., None] for ch in use_channels], axis=-1)  # (Nt, N, C_in)
        y = fields["dq_dx"]  # (Nt, N)

        self.L = int(L)
        self.stride = int(stride)
        self.use_channels = tuple(use_channels)

        # valid indices: predict at i using [i-L+1 ... i]
        self.idxs = np.arange(self.L - 1, self.Nt, self.stride, dtype=np.int64)

        self.normalize = bool(normalize)
        if self.normalize:
            if stats is None:
                mu = X.reshape(-1, X.shape[-1]).mean(axis=0)
                sig = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-8
                y_mu = y.mean()
                y_sig = y.std() + 1e-8
                self.stats = {
                    "mu": mu.astype(np.float32),
                    "sig": sig.astype(np.float32),
                    "y_mu": np.float32(y_mu),
                    "y_sig": np.float32(y_sig),
                    "use_channels": self.use_channels,
                }
            else:
                self.stats = stats

            X = (X - self.stats["mu"][None, None, :]) / self.stats["sig"][None, None, :]
            y = (y - self.stats["y_mu"]) / self.stats["y_sig"]

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, j: int):
        i = self.idxs[j]
        X_seq = self.X[i - self.L + 1 : i + 1]  # (L, N, C_in)
        y_t = self.y[i]                         # (N,)
        return torch.from_numpy(X_seq), torch.from_numpy(y_t)


# ============================================================
# Temporal Encoder: per-x time conv (shared weights across x)
# Input:  X_seq (B, L, N, C_in)
# Output: M     (B, N, d_m)
#
# Implementation:
#   reshape so each x is treated as an independent sequence:
#   (B, L, N, C) -> (B*N, C, L) -> Conv1d over time -> pool -> (B, N, d_m)
# ============================================================
class TemporalEncoderTCN(nn.Module):
    def __init__(self, C_in: int = 3, d_m: int = 16, hidden: int = 64, layers: int = 2, kernel: int = 5):
        super().__init__()
        mods: list[nn.Module] = []
        c = C_in
        for _ in range(layers):
            mods += [
                nn.Conv1d(c, hidden, kernel_size=kernel, padding=kernel // 2),
                nn.GELU(),
            ]
            c = hidden
        mods += [nn.Conv1d(c, d_m, kernel_size=1)]
        self.net = nn.Sequential(*mods)

    def forward(self, X_seq: torch.Tensor) -> torch.Tensor:
        # X_seq: (B, L, N, C)
        B, L, N, C = X_seq.shape
        X = X_seq.permute(0, 2, 3, 1).contiguous().view(B * N, C, L)  # (B*N, C, L)
        H = self.net(X)  # (B*N, d_m, L)
        M = H.mean(dim=-1)  # (B*N, d_m)
        return M.view(B, N, -1)  # (B, N, d_m)


# ============================================================
# Spatial Nonlocal Mapper: neuralop FNO
# Input:  M (B, N, d_m)
# Output: y_hat (B, N)
#
# neuralop FNO expects (B, in_channels, N) for 1D grids.
# ============================================================
class ClosureModel(nn.Module):
    def __init__(
        self,
        C_in: int = 3,
        L: int = 32,
        d_m: int = 16,
        # temporal encoder
        t_hidden: int = 64,
        t_layers: int = 2,
        t_kernel: int = 5,
        # FNO
        fno_modes: int = 16,
        fno_hidden: int = 64,
        fno_layers: int = 4,
        out_channels: int = 1,
    ):
        super().__init__()
        self.L = L
        self.temporal = TemporalEncoderTCN(C_in=C_in, d_m=d_m, hidden=t_hidden, layers=t_layers, kernel=t_kernel)

        # neuralop FNO: N-dimensional; for 1D use n_modes=(modes,)
        self.fno = FNO(
            n_modes=(fno_modes,),
            in_channels=d_m,
            out_channels=out_channels,
            hidden_channels=fno_hidden,
            n_layers=fno_layers,
        )

    def forward(self, X_seq: torch.Tensor) -> torch.Tensor:
        # X_seq: (B, L, N, C_in)
        M = self.temporal(X_seq)         # (B, N, d_m)
        M = M.permute(0, 2, 1)           # (B, d_m, N)  -> for FNO
        y = self.fno(M)                  # (B, 1, N)
        return y.squeeze(1)              # (B, N)


# ============================================================
# Training
# ============================================================
def train(
    npz_train: str,
    npz_valid: str | None = None,
    L: int = 32,
    batch_size: int = 16,
    lr: float = 2e-3,
    epochs: int = 30,
    # FNO params
    fno_modes: int = 16,
    fno_hidden: int = 64,
    fno_layers: int = 4,
    # temporal params
    d_m: int = 16,
    t_hidden: int = 64,
    t_layers: int = 2,
    t_kernel: int = 5,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_train = MomentsWindowDataset(npz_train, L=L, stride=1, use_channels=("n", "u", "p"), normalize=True)
    stats = ds_train.stats
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    dl_valid = None
    if npz_valid is not None:
        ds_valid = MomentsWindowDataset(
            npz_valid, L=L, stride=1, use_channels=("n", "u", "p"), normalize=True, stats=stats
        )
        dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    model = ClosureModel(
        C_in=3,
        L=L,
        d_m=d_m,
        t_hidden=t_hidden,
        t_layers=t_layers,
        t_kernel=t_kernel,
        fno_modes=fno_modes,
        fno_hidden=fno_hidden,
        fno_layers=fno_layers,
        out_channels=1,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0

        for X_seq, y in dl_train:
            X_seq = X_seq.to(device)  # (B, L, N, C)
            y = y.to(device)          # (B, N)

            y_hat = model(X_seq)      # (B, N)
            loss = loss_fn(y_hat, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += float(loss.item())

        tr_loss /= max(1, len(dl_train))

        if dl_valid is not None:
            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for X_seq, y in dl_valid:
                    X_seq = X_seq.to(device)
                    y = y.to(device)
                    y_hat = model(X_seq)
                    va_loss += float(loss_fn(y_hat, y).item())
            va_loss /= max(1, len(dl_valid))
            print(f"[ep {ep:03d}] train {tr_loss:.6e} | valid {va_loss:.6e}")
        else:
            print(f"[ep {ep:03d}] train {tr_loss:.6e}")

    # save model
    ckpt = {
        "model": model.state_dict(),
        "stats": stats,
        "L": L,
        "arch": {
            "d_m": d_m,
            "t_hidden": t_hidden,
            "t_layers": t_layers,
            "t_kernel": t_kernel,
            "fno_modes": fno_modes,
            "fno_hidden": fno_hidden,
            "fno_layers": fno_layers,
        },
    }
    torch.save(ckpt, "simpleencode_closure_neuralop_fno.pth")
    print("saved: simpleencode_closure_neuralop_fno.pth")


if __name__ == "__main__":
    train(
        npz_train="../vlasov_single_data/A=0.1_k=0.35/moments_dt2e-3.npz",
        npz_valid="../vlasov_single_data/A=0.1_k=0.35/moments_dt2e-3.npz",
        L=32,
        batch_size=16,
        lr=2e-3,
        epochs=10,
        fno_modes=16,
        fno_hidden=64,
        fno_layers=4,
        d_m=16,
        t_hidden=64,
        t_layers=2,
        t_kernel=5,
    )
