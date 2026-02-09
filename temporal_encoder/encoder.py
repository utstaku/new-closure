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
import math
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
# Temporal Encoder: per-x self-attention over time (shared weights across x)
# Input:  X_seq (B, L, N, C_in)
# Output: M     (B, N, d_m)
#
# Implementation:
#   (B, L, N, C) -> (B*N, L, C) -> proj -> TransformerEncoder -> attn pool
# ============================================================
class TemporalEncoderAttention(nn.Module):
    def __init__(
        self,
        C_in: int = 3,
        d_m: int = 16,
        n_heads: int = 4,
        layers: int = 2,
        ff_hidden: int = 64,
        dropout: float = 0.1,
        max_len: int = 32,
    ):
        super().__init__()
        self.input_proj = nn.Linear(C_in, d_m)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_m))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_m,
            nhead=n_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pool = nn.Parameter(torch.randn(1, d_m))
        self.dropout = nn.Dropout(dropout)

    def forward(self, X_seq: torch.Tensor) -> torch.Tensor:
        # X_seq: (B, L, N, C)
        B, L, N, C = X_seq.shape
        if L > self.pos_emb.shape[1]:
            raise ValueError(f"Sequence length {L} exceeds max_len {self.pos_emb.shape[1]}")
        X = X_seq.permute(0, 2, 1, 3).contiguous().view(B * N, L, C)  # (B*N, L, C)
        H = self.input_proj(X) + self.pos_emb[:, :L, :]
        H = self.dropout(H)
        H = self.encoder(H)  # (B*N, L, d_m)

        # attention pooling over time
        q = self.pool.view(1, 1, -1)  # (1, 1, d_m)
        scores = torch.sum(H * q, dim=-1) / math.sqrt(H.shape[-1])  # (B*N, L)
        weights = torch.softmax(scores, dim=-1)
        M = torch.sum(H * weights.unsqueeze(-1), dim=1)  # (B*N, d_m)
        return M.view(B, N, -1)  # (B, N, d_m)


# ============================================================
# Temporal Decoder: per-x latent -> time history reconstruction
# Input:  M      (B, N, d_m)
# Output: X_rec  (B, L, N, C_in)
# ============================================================
class TemporalDecoderMLP(nn.Module):
    def __init__(self, d_m: int = 16, L: int = 32, C_out: int = 3, hidden: int = 128, layers: int = 2):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        mods: list[nn.Module] = []
        c = d_m
        for _ in range(max(0, layers - 1)):
            mods += [nn.Linear(c, hidden), nn.GELU()]
            c = hidden
        mods += [nn.Linear(c, L * C_out)]
        self.net = nn.Sequential(*mods)
        self.L = L
        self.C_out = C_out

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        # M: (B, N, d_m)
        B, N, d_m = M.shape
        X = self.net(M.view(B * N, d_m))               # (B*N, L*C_out)
        X = X.view(B, N, self.L, self.C_out)           # (B, N, L, C_out)
        return X.permute(0, 2, 1, 3).contiguous()      # (B, L, N, C_out)


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
        temporal_encoder: str = "attn",  # "attn" | "tcn"
        t_hidden: int = 64,
        t_layers: int = 2,
        t_kernel: int = 5,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        # decoder (aux reconstruction)
        use_decoder: bool = True,
        dec_hidden: int = 128,
        dec_layers: int = 2,
        # FNO
        fno_modes: int = 16,
        fno_hidden: int = 64,
        fno_layers: int = 4,
        out_channels: int = 1,
    ):
        super().__init__()
        self.L = L
        if temporal_encoder == "tcn":
            self.temporal = TemporalEncoderTCN(C_in=C_in, d_m=d_m, hidden=t_hidden, layers=t_layers, kernel=t_kernel)
        elif temporal_encoder == "attn":
            self.temporal = TemporalEncoderAttention(
                C_in=C_in,
                d_m=d_m,
                n_heads=attn_heads,
                layers=t_layers,
                ff_hidden=t_hidden,
                dropout=attn_dropout,
                max_len=L,
            )
        else:
            raise ValueError(f"Unknown temporal_encoder: {temporal_encoder}")

        self.use_decoder = bool(use_decoder)
        if self.use_decoder:
            self.decoder = TemporalDecoderMLP(
                d_m=d_m,
                L=L,
                C_out=C_in,
                hidden=dec_hidden,
                layers=dec_layers,
            )

        # neuralop FNO: N-dimensional; for 1D use n_modes=(modes,)
        self.fno = FNO(
            n_modes=(fno_modes,),
            in_channels=d_m,
            out_channels=out_channels,
            hidden_channels=fno_hidden,
            n_layers=fno_layers,
        )

    def forward(self, X_seq: torch.Tensor, return_recon: bool = False):
        # X_seq: (B, L, N, C_in)
        M = self.temporal(X_seq)         # (B, N, d_m)
        M_fno = M.permute(0, 2, 1)       # (B, d_m, N)  -> for FNO
        y = self.fno(M_fno).squeeze(1)   # (B, N)

        if return_recon:
            if not self.use_decoder:
                raise ValueError("return_recon=True but decoder is disabled (use_decoder=False)")
            X_rec = self.decoder(M)       # (B, L, N, C_in)
            return y, X_rec
        return y


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
    temporal_encoder: str = "attn",
    d_m: int = 16,
    t_hidden: int = 64,
    t_layers: int = 2,
    t_kernel: int = 5,
    attn_heads: int = 4,
    attn_dropout: float = 0.1,
    # reconstruction loss
    use_decoder: bool = True,
    recon_weight: float = 0.1,
    dec_hidden: int = 128,
    dec_layers: int = 2,
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
        temporal_encoder=temporal_encoder,
        t_hidden=t_hidden,
        t_layers=t_layers,
        t_kernel=t_kernel,
        attn_heads=attn_heads,
        attn_dropout=attn_dropout,
        use_decoder=use_decoder,
        dec_hidden=dec_hidden,
        dec_layers=dec_layers,
        fno_modes=fno_modes,
        fno_hidden=fno_hidden,
        fno_layers=fno_layers,
        out_channels=1,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pred_loss_fn = nn.MSELoss()
    recon_loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_pred = 0.0
        tr_recon = 0.0

        for X_seq, y in dl_train:
            X_seq = X_seq.to(device)  # (B, L, N, C)
            y = y.to(device)          # (B, N)

            if use_decoder:
                y_hat, X_rec = model(X_seq, return_recon=True)
                pred_loss = pred_loss_fn(y_hat, y)
                recon_loss = recon_loss_fn(X_rec, X_seq)
                loss = pred_loss + recon_weight * recon_loss
                tr_recon += float(recon_loss.item())
            else:
                y_hat = model(X_seq)
                pred_loss = pred_loss_fn(y_hat, y)
                loss = pred_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += float(loss.item())
            tr_pred += float(pred_loss.item())

        tr_loss /= max(1, len(dl_train))
        tr_pred /= max(1, len(dl_train))
        if use_decoder:
            tr_recon /= max(1, len(dl_train))

        if dl_valid is not None:
            model.eval()
            va_loss = 0.0
            va_pred = 0.0
            va_recon = 0.0
            with torch.no_grad():
                for X_seq, y in dl_valid:
                    X_seq = X_seq.to(device)
                    y = y.to(device)
                    if use_decoder:
                        y_hat, X_rec = model(X_seq, return_recon=True)
                        pred_loss = pred_loss_fn(y_hat, y)
                        recon_loss = recon_loss_fn(X_rec, X_seq)
                        loss = pred_loss + recon_weight * recon_loss
                        va_recon += float(recon_loss.item())
                    else:
                        y_hat = model(X_seq)
                        pred_loss = pred_loss_fn(y_hat, y)
                        loss = pred_loss
                    va_loss += float(loss.item())
                    va_pred += float(pred_loss.item())
            va_loss /= max(1, len(dl_valid))
            va_pred /= max(1, len(dl_valid))
            if use_decoder:
                va_recon /= max(1, len(dl_valid))
                print(
                    f"[ep {ep:03d}] "
                    f"train total {tr_loss:.6e} pred {tr_pred:.6e} recon {tr_recon:.6e} | "
                    f"valid total {va_loss:.6e} pred {va_pred:.6e} recon {va_recon:.6e}"
                )
            else:
                print(f"[ep {ep:03d}] train {tr_loss:.6e} | valid {va_loss:.6e}")
        else:
            if use_decoder:
                print(f"[ep {ep:03d}] train total {tr_loss:.6e} pred {tr_pred:.6e} recon {tr_recon:.6e}")
            else:
                print(f"[ep {ep:03d}] train {tr_loss:.6e}")

    # save model
    ckpt = {
        "model": model.state_dict(),
        "stats": stats,
        "L": L,
        "arch": {
            "d_m": d_m,
            "temporal_encoder": temporal_encoder,
            "t_hidden": t_hidden,
            "t_layers": t_layers,
            "t_kernel": t_kernel,
            "attn_heads": attn_heads,
            "attn_dropout": attn_dropout,
            "use_decoder": use_decoder,
            "recon_weight": recon_weight,
            "dec_hidden": dec_hidden,
            "dec_layers": dec_layers,
            "fno_modes": fno_modes,
            "fno_hidden": fno_hidden,
            "fno_layers": fno_layers,
        },
    }
    torch.save(ckpt, "simpleencode_closure_neuralop_fno_dt0.2.pth")
    print("saved: simpleencode_closure_neuralop_fno_dt0.2.pth")


if __name__ == "__main__":
    train(
        npz_train="../vlasov_single_data/A=0.1_k=0.35/moments_dt0.2.npz",
        npz_valid="../vlasov_single_data/A=0.1_k=0.35/moments_dt0.2.npz",
        L=64,
        batch_size=16,
        lr=2e-3,
        epochs=35,
        fno_modes=16,
        fno_hidden=64,
        fno_layers=4,
        temporal_encoder="attn",
        d_m=16,
        t_hidden=64,
        t_layers=2,
        t_kernel=5,
        attn_heads=4,
        attn_dropout=0.1,
        use_decoder=True,
        recon_weight=0.1,
        dec_hidden=128,
        dec_layers=2,
    )
