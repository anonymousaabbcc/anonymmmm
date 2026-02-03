from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_xy_t_mask(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected x dim=3 (B,T,F), got {tuple(x.shape)}")
    fdim = int(x.size(-1))
    if fdim == 4:
        return x
    if fdim == 3:
        B, T, _ = x.shape
        out = x.new_zeros((B, T, 4))
        out[..., 0:2] = x[..., 0:2]
        out[..., 2] = 0.0
        out[..., 3] = x[..., 2]
        return out
    raise ValueError(f"Expected last dim 3 or 4, got {fdim} with shape {tuple(x.shape)}")


def segment_pool_traj_to_fixed_len(
    x: torch.Tensor,
    pad_mask: Optional[torch.Tensor],
    target_len: int,
) -> torch.Tensor:
    x = _ensure_xy_t_mask(x)
    assert x.dim() == 3 and x.size(-1) == 4, f"Expected (B,T,4), got {tuple(x.shape)}"

    B, T, _ = x.shape
    device = x.device
    L = int(target_len)
    if L <= 0:
        raise ValueError("target_len must be positive")

    if pad_mask is None:
        pad_mask_b = (x[..., 3] > 0.5)
    else:
        pad_mask_b = pad_mask.to(dtype=torch.bool)

    out = x.new_zeros((B, L, 4))
    out[..., 3] = 1.0

    x_f = x.float()

    for i in range(B):
        valid = pad_mask_b[i]
        Ti = int(valid.sum().item())
        if Ti <= 0:
            xyz0 = x_f[i, 0:1, 0:3] if T > 0 else x_f.new_zeros((1, 3))
            out[i, :, 0:3] = xyz0.to(out.dtype).expand(L, 3)
            continue

        xi = x_f[i, valid, 0:3]
        if Ti == 1:
            out[i, :, 0:3] = xi.to(out.dtype).expand(L, 3)
            continue

        j = torch.arange(Ti, device=device, dtype=torch.long)
        bin_idx = (j * L) // Ti

        sums = torch.zeros((L, 3), device=device, dtype=torch.float32)
        counts = torch.zeros((L,), device=device, dtype=torch.float32)

        sums.index_add_(0, bin_idx, xi)
        counts.index_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=torch.float32, device=device))

        means = sums / counts.clamp_min(1.0).unsqueeze(-1)
        nonempty = counts > 0

        if not bool(nonempty.all().item()):
            last = None
            for k in range(L):
                if bool(nonempty[k].item()):
                    last = means[k].clone()
                elif last is not None:
                    means[k] = last

            if not bool(nonempty[0].item()):
                first = None
                for k in range(L):
                    if bool(nonempty[k].item()):
                        first = means[k].clone()
                        break
                if first is None:
                    first = xi[0]
                for k in range(L):
                    if bool(nonempty[k].item()):
                        break
                    means[k] = first

        out[i, :, 0:3] = means.to(out.dtype)

    return out


class PosEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :].to(dtype=x.dtype)


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        in_feat: int = 4,
        emb_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        conv_channels: int = 64,
        dropout: float = 0.0,
        max_len: int = 4096,
        node_len: int = 64,
    ):
        super().__init__()
        if int(in_feat) not in (3, 4):
            raise ValueError("TrajectoryEncoder expects in_feat=4 or 3")

        self.node_len = int(node_len)
        eff_feat = 4

        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=(3, eff_feat),
            padding=(1, 0),
            bias=True,
        )
        self.norm2d = nn.GroupNorm(num_groups=1, num_channels=conv_channels)
        self.act = nn.GELU()
        self.drop2d = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.proj = nn.Linear(conv_channels, emb_dim, bias=True)
        self.ln = nn.LayerNorm(emb_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.trans = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pe = PosEncoding(emb_dim, max_len=max_len)

        nn.init.kaiming_uniform_(self.conv2d.weight, a=math.sqrt(5))
        if self.conv2d.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv2d.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.conv2d.bias, -bound, bound)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = _ensure_xy_t_mask(x)
        key_pad = None

        if x.size(1) != self.node_len:
            x = segment_pool_traj_to_fixed_len(x, pad_mask=pad_mask, target_len=self.node_len)
            key_pad = None
        else:
            if pad_mask is not None:
                pm = pad_mask.to(dtype=torch.bool)
                key_pad = ~pm
                x = x.clone()
                x[..., 3] = pm.to(dtype=x.dtype)
            else:
                key_pad = None
                x = x.clone()
                x[..., 3] = 1.0

        B, T, Fdim = x.shape
        assert T == self.node_len and Fdim == 4

        x = x.unsqueeze(1)
        x = self.conv2d(x)
        x = self.norm2d(x)
        x = self.act(x)
        x = self.drop2d(x)

        x = x.squeeze(-1).transpose(1, 2)
        x = self.proj(x)
        x = self.ln(x)

        x = self.pe(x)
        x = self.trans(x, src_key_padding_mask=key_pad)
        return x


class TokenConvBlock(nn.Module):
    def __init__(self, dim: int, kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        k = int(kernel)
        pad = (k - 1) // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size=k, padding=pad, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.drop(self.act(h))
        return self.ln(x + h)


class NodeProjector(nn.Module):
    def __init__(
        self,
        emb_dim: int = 128,
        node_dim: int = 16,
        node_len: int = 64,
        dropout: float = 0.0,
        use_token_conv: bool = True,
    ):
        super().__init__()
        self.node_len = int(node_len)
        self.node_dim = int(node_dim)
        self.proj = nn.Linear(emb_dim, node_dim, bias=True)
        self.post = TokenConvBlock(node_dim, kernel=3, dropout=dropout) if use_token_conv else nn.Identity()
        self.out_ln = nn.LayerNorm(node_dim)

    def forward(self, seq_feat: torch.Tensor) -> torch.Tensor:
        B, T, _ = seq_feat.shape
        assert T == self.node_len, f"Expected T=={self.node_len}, got {T}"
        x = self.proj(seq_feat)
        x = self.post(x)
        x = self.out_ln(x)
        return x


class MultiCityDualCL(nn.Module):
    def __init__(
        self,
        in_feat: int = 4,
        emb_dim: int = 128,
        proj_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        conv_channels: int = 64,
        dropout: float = 0.0,
        node_len: int = 64,
        node_dim: int = 16,
        use_token_conv: bool = True,
        cl_tanh: bool = True,
        cl_tanh_scale: float = 2.0,
        cl_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if int(in_feat) not in (3, 4):
            raise ValueError("MultiCityDualCL expects in_feat=4 or 3.")
        self.node_len = int(node_len)
        self.node_dim = int(node_dim)
        self._cl_norm_eps = float(cl_norm_eps)
        self._cl_tanh = bool(cl_tanh)
        self._cl_tanh_scale = float(cl_tanh_scale)

        self.traj_encoder = TrajectoryEncoder(
            in_feat=in_feat,
            emb_dim=emb_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            conv_channels=conv_channels,
            dropout=dropout,
            node_len=self.node_len,
        )
        self.subtraj_encoder = TrajectoryEncoder(
            in_feat=in_feat,
            emb_dim=emb_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            conv_channels=conv_channels,
            dropout=dropout,
            node_len=self.node_len,
        )

        self.node_projector = NodeProjector(
            emb_dim=emb_dim,
            node_dim=self.node_dim,
            node_len=self.node_len,
            dropout=dropout,
            use_token_conv=use_token_conv,
        )
        self.proj_dim = int(proj_dim)

    def encode_traj(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_feat = self.traj_encoder(x, pad_mask=pad_mask)
        node_seq = self.node_projector(seq_feat)
        return node_seq

    def encode_subtraj(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_feat = self.subtraj_encoder(x, pad_mask=pad_mask)
        node_seq = self.node_projector(seq_feat)
        return node_seq

    def project_cl(self, node_seq: torch.Tensor) -> torch.Tensor:
        B, T, D = node_seq.shape
        assert T == self.node_len and D == self.node_dim, f"Expected (B,{self.node_len},{self.node_dim}), got {tuple(node_seq.shape)}"

        if self._cl_tanh:
            s = max(self._cl_tanh_scale, 1e-6)
            return torch.tanh(node_seq / s)
        return node_seq

    @torch.no_grad()
    def encode_node_embedding(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        node_seq = self.encode_traj(x, pad_mask=pad_mask)
        return self.project_cl(node_seq)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        mode: str = "traj",
        return_seq: bool = False,
    ) -> torch.Tensor:
        if mode == "traj":
            node_seq = self.encode_traj(x, pad_mask)
        elif mode == "subtraj":
            node_seq = self.encode_subtraj(x, pad_mask)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if return_seq:
            return node_seq
        return self.project_cl(node_seq)


def info_nce_loss_masked(
    z1: torch.Tensor,
    z2: torch.Tensor,
    ids1: torch.Tensor,
    ids2: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    assert z1.dim() in (2, 3) and z2.dim() in (2, 3), "z must be (B,D) or (B,T,D)"

    eps_eff = max(float(eps), 1e-6)
    temp = max(float(temperature), eps_eff)

    if z1.dim() == 3:
        B = z1.size(0)
        z1 = z1.reshape(B, -1)
        z2 = z2.reshape(B, -1)

    z1 = z1.float()
    z2 = z2.float()

    z1 = F.normalize(z1, dim=-1, eps=eps_eff)
    z2 = F.normalize(z2, dim=-1, eps=eps_eff)

    logits = (z1 @ z2.t()) / temp
    B = logits.size(0)
    labels = torch.arange(B, device=logits.device)

    same = (ids1[:, None] == ids2[None, :])
    eye = torch.eye(B, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same & (~eye), float("-inf"))

    row_ok = torch.isfinite(logits).any(dim=1)
    if not row_ok.all():
        bad = ~row_ok
        logits[bad, :] = -float("inf")
        logits[bad, labels[bad]] = 0.0

    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    return loss
