# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time
import math
import random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from numpy.lib.format import open_memmap

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CL_util_cities import set_seed, MultiCityCLDataset, MultiCityFullViewDataset
from CL_model_cities import MultiCityDualCL, info_nce_loss_masked


def _limit_cpu_threads():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _seed_worker(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


@dataclass
class CitySpec:
    name: str
    train_npy: str
    test_npy: str


@dataclass
class TrainConfig:
    cities: List[CitySpec] = field(default_factory=list)

    out_dir: str = "./cl_ckpt_normalize_train"
    emb_dir: str = "./embedding_normalize"

    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.1
    grad_clip: float = 1.0

    prepool_len: int = 64
    target_len: int = 0
    future_len: int = 4

    k_range: Tuple[int, int] = (1, 5)
    sub_len_range: Tuple[int, int] = (16, 64)
    st_neighbor: int = 2

    emb_dim: int = 128
    proj_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    conv_channels: int = 64
    dropout: float = 0.0
    node_len: int = 64
    node_dim: int = 16
    use_token_conv: bool = True
    cl_tanh: bool = True

    seed: int = 42
    num_workers: int = 8
    prefetch_factor: int = 4
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    use_amp: bool = True
    amp_dtype: str = "bf16"

    warmup_steps: int = 500
    log_every: int = 500

    force_fp32_nce: bool = True
    ce_floor_eps: float = 2e-3
    ce_floor_patience: int = 200
    lr_drop_factor: float = 0.5
    min_lr: float = 1e-6
    debug_on_collapse: bool = True

    shuffle_train: bool = True
    save_every: int = 10
    save_best: bool = True
    save_last: bool = True


def _make_joint_train_loader(cfg: TrainConfig) -> DataLoader:
    if not cfg.cities:
        raise ValueError("cfg.cities is empty")

    sources = [(c.name, c.train_npy) for c in cfg.cities]
    ds = MultiCityCLDataset(
        npy_path=sources,
        target_len=cfg.target_len,
        prepool_len=cfg.prepool_len,
        delta_range=cfg.k_range,
        sub_len_range=cfg.sub_len_range,
        shift_range=(-cfg.st_neighbor, cfg.st_neighbor),
        seed=cfg.seed,
        future_len=cfg.future_len,
        avoid_zero_shift=True,
    )

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    dl_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=bool(cfg.shuffle_train),
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=_seed_worker if cfg.num_workers > 0 else None,
        generator=g,
    )
    if cfg.num_workers > 0:
        dl_kwargs["prefetch_factor"] = int(cfg.prefetch_factor)

    return DataLoader(ds, **dl_kwargs)


def _make_export_loader(npy_path: str, cfg: TrainConfig) -> DataLoader:
    ds = MultiCityFullViewDataset(
        npy_path=npy_path,
        target_len=cfg.target_len,
        prepool_len=cfg.prepool_len,
        future_len=cfg.future_len,
    )

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    dl_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=_seed_worker if cfg.num_workers > 0 else None,
        generator=g,
    )
    if cfg.num_workers > 0:
        dl_kwargs["prefetch_factor"] = int(cfg.prefetch_factor)

    return DataLoader(ds, **dl_kwargs)


def _build_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _autocast_ctx(cfg: TrainConfig, device: torch.device):
    if (not cfg.use_amp) or (device.type != "cuda"):
        return torch.autocast(device_type="cpu", enabled=False)

    if cfg.amp_dtype.lower() == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


@torch.no_grad()
def _debug_pair_stats(zA: torch.Tensor, zP: torch.Tensor, name: str):
    A = zA.mean(dim=1).float()
    P = zP.mean(dim=1).float()
    A = torch.nn.functional.normalize(A, dim=-1)
    P = torch.nn.functional.normalize(P, dim=-1)
    sim = A @ P.t()
    diag = sim.diag()
    off = sim[~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)]

    z_std = float(zA.float().std().item())
    diag_m = float(diag.mean().item())
    diag_min = float(diag.min().item())
    diag_max = float(diag.max().item())
    off_m = float(off.mean().item())
    off_std = float(off.std().item())

    print(
        f"[DBG][{name}] z_std={z_std:.4e} | "
        f"diag_cos mean/min/max={diag_m:.4f}/{diag_min:.4f}/{diag_max:.4f} | "
        f"off_cos mean/std={off_m:.4f}/{off_std:.4f}"
    )


def _maybe_drop_lr(optimizer: torch.optim.Optimizer, factor: float, min_lr: float) -> bool:
    dropped = False
    for pg in optimizer.param_groups:
        old = float(pg["lr"])
        new = max(min_lr, old * factor)
        if new < old - 1e-12:
            pg["lr"] = new
            dropped = True
    return dropped


def _train_one_epoch(
    model: MultiCityDualCL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
    global_step: int,
) -> Tuple[float, int]:
    model.train()
    running = 0.0
    total_batches = len(loader)

    scaler = None
    use_fp16 = cfg.use_amp and (device.type == "cuda") and (cfg.amp_dtype.lower() == "fp16")
    if use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    floor_cnt_d = 0
    floor_cnt_s = 0

    t0 = time.time()
    print(f"\n[Train] Epoch {epoch} — total {total_batches} batches")

    for bi, batch in enumerate(loader):
        dA = batch["density_anchor"].to(device, non_blocking=True)
        dP = batch["density_pos"].to(device, non_blocking=True)
        sA = batch["st_anchor"].to(device, non_blocking=True)
        sP = batch["st_pos"].to(device, non_blocking=True)

        ids = batch["traj_index"].to(device, non_blocking=True)

        B_now = int(ids.shape[0])
        ce_floor = math.log(max(2, B_now))

        use_pad_mask = (cfg.prepool_len is None) or (int(cfg.prepool_len) <= 0)
        if use_pad_mask:
            m_dA = batch["mask_density_anchor"].to(device, non_blocking=True)
            m_dP = batch["mask_density_pos"].to(device, non_blocking=True)
            m_sA = batch["mask_st_anchor"].to(device, non_blocking=True)
            m_sP = batch["mask_st_pos"].to(device, non_blocking=True)
        else:
            m_dA = m_dP = m_sA = m_sP = None

        optimizer.zero_grad(set_to_none=True)

        with _autocast_ctx(cfg, device):
            z_dA = model(dA, pad_mask=m_dA, mode="traj")
            z_dP = model(dP, pad_mask=m_dP, mode="traj")
            z_sA = model(sA, pad_mask=m_sA, mode="subtraj")
            z_sP = model(sP, pad_mask=m_sP, mode="subtraj")

        if cfg.force_fp32_nce:
            z_dA_l = z_dA.float()
            z_dP_l = z_dP.float()
            z_sA_l = z_sA.float()
            z_sP_l = z_sP.float()
        else:
            z_dA_l, z_dP_l, z_sA_l, z_sP_l = z_dA, z_dP, z_sA, z_sP

        loss_d = info_nce_loss_masked(z_dA_l, z_dP_l, ids, ids, temperature=cfg.temperature)
        loss_s = info_nce_loss_masked(z_sA_l, z_sP_l, ids, ids, temperature=cfg.temperature)
        loss = loss_d + loss_s

        if not torch.isfinite(loss):
            print(f"[WARN] non-finite loss at ep{epoch} b{bi} -> skip step. loss={loss}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if abs(float(loss_d.item()) - ce_floor) < cfg.ce_floor_eps:
            floor_cnt_d += 1
        else:
            floor_cnt_d = 0

        if abs(float(loss_s.item()) - ce_floor) < cfg.ce_floor_eps:
            floor_cnt_s += 1
        else:
            floor_cnt_s = 0

        if (floor_cnt_d >= cfg.ce_floor_patience) or (floor_cnt_s >= cfg.ce_floor_patience):
            which = "density" if floor_cnt_d >= cfg.ce_floor_patience else "st"
            dropped = _maybe_drop_lr(optimizer, cfg.lr_drop_factor, cfg.min_lr)
            if dropped:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"[WARN] LR dropped -> {lr_now:.3e}")
            floor_cnt_d = 0
            floor_cnt_s = 0

            if cfg.debug_on_collapse:
                try:
                    if which == "density":
                        _debug_pair_stats(z_dA_l, z_dP_l, "density")
                    else:
                        _debug_pair_stats(z_sA_l, z_sP_l, "st")
                except Exception as e:
                    print(f"[DBG] collapse debug failed: {e}")

        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if not torch.isfinite(gnorm):
                    print(f"[WARN] non-finite grad norm at ep{epoch} b{bi} -> skip step.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if not torch.isfinite(gnorm):
                    print(f"[WARN] non-finite grad norm at ep{epoch} b{bi} -> skip step.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running += float(loss.item())
        global_step += 1

        if (bi % max(1, cfg.log_every) == 0) or (bi == total_batches - 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[Train][ep{epoch:03d} b{bi:04d}/{total_batches}] "
                f"loss={loss.item():.4f} (d={loss_d.item():.4f}, s={loss_s.item():.4f}) lr={lr_now:.3e}"
            )

    dt = time.time() - t0
    avg_loss = running / max(1, total_batches)
    print(f"[Train] Epoch {epoch} finished — avg loss={avg_loss:.4f}  time={dt:.1f}s")
    return avg_loss, global_step


@torch.no_grad()
def _export_embeddings(
    npy_path: str,
    city: str,
    tag: str,
    cfg: TrainConfig,
    model: MultiCityDualCL,
    device: torch.device,
):
    os.makedirs(cfg.emb_dir, exist_ok=True)
    loader = _make_export_loader(npy_path, cfg)
    model.eval()

    n_total = len(loader.dataset)
    out_path = os.path.join(cfg.emb_dir, f"{city}_{tag}_embed_normalize.npy")

    mm = open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_total, cfg.node_len, cfg.node_dim),
    )

    total_elems = 0
    sum_x = 0.0
    sum_x2 = 0.0
    global_min = float("inf")
    global_max = float("-inf")

    n_samples = 0
    sum_norm = 0.0
    min_norm = float("inf")
    max_norm = float("-inf")

    n_tokens = 0
    sum_tok_norm = 0.0
    min_tok_norm = float("inf")
    max_tok_norm = float("-inf")

    write_ptr = 0

    for batch in loader:
        x = batch["full"].to(device, non_blocking=True)
        use_pad_mask = (cfg.prepool_len is None) or (int(cfg.prepool_len) <= 0)
        m = batch["mask"].to(device, non_blocking=True) if use_pad_mask else None

        node_seq = model.encode_node_embedding(x, pad_mask=m)
        node_np = node_seq.float().cpu().numpy()

        bsz = node_np.shape[0]
        mm[write_ptr:write_ptr + bsz] = node_np
        write_ptr += bsz

        bmin = float(node_seq.min().item())
        bmax = float(node_seq.max().item())
        global_min = min(global_min, bmin)
        global_max = max(global_max, bmax)

        total_elems += int(node_seq.numel())
        sum_x += float(node_seq.sum().item())
        sum_x2 += float((node_seq * node_seq).sum().item())

        flat = node_seq.reshape(node_seq.size(0), -1)
        norms = torch.linalg.norm(flat, dim=1)
        n_samples += int(norms.numel())
        sum_norm += float(norms.sum().item())
        min_norm = min(min_norm, float(norms.min().item()))
        max_norm = max(max_norm, float(norms.max().item()))

        tok_norms = torch.linalg.norm(node_seq, dim=-1)
        n_tokens += int(tok_norms.numel())
        sum_tok_norm += float(tok_norms.sum().item())
        min_tok_norm = min(min_tok_norm, float(tok_norms.min().item()))
        max_tok_norm = max(max_tok_norm, float(tok_norms.max().item()))

    if write_ptr != n_total:
        raise RuntimeError(f"export write mismatch: write_ptr={write_ptr}, n_total={n_total}")
    del mm

    mean_x = sum_x / max(1, total_elems)
    var_x = (sum_x2 / max(1, total_elems)) - (mean_x * mean_x)
    std_x = float(np.sqrt(max(var_x, 0.0)))

    mean_norm = sum_norm / max(1, n_samples)
    mean_tok_norm = sum_tok_norm / max(1, n_tokens)

    print(f"[Export] {city}:{tag} → {out_path}  shape=({n_total},{cfg.node_len},{cfg.node_dim})")
    print(f"[ExportStats] value min/max = ({global_min:.6f}, {global_max:.6f}), mean={mean_x:.6f}, std={std_x:.6f}")
    print(f"[ExportStats] sample L2 norm (flatten) mean/min/max = ({mean_norm:.6f}, {min_norm:.6f}, {max_norm:.6f})")
    print(f"[ExportStats] token  L2 norm (per node) mean/min/max = ({mean_tok_norm:.6f}, {min_tok_norm:.6f}, {max_tok_norm:.6f})")


def _train_joint_and_export(cfg: TrainConfig):
    if not cfg.cities:
        raise ValueError("cfg.cities is empty")

    device = torch.device(cfg.device)
    _limit_cpu_threads()
    set_seed(cfg.seed)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    joint_ckpt_dir = os.path.join(cfg.out_dir, "joint_all_cities")
    os.makedirs(joint_ckpt_dir, exist_ok=True)

    model = MultiCityDualCL(
        in_feat=3,
        emb_dim=cfg.emb_dim,
        proj_dim=cfg.proj_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        conv_channels=cfg.conv_channels,
        dropout=cfg.dropout,
        node_len=cfg.node_len,
        node_dim=cfg.node_dim,
        use_token_conv=cfg.use_token_conv,
        cl_tanh=cfg.cl_tanh,
    ).to(device)

    opt_kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    if device.type == "cuda":
        opt_kwargs["fused"] = True

    try:
        optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = _make_joint_train_loader(cfg)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = _build_lr_scheduler(optimizer, total_steps=total_steps, warmup_steps=cfg.warmup_steps)

    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    global_step = 0
    loss_tr = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        loss_tr, global_step = _train_one_epoch(
            model, train_loader, optimizer, scheduler, device, cfg, epoch, global_step
        )

        if cfg.save_best and loss_tr < best_loss:
            best_loss = loss_tr
            best_epoch = epoch
            best_state = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "train_loss": loss_tr,
            }

        if cfg.save_every > 0 and (epoch % int(cfg.save_every) == 0):
            ckpt_path = os.path.join(joint_ckpt_dir, f"joint_normalize_epoch{epoch:03d}.pt")
            torch.save(
                {"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch, "train_loss": loss_tr},
                ckpt_path,
            )
            print(f"  -> saved: {ckpt_path}")

    if cfg.save_best and best_state is not None:
        best_path = os.path.join(joint_ckpt_dir, f"joint_best_epoch{best_epoch:03d}.pt")
        torch.save(best_state, best_path)
        print(f"  -> saved BEST ({best_loss:.4f}) at epoch {best_epoch}: {best_path}")

    if cfg.save_last:
        final_path = os.path.join(joint_ckpt_dir, "joint_final.pt")
        torch.save(
            {"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": cfg.epochs, "train_loss": loss_tr},
            final_path,
        )
        print(f"  -> saved FINAL: {final_path}")

    for spec in cfg.cities:
        _export_embeddings(spec.train_npy, spec.name, "train", cfg, model, device)
        _export_embeddings(spec.test_npy, spec.name, "test", cfg, model, device)


def main(cfg: Optional[TrainConfig] = None):
    cfg = cfg or TrainConfig()
    _train_joint_and_export(cfg)


if __name__ == "__main__":

    cfg = TrainConfig(
        # Please change the data path to run the code.
        cities=[
            CitySpec("porto", "./",
                     "./"),
            CitySpec("BJ", "./",
                     "./"),
            CitySpec("CD", "./",
                     "./"),
            CitySpec("XA", "./",
                     "./"),
        ],
        st_neighbor=2,
        k_range=(2, 2),
        target_len=0,
        future_len=4,
        epochs=100,
        node_len=64,
        node_dim=16,
        prepool_len=64,
        batch_size=128,
        lr=1e-4,
        use_amp=True,
        amp_dtype="bf16",
        log_every=500,
        num_workers=8,
        prefetch_factor=4,
        force_fp32_nce=True,
        save_every=10,
    )
    main(cfg)
