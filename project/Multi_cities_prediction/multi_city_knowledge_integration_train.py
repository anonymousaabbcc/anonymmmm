from __future__ import annotations

import os
import time
import math
import random
import re
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from project.Graph_based_prompt_align import PromptAlignedGraphSample, MODEL_NAME as STAGE2_MODEL_NAME
from project.Multi_cities_prediction.multi_city_knowledge_integration_model import MultiCityKnowledgeIntegrationModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

@dataclass
class CitySpec:
    name: str
    train_npy: str
    test_npy: str

@dataclass
class MultiCityTrainingConfig:
    cities: List[CitySpec] = field(default_factory=list)

    graph_dir: str = ""
    train_tag: str = "train"
    test_tag: str = "test"
    graph_cache_dir: str = ""
    use_graph_cache: bool = True
    cache_suffix: str = "stage3cache"

    ckpt_every_epochs: int = 1
    ckpt_save_last: bool = True

    llm_model_name: str = STAGE2_MODEL_NAME
    d_traj: int = 16
    hidden_proj: int = 0
    pred_len: int = 4
    out_dim: int = 2

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    graph_dropout: float = 0.10
    semantic_tau: float = 1.0
    tanh_scale: float = 1.0
    head_conv_kernel: int = 3
    head_conv_layers: int = 2
    head_dropout: float = 0.10

    batch_size: int = 64
    epochs: int = 15
    num_workers: int = 0
    seed: int = 2025

    future_len: int = 4
    val_ratio: float = 0.10

    train_ratio: float = 1.0
    train_cap: int = 0
    test_ratio: float = 1.0
    test_cap: int = 0

    city_round_shuffle: bool = False
    rounds_per_epoch: int = 0

    city_block_min_batches: int = 2
    city_block_max_batches: int = 10

    city_quota_mix: float = 0.6
    city_quota_min_factor: float = 0.7
    city_quota_max_factor: float = 1.6

    lr_lora: float = 1e-4
    lr_head: float = 5e-4
    lr_graph: float = 1e-4
    wd_lora: float = 0.0
    wd_head: float = 1e-4
    wd_graph: float = 1e-2

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    use_scheduler: bool = True
    warmup_ratio: float = 0.0
    min_lr_scale: float = 0.10

    grad_clip_lora: float = 1.0
    grad_clip_head: float = 5.0
    grad_clip_graph: float = 1.0

    mr_threshold: float = 0.2
    lambda_pred_l2: float = 2e-3

    lambda_orth: float = 1.0
    pecl_dir_ema_beta: float = 0.8
    dir_update_every: int = 16

    pecl_constrain_lora: bool = True
    pecl_constrain_head: bool = False
    pecl_constrain_graph: bool = False

    freeze_graph: bool = False
    force_fp32_trainables: bool = True

    graph_step_freeze: int = 0

    warm_start_head_last_linear: bool = True
    head_last_init_std: float = 1e-3

    log_every: int = 100

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir: str = "./models"

def set_seed(seed: int):
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)

def summarize_trainables(model: nn.Module) -> Dict[str, int]:
    counts = {"lora": 0, "graph": 0, "head": 0, "other": 0}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            counts["lora"] += p.numel()
        elif n.startswith("graph_encoder."):
            counts["graph"] += p.numel()
        elif n.startswith("reg_head."):
            counts["head"] += p.numel()
        else:
            counts["other"] += p.numel()
    return counts

def cosine_schedule_scale(step: int, total_steps: int, warmup_steps: int, min_scale: float) -> float:
    if total_steps <= 0:
        return 1.0
    step = int(max(0, min(int(step), int(total_steps) - 1)))
    warmup_steps = int(max(0, int(warmup_steps)))
    min_scale = float(max(0.0, min(float(min_scale), 1.0)))
    if warmup_steps > 0 and step < warmup_steps:
        w = float(step) / float(max(1, warmup_steps))
        return min_scale + (1.0 - min_scale) * w
    if total_steps - warmup_steps <= 1:
        return 1.0
    progress = float(step - warmup_steps) / float(max(1, (total_steps - warmup_steps - 1)))
    c = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_scale + (1.0 - min_scale) * c

def _pick_count(total: int, ratio: float, cap: int) -> int:
    total = int(total)
    if total <= 0:
        return 0
    ratio = float(min(max(ratio, 0.0), 1.0))
    n = int(math.floor(total * ratio + 1e-9))
    if int(cap) > 0:
        n = min(n, int(cap))
    n = max(1, min(n, total))
    return n

def grads_all_finite(params: List[torch.nn.Parameter]) -> bool:
    for p in params:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True

def _force_fp32_trainables(model: nn.Module):
    changed = 0
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.is_floating_point() and p.dtype != torch.float32:
            p.data = p.data.float()
            changed += 1
    logger.info(f"[FP32] trainable tensors cast to fp32: {changed}")

def _warm_start_head_last_linear_if_zero(model: nn.Module, std: float = 1e-3) -> bool:
    head = getattr(model, "reg_head", None)
    if head is None:
        return False
    mlp = getattr(head, "mlp", None)
    if not isinstance(mlp, nn.Sequential) or len(mlp) < 1:
        return False
    last = mlp[-1]
    if not isinstance(last, nn.Linear):
        return False
    with torch.no_grad():
        w = last.weight
        b = last.bias
        if w is None:
            return False
        is_w_zero = (torch.count_nonzero(w).item() == 0)
        is_b_zero = (b is None) or (torch.count_nonzero(b).item() == 0)
        if is_w_zero and is_b_zero:
            nn.init.normal_(w, mean=0.0, std=float(std))
            if b is not None:
                nn.init.zeros_(b)
            logger.info(f"[HeadInit] reg_head.mlp[-1] was all-zero; warm-started with N(0,{std}).")
            return True
    return False

def compute_city_block_batches_per_round(
    active_cities: List[str],
    city_train_sizes: Dict[str, int],
    min_batches: int = 2,
    max_batches: int = 10,
) -> Dict[str, int]:
    assert len(active_cities) > 0
    min_batches = int(min_batches)
    max_batches = int(max_batches)
    assert min_batches >= 1 and max_batches >= min_batches
    sizes = np.array([max(1, int(city_train_sizes.get(c, 1))) for c in active_cities], dtype=np.float64)
    p = sizes / max(1e-12, sizes.sum())
    p_max = float(p.max())
    scale = float(max_batches) / max(1e-12, p_max)
    out: Dict[str, int] = {}
    for i, c in enumerate(active_cities):
        nums = int(np.rint(p[i] * scale))
        nums = max(min_batches, min(max_batches, nums))
        out[c] = int(nums)
    return out

def torch_load_compat(path: str):
    def _load(weights_only: Optional[bool]):
        if weights_only is None:
            return torch.load(path, map_location="cpu")
        try:
            return torch.load(path, map_location="cpu", weights_only=weights_only)
        except TypeError:
            return torch.load(path, map_location="cpu")
    try:
        return _load(weights_only=False)
    except Exception as e:
        msg = str(e)
        if ("__main__" in msg) or ("PromptAlignedGraphSample" in msg) or ("CitySpec" in msg) or ("MultiCityTrainingConfig" in msg):
            import __main__ as _main
            _main.PromptAlignedGraphSample = PromptAlignedGraphSample
            _main.CitySpec = CitySpec
            _main.MultiCityTrainingConfig = MultiCityTrainingConfig
            return _load(weights_only=False)
        raise

def extract_true_future_and_histlen_from_npy(
    npy_path: str,
    future_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    data = np.load(npy_path)
    assert data.ndim == 3 and data.shape[2] == 4, f"bad npy shape: {data.shape}"
    N, L, _ = data.shape
    hist_lens_all = np.zeros((N,), dtype=np.int32)
    keep_mask = np.zeros((N,), dtype=np.bool_)
    futures_list = []
    stats = {"N_all": N, "valid": 0, "short": 0}
    for i in range(N):
        traj = data[i]
        m = (traj[:, 3].astype(np.int32) == 1)
        idx = np.where(m)[0]
        T_real = int(len(idx))
        if T_real <= 1:
            stats["short"] += 1
            continue
        Tf = min(int(future_len), T_real - 1)
        Th = T_real - Tf
        hist_lens_all[i] = Th
        if Tf < int(pred_len) or Th < int(pred_len):
            stats["short"] += 1
            continue
        fut_idx = idx[T_real - Tf: T_real]
        futures_list.append(traj[fut_idx[:pred_len], :2].astype(np.float32))
        keep_mask[i] = True
    futures_keep = np.stack(futures_list, axis=0) if futures_list else np.zeros((0, pred_len, 2), dtype=np.float32)
    stats["valid"] = int(len(futures_list))
    return futures_keep, hist_lens_all, keep_mask, stats

def load_or_create_graphs(city: str, tag: str, cfg: MultiCityTrainingConfig) -> List[PromptAlignedGraphSample]:
    if cfg.graph_cache_dir:
        os.makedirs(cfg.graph_cache_dir, exist_ok=True)
        cache_path = os.path.join(cfg.graph_cache_dir, f"{city}_{tag}_graphs_{cfg.cache_suffix}.pt")
    else:
        cache_path = ""
    stage2_path = os.path.join(cfg.graph_dir, f"{city}_{tag}_graphs.pt")
    if cfg.use_graph_cache and cache_path and os.path.exists(cache_path):
        return torch_load_compat(cache_path)
    if not os.path.exists(stage2_path):
        raise FileNotFoundError(f"Stage2 graphs not found: {stage2_path}")
    graphs = torch_load_compat(stage2_path)
    if cfg.use_graph_cache and cache_path:
        torch.save(graphs, cache_path)
    return graphs

def maybe_attach_traj_valid_len(graphs: List[PromptAlignedGraphSample], hist_lens_for_graphs: np.ndarray):
    if not graphs:
        return
    if hasattr(graphs[0], "traj_valid_len"):
        return
    assert len(graphs) == int(hist_lens_for_graphs.shape[0]), (len(graphs), hist_lens_for_graphs.shape)
    for i, g in enumerate(graphs):
        T_full = int(g.traj_tokens.size(0)) if hasattr(g, "traj_tokens") else 0
        g.traj_valid_len = max(0, min(int(hist_lens_for_graphs[i]), T_full))

def align_graphs_and_targets(
    graphs_all: List[PromptAlignedGraphSample],
    futures_keep: np.ndarray,
    keep_mask: np.ndarray,
    stats: Dict[str, int],
) -> Tuple[List[PromptAlignedGraphSample], np.ndarray]:
    N_graph = len(graphs_all)
    N_all = int(stats["N_all"])
    N_valid = int(stats["valid"])
    keep_indices = np.where(keep_mask)[0]
    if N_graph == N_all:
        graphs_keep = [graphs_all[i] for i in keep_indices]
        if len(graphs_keep) != futures_keep.shape[0]:
            raise RuntimeError(f"[Align] mismatch after filtering: graphs_keep={len(graphs_keep)} vs futures_keep={futures_keep.shape[0]}")
        return graphs_keep, futures_keep
    if N_graph == N_valid:
        if N_graph != futures_keep.shape[0]:
            raise RuntimeError(f"[Align] mismatch: graphs_all={N_graph} vs futures_keep={futures_keep.shape[0]}")
        logger.info("[Align][WARN] Stage2 graphs appear already filtered (N_graph == N_valid). Using graphs as-is.")
        return graphs_all, futures_keep
    raise RuntimeError(f"[Align] Cannot align graphs and targets: N_graph={N_graph}, N_all={N_all}, N_valid={N_valid}.")

class CityTrajDataset(Dataset):
    def __init__(self, graphs: List[PromptAlignedGraphSample], futures: np.ndarray):
        assert len(graphs) == futures.shape[0], (len(graphs), futures.shape)
        self.graphs = graphs
        self.futures = futures
    def __len__(self) -> int:
        return len(self.graphs)
    def __getitem__(self, idx: int):
        return self.graphs[idx], self.futures[idx]

def collate_fn(batch):
    graphs, futures = zip(*batch)
    return list(graphs), np.stack(futures, axis=0).astype(np.float32)

def compute_ade_fde_mr(pred: np.ndarray, gt: np.ndarray, mr_threshold: float) -> Tuple[float, float, float]:
    disp = np.linalg.norm(pred - gt, axis=-1)
    ade = float(disp.mean())
    fde = float(disp[:, -1].mean())
    mr = float((disp[:, -1] > float(mr_threshold)).mean())
    return ade, fde, mr

def is_constrained_param(name: str, cfg: MultiCityTrainingConfig) -> bool:
    if cfg.pecl_constrain_lora and ("lora_" in name):
        return True
    if cfg.pecl_constrain_head and name.startswith("reg_head."):
        return True
    if cfg.pecl_constrain_graph and name.startswith("graph_encoder."):
        return True
    return False

def _base_lr_for_param(name: str, cfg: MultiCityTrainingConfig) -> float:
    if "lora_" in name:
        return float(cfg.lr_lora)
    if name.startswith("reg_head."):
        return float(cfg.lr_head)
    if name.startswith("graph_encoder."):
        return float(cfg.lr_graph)
    return float(cfg.lr_lora)

def collect_lora_params(model: nn.Module) -> List[torch.nn.Parameter]:
    ps = []
    for n, p in model.named_parameters():
        if p.requires_grad and ("lora_" in n):
            ps.append(p)
    return ps

def collect_head_params(model: nn.Module) -> List[torch.nn.Parameter]:
    ps = []
    for n, p in model.named_parameters():
        if p.requires_grad and n.startswith("reg_head."):
            ps.append(p)
    return ps

def collect_graph_params(model: nn.Module) -> List[torch.nn.Parameter]:
    ps = []
    for n, p in model.named_parameters():
        if p.requires_grad and n.startswith("graph_encoder."):
            ps.append(p)
    return ps

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)")

def _infer_block_key(name: str) -> Tuple[str, str]:
    if "lora_" in name:
        m = _LAYER_RE.search(name)
        if m is not None:
            return f"layer{int(m.group(1))}", "lora"
        return "lora_misc", "lora"
    if name.startswith("reg_head."):
        return "head", "head"
    if name.startswith("graph_encoder."):
        return "graph", "graph"
    return "other", "other"

@dataclass
class ConstrainedBlock:
    key: str
    group: str
    names: List[str] = field(default_factory=list)
    params: List[torch.nn.Parameter] = field(default_factory=list)
    sizes: List[int] = field(default_factory=list)
    base_lrs: List[float] = field(default_factory=list)
    dim: int = 0
    lr_cat_base: Optional[torch.Tensor] = None

def build_lr_cat_base(sizes: List[int], base_lrs: List[float], device: torch.device) -> torch.Tensor:
    parts = []
    for sz, lr in zip(sizes, base_lrs):
        parts.append(torch.full((int(sz),), float(lr), device=device, dtype=torch.float32))
    return torch.cat(parts, dim=0)

def collect_constrained_blocks(model: nn.Module, cfg: MultiCityTrainingConfig, device: torch.device) -> Dict[str, ConstrainedBlock]:
    blocks: Dict[str, ConstrainedBlock] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not is_constrained_param(n, cfg):
            continue
        bkey, grp = _infer_block_key(n)
        if bkey not in blocks:
            blocks[bkey] = ConstrainedBlock(key=bkey, group=grp)
        blk = blocks[bkey]
        blk.names.append(n)
        blk.params.append(p)
        blk.sizes.append(int(p.numel()))
        blk.base_lrs.append(_base_lr_for_param(n, cfg))
    if len(blocks) == 0:
        raise RuntimeError("No constrained params found. Expected at least LoRA to be constrained.")
    for bkey, blk in blocks.items():
        blk.dim = int(sum(blk.sizes))
        blk.lr_cat_base = build_lr_cat_base(blk.sizes, blk.base_lrs, device=device)
    return blocks

def build_grad_cat_from_params(params: List[torch.nn.Parameter], sizes: List[int]) -> torch.Tensor:
    parts = []
    for p, sz in zip(params, sizes):
        if p.grad is None:
            parts.append(torch.zeros((sz,), device=p.device, dtype=torch.float32))
        else:
            parts.append(p.grad.detach().float().contiguous().view(-1))
    return torch.cat(parts, dim=0)

def apply_delta_cat_to_params(params: List[torch.nn.Parameter], sizes: List[int], delta_cat: torch.Tensor):
    with torch.no_grad():
        off = 0
        for p, sz in zip(params, sizes):
            d = delta_cat[off: off + sz].view_as(p.data)
            p.data.add_(-d.to(dtype=p.data.dtype))
            off += sz

def pecl_project_cat_orth(
    vec_cat: torch.Tensor,
    unit_dir_pool_cat: Dict[str, torch.Tensor],
    other_cities: List[str],
    lam: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    if (not other_cities) or float(lam) <= 0.0:
        return vec_cat
    q_list: List[torch.Tensor] = []
    for c in other_cities:
        v = unit_dir_pool_cat.get(c, None)
        if v is None or v.numel() != vec_cat.numel():
            continue
        u = v.detach().float()
        for q in q_list:
            u = u - torch.dot(u, q) * q
        n = u.norm()
        if float(n.item()) > float(eps):
            q_list.append(u / n)
    if len(q_list) == 0:
        return vec_cat
    vec_f = vec_cat.detach().float()
    proj = torch.zeros_like(vec_f)
    for q in q_list:
        proj.add_(q, alpha=torch.dot(vec_f, q).item())
    return vec_cat - float(lam) * proj.to(dtype=vec_cat.dtype)

def update_city_direction_pool_cat(
    city: str,
    raw_dir_pool_cat: Dict[str, torch.Tensor],
    unit_dir_pool_cat: Dict[str, torch.Tensor],
    delta_cat_mean: torch.Tensor,
    ema_beta: float,
    eps: float = 1e-12,
):
    with torch.no_grad():
        d = delta_cat_mean.detach().float()
        n0 = d.norm()
        if float(n0.item()) <= float(eps):
            return
        d = d / n0
        if city in raw_dir_pool_cat:
            if torch.dot(raw_dir_pool_cat[city].detach().float(), d).item() < 0.0:
                d = -d
        if city in raw_dir_pool_cat:
            raw_dir_pool_cat[city].mul_(float(ema_beta)).add_(d, alpha=(1.0 - float(ema_beta)))
        else:
            raw_dir_pool_cat[city] = d.clone()
        nrm = raw_dir_pool_cat[city].norm()
        if float(nrm.item()) > float(eps):
            unit_dir_pool_cat[city] = raw_dir_pool_cat[city] / nrm
        else:
            unit_dir_pool_cat.pop(city, None)

class ProjectedAdamCatState:
    def __init__(
        self,
        total_dim: int,
        device: torch.device,
        lr_cat_base: torch.Tensor,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.total_dim = int(total_dim)
        self.device = device
        assert lr_cat_base.ndim == 1 and lr_cat_base.numel() == self.total_dim
        self.lr_cat_base = lr_cat_base.detach().float()
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self._m: Dict[str, torch.Tensor] = {}
        self._v: Dict[str, torch.Tensor] = {}
        self._t: Dict[str, int] = {}

    def _get_state(self, city: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        m = self._m.get(city, None)
        v = self._v.get(city, None)
        t = int(self._t.get(city, 0))
        if m is None or m.numel() != self.total_dim or m.device != self.device:
            m = torch.zeros((self.total_dim,), device=self.device, dtype=torch.float32)
            self._m[city] = m
        if v is None or v.numel() != self.total_dim or v.device != self.device:
            v = torch.zeros((self.total_dim,), device=self.device, dtype=torch.float32)
            self._v[city] = v
        return m, v, t

    @torch.no_grad()
    def step(
        self,
        city: str,
        grad_cat: torch.Tensor,
        lr_scale: float,
        do_pecl: bool,
        unit_dir_pool_cat: Dict[str, torch.Tensor],
        other_cities: List[str],
        lam: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert grad_cat.ndim == 1 and grad_cat.numel() == self.total_dim
        m, v, t = self._get_state(city)
        t = t + 1
        self._t[city] = int(t)
        b1, b2 = self.beta1, self.beta2
        m.mul_(b1).add_(grad_cat, alpha=(1.0 - b1))
        v.mul_(b2).addcmul_(grad_cat, grad_cat, value=(1.0 - b2))
        bc1 = 1.0 - (b1 ** t)
        bc2 = 1.0 - (b2 ** t)
        m_hat = m / float(max(bc1, 1e-12))
        v_hat = v / float(max(bc2, 1e-12))
        denom = torch.sqrt(v_hat).add_(self.eps)
        u = m_hat / denom
        lr_cat_scaled = self.lr_cat_base * float(lr_scale)
        delta_raw = lr_cat_scaled * u
        if (not do_pecl) or (len(other_cities) == 0) or float(lam) <= 0.0:
            return delta_raw, delta_raw
        delta_apply = pecl_project_cat_orth(
            vec_cat=delta_raw,
            unit_dir_pool_cat=unit_dir_pool_cat,
            other_cities=other_cities,
            lam=float(lam),
        )
        sqrt_v_hat = torch.sqrt(v_hat)
        sqrt_v_hat_purged = pecl_project_cat_orth(
            vec_cat=sqrt_v_hat,
            unit_dir_pool_cat=unit_dir_pool_cat,
            other_cities=other_cities,
            lam=float(lam),
        )
        sqrt_v_hat_purged = sqrt_v_hat_purged.clamp(min=0.0)
        v_hat_new = sqrt_v_hat_purged * sqrt_v_hat_purged
        v_new = v_hat_new * float(max(bc2, 1e-12))
        v.copy_(v_new)
        denom_new = sqrt_v_hat_purged + self.eps
        safe_lr = lr_cat_scaled.clone()
        safe_lr.abs_().clamp_(min=1e-12)
        u_apply = delta_apply / safe_lr
        m_hat_new = u_apply * denom_new
        m_new = m_hat_new * float(max(bc1, 1e-12))
        m.copy_(m_new)
        return delta_apply, delta_raw

def sample_city_minibatch(
    graphs: List[PromptAlignedGraphSample],
    futures: np.ndarray,
    idx_pool: np.ndarray,
    batch_size: int,
    rng: np.random.RandomState,
):
    idx_pool = np.asarray(idx_pool, dtype=np.int64)
    n = int(idx_pool.size)
    if n <= 0:
        return [], np.zeros((0,) + futures.shape[1:], dtype=np.float32)
    B = int(min(int(batch_size), n))
    replace = (n < B)
    sel = rng.choice(n, size=B, replace=replace).astype(np.int64)
    batch_idx = idx_pool[sel]
    batch_graphs = [graphs[int(i)] for i in batch_idx.tolist()]
    batch_future = futures[batch_idx].astype(np.float32)
    return batch_graphs, batch_future

@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mr_threshold: float,
    lambda_pred_l2: float,
) -> Tuple[float, float, float, float]:
    was_training = model.training
    model.eval()
    amp_dtype = getattr(model, "llm_dtype", torch.float32)
    use_amp = (device.type == "cuda") and (amp_dtype in (torch.float16, torch.bfloat16))
    loss_sum = ade_sum = fde_sum = mr_sum = 0.0
    n_sum = 0
    for batch_graphs, batch_future_np in loader:
        B = len(batch_graphs)
        if B == 0:
            continue
        batch_future = torch.from_numpy(batch_future_np).to(device=device, dtype=torch.float32)
        beta = 0.5
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                preds = model(batch_graphs)
                preds_f = preds.float()
                loss_l1 = F.smooth_l1_loss(preds_f, batch_future, reduction="mean", beta=beta)
                loss_reg = float(lambda_pred_l2) * (preds_f.pow(2).mean())
                loss = loss_l1 + loss_reg
        else:
            preds = model(batch_graphs)
            preds_f = preds.float()
            loss_l1 = F.smooth_l1_loss(preds_f, batch_future, reduction="mean", beta=beta)
            loss_reg = float(lambda_pred_l2) * (preds_f.pow(2).mean())
            loss = loss_l1 + loss_reg
        preds_np = preds.detach().float().cpu().numpy()
        ade, fde, mr = compute_ade_fde_mr(preds_np, batch_future_np, mr_threshold)
        loss_sum += float(loss.item()) * B
        ade_sum += float(ade) * B
        fde_sum += float(fde) * B
        mr_sum += float(mr) * B
        n_sum += B
    if was_training:
        model.train()
    if n_sum == 0:
        return 0.0, 0.0, 0.0, 0.0
    return loss_sum / n_sum, ade_sum / n_sum, fde_sum / n_sum, mr_sum / n_sum

def build_city_test_loader(
    spec: CitySpec,
    cfg: MultiCityTrainingConfig,
    device: torch.device,
    rng: np.random.RandomState,
) -> Tuple[str, DataLoader, int]:
    city = spec.name
    futures_keep, hist_lens_all, keep_mask, stats = extract_true_future_and_histlen_from_npy(
        npy_path=spec.test_npy,
        future_len=int(cfg.future_len),
        pred_len=int(cfg.pred_len),
    )
    graphs_all = load_or_create_graphs(city=city, tag=cfg.test_tag, cfg=cfg)
    graphs_keep, futures_keep = align_graphs_and_targets(graphs_all, futures_keep, keep_mask, stats)
    hist_lens_keep = hist_lens_all[keep_mask]
    maybe_attach_traj_valid_len(graphs_keep, hist_lens_keep)
    dataset = CityTrajDataset(graphs_keep, futures_keep)
    n_all = len(dataset)
    n_use = _pick_count(n_all, cfg.test_ratio, cfg.test_cap) if n_all > 0 else 0
    if n_use <= 0:
        use_idx = np.array([], dtype=np.int64)
    elif n_use >= n_all:
        use_idx = np.arange(n_all, dtype=np.int64)
    else:
        use_idx = rng.choice(n_all, size=n_use, replace=False).astype(np.int64)
    subset = Subset(dataset, use_idx) if use_idx.size > 0 else Subset(dataset, [])
    test_loader = DataLoader(
        subset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return city, test_loader, int(len(subset))

def run_multi_city_training(cfg: MultiCityTrainingConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.model_dir, exist_ok=True)

    def _smooth_l1(pred: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
        try:
            return F.smooth_l1_loss(pred, target, reduction="mean", beta=float(beta))
        except TypeError:
            return F.smooth_l1_loss(pred, target, reduction="mean")

    if not cfg.pecl_constrain_lora:
        raise RuntimeError("This train file assumes LoRA is constrained. Set pecl_constrain_lora=True.")
    if float(cfg.wd_lora) != 0.0:
        raise RuntimeError(f"STRICT PECL requires wd_lora=0. Got wd_lora={cfg.wd_lora}.")
    if cfg.pecl_constrain_head and float(cfg.wd_head) != 0.0:
        raise RuntimeError(f"STRICT PECL on head requires wd_head=0. Got wd_head={cfg.wd_head}.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model = MultiCityKnowledgeIntegrationModel(
        llm_model_name=cfg.llm_model_name,
        d_traj=cfg.d_traj,
        hidden_proj=cfg.hidden_proj,
        pred_len=cfg.pred_len,
        out_dim=cfg.out_dim,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        graph_dropout=cfg.graph_dropout,
        semantic_tau=cfg.semantic_tau,
        head_conv_kernel=cfg.head_conv_kernel,
        head_conv_layers=cfg.head_conv_layers,
        head_dropout=cfg.head_dropout,
        tanh_scale=cfg.tanh_scale,
        base_device=device,
    ).to(device)

    graph_frozen = False
    if cfg.freeze_graph:
        for n, p in model.named_parameters():
            if n.startswith("graph_encoder."):
                p.requires_grad_(False)
        graph_frozen = True

    if cfg.force_fp32_trainables:
        _force_fp32_trainables(model)

    if cfg.warm_start_head_last_linear:
        _warm_start_head_last_linear_if_zero(model, std=float(cfg.head_last_init_std))

    counts = summarize_trainables(model)

    amp_dtype = getattr(model, "llm_dtype", torch.float32)
    use_amp = (device.type == "cuda") and (amp_dtype in (torch.float16, torch.bfloat16))
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    lora_params = collect_lora_params(model)
    head_params = collect_head_params(model)
    graph_params = collect_graph_params(model)

    if len(lora_params) == 0:
        raise RuntimeError("No trainable LoRA params detected.")
    if len(head_params) == 0:
        raise RuntimeError("No trainable reg_head params detected.")

    blocks = collect_constrained_blocks(model, cfg, device=device)
    params_c_all: List[torch.nn.Parameter] = []
    for bk in sorted(blocks.keys()):
        params_c_all.extend(blocks[bk].params)
    if len(params_c_all) == 0:
        raise RuntimeError("No constrained params found after blocking.")

    proj_adam_blocks: Dict[str, ProjectedAdamCatState] = {}
    for bk, blk in blocks.items():
        proj_adam_blocks[bk] = ProjectedAdamCatState(
            total_dim=int(blk.dim),
            device=device,
            lr_cat_base=blk.lr_cat_base,
            beta1=float(cfg.adam_beta1),
            beta2=float(cfg.adam_beta2),
            eps=float(cfg.adam_eps),
        )

    dummy_unscale_opt = torch.optim.SGD(params_c_all, lr=1.0)

    optimizer_head = None
    if not cfg.pecl_constrain_head:
        optimizer_head = torch.optim.AdamW(
            head_params,
            lr=float(cfg.lr_head),
            weight_decay=float(cfg.wd_head),
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    optimizer_graph = None
    if (not cfg.freeze_graph) and (not cfg.pecl_constrain_graph) and len(graph_params) > 0:
        optimizer_graph = torch.optim.AdamW(
            graph_params,
            lr=float(cfg.lr_graph),
            weight_decay=float(cfg.wd_graph),
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    train_graphs: Dict[str, List[PromptAlignedGraphSample]] = {}
    train_futures: Dict[str, np.ndarray] = {}
    city_train_indices_full: Dict[str, np.ndarray] = {}
    city_batches_est: Dict[str, int] = {}

    test_loaders: Dict[str, DataLoader] = {}
    test_sizes: Dict[str, int] = {}

    rng_split = np.random.RandomState(int(cfg.seed))
    rng_test = np.random.RandomState(int(cfg.seed) + 17)

    active_cities: List[str] = []

    for spec in cfg.cities:
        city = spec.name
        futures_keep, hist_lens_all, keep_mask, stats = extract_true_future_and_histlen_from_npy(
            npy_path=spec.train_npy,
            future_len=int(cfg.future_len),
            pred_len=int(cfg.pred_len),
        )
        graphs_all = load_or_create_graphs(city=city, tag=cfg.train_tag, cfg=cfg)
        graphs_keep, futures_keep = align_graphs_and_targets(graphs_all, futures_keep, keep_mask, stats)
        hist_lens_keep = hist_lens_all[keep_mask]
        maybe_attach_traj_valid_len(graphs_keep, hist_lens_keep)
        N = len(graphs_keep)
        train_graphs[city] = graphs_keep
        train_futures[city] = futures_keep
        if N < 2:
            tr_idx_full = np.arange(N, dtype=np.int64)
        else:
            perm = rng_split.permutation(N)
            n_val = max(1, int(N * float(cfg.val_ratio)))
            tr_idx_full = perm[n_val:]
        city_train_indices_full[city] = tr_idx_full
        tr_full = int(tr_idx_full.size)
        if tr_full <= 0:
            city_batches_est[city] = 0
        else:
            n_budget = _pick_count(tr_full, cfg.train_ratio, cfg.train_cap)
            bs = int(cfg.batch_size)
            nb = 1 if n_budget < bs else max(1, int(math.ceil(float(n_budget) / float(bs))))
            city_batches_est[city] = int(nb)
        _, tl, n_test_use = build_city_test_loader(spec, cfg, device, rng_test)
        test_loaders[city] = tl
        test_sizes[city] = int(n_test_use)
        if tr_full > 0:
            active_cities.append(city)

    if len(active_cities) == 0:
        raise RuntimeError("No active cities with training samples.")

    auto_rounds = max(int(city_batches_est[c]) for c in active_cities) if active_cities else 0
    rounds_per_epoch = int(cfg.rounds_per_epoch) if int(cfg.rounds_per_epoch) > 0 else int(auto_rounds)
    rounds_per_epoch = max(1, int(rounds_per_epoch))

    city_train_sizes = {c: int(city_train_indices_full[c].size) for c in active_cities}
    batches_per_round = compute_city_block_batches_per_round(
        active_cities=active_cities,
        city_train_sizes=city_train_sizes,
        min_batches=int(cfg.city_block_min_batches),
        max_batches=int(cfg.city_block_max_batches),
    )
    steps_per_round = int(sum(int(batches_per_round[c]) for c in active_cities))
    steps_per_epoch = int(rounds_per_epoch) * int(steps_per_round)
    total_steps = int(steps_per_epoch * int(cfg.epochs))
    warmup_steps = int(float(cfg.warmup_ratio) * float(total_steps)) if cfg.use_scheduler else 0
    warmup_steps = max(0, warmup_steps)
    min_lr_scale = float(max(0.0, min(float(cfg.min_lr_scale), 1.0)))

    raw_dir_pool_blocks: Dict[str, Dict[str, torch.Tensor]] = {bk: {} for bk in blocks.keys()}
    unit_dir_pool_blocks: Dict[str, Dict[str, torch.Tensor]] = {bk: {} for bk in blocks.keys()}

    dir_sum_blocks: Dict[str, Dict[str, torch.Tensor]] = {
        c: {bk: torch.zeros((blocks[bk].dim,), device=device, dtype=torch.float32) for bk in blocks.keys()}
        for c in active_cities
    }
    dir_cnt_blocks: Dict[str, Dict[str, int]] = {c: {bk: 0 for bk in blocks.keys()} for c in active_cities}

    global_step = 0

    for epoch in range(1, int(cfg.epochs) + 1):
        epoch_start = time.time()
        lam_epoch = float(cfg.lambda_orth)
        model.train()

        city_rng: Dict[str, np.random.RandomState] = {}
        for city in active_cities:
            seed_stream = int(cfg.seed) + 10007 * int(epoch) + (abs(hash(city)) % 100000)
            city_rng[city] = np.random.RandomState(int(seed_stream))

        base_city_order = [s.name for s in cfg.cities if s.name in active_cities]

        city_stats = {c: {"loss": 0.0, "ade": 0.0, "fde": 0.0, "mr": 0.0, "n": 0} for c in active_cities}
        g_loss = g_ade = g_fde = g_mr = 0.0
        g_n = 0

        for r in range(1, int(rounds_per_epoch) + 1):
            city_order = list(base_city_order)
            if cfg.city_round_shuffle:
                rng_order = np.random.RandomState(int(cfg.seed) + 100000 * int(epoch) + int(r))
                rng_order.shuffle(city_order)

            for city in city_order:
                n_steps_city = int(batches_per_round.get(city, int(cfg.city_block_min_batches)))
                if n_steps_city <= 0:
                    continue

                graphs = train_graphs[city]
                futures = train_futures[city]
                idx_pool = city_train_indices_full[city]
                rng = city_rng[city]

                for _k in range(n_steps_city):
                    if (not graph_frozen) and int(cfg.graph_step_freeze) > 0 and int(global_step) >= int(cfg.graph_step_freeze):
                        for n, p in model.named_parameters():
                            if n.startswith("graph_encoder."):
                                p.requires_grad_(False)
                        graph_frozen = True
                        if optimizer_graph is not None:
                            optimizer_graph = None
                        graph_params = []

                    batch_graphs, batch_future_np = sample_city_minibatch(
                        graphs=graphs,
                        futures=futures,
                        idx_pool=idx_pool,
                        batch_size=int(cfg.batch_size),
                        rng=rng,
                    )
                    B = len(batch_graphs)
                    if B == 0:
                        continue

                    batch_future = torch.from_numpy(batch_future_np).to(device=device, dtype=torch.float32)

                    model.zero_grad(set_to_none=True)
                    if optimizer_head is not None:
                        optimizer_head.zero_grad(set_to_none=True)
                    if optimizer_graph is not None:
                        optimizer_graph.zero_grad(set_to_none=True)

                    beta = 0.5
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            preds = model(batch_graphs)
                            preds_f = preds.float()
                            loss_l1 = _smooth_l1(preds_f, batch_future, beta=beta)
                            loss_reg = float(cfg.lambda_pred_l2) * (preds_f.pow(2).mean())
                            loss = loss_l1 + loss_reg
                    else:
                        preds = model(batch_graphs)
                        preds_f = preds.float()
                        loss_l1 = _smooth_l1(preds_f, batch_future, beta=beta)
                        loss_reg = float(cfg.lambda_pred_l2) * (preds_f.pow(2).mean())
                        loss = loss_l1 + loss_reg

                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.unscale_(dummy_unscale_opt)
                        if optimizer_head is not None:
                            scaler.unscale_(optimizer_head)
                        if optimizer_graph is not None:
                            scaler.unscale_(optimizer_graph)
                    else:
                        loss.backward()

                    if float(cfg.grad_clip_lora) > 0:
                        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=float(cfg.grad_clip_lora))
                    if float(cfg.grad_clip_head) > 0:
                        torch.nn.utils.clip_grad_norm_(head_params, max_norm=float(cfg.grad_clip_head))
                    if optimizer_graph is not None and len(graph_params) > 0 and float(cfg.grad_clip_graph) > 0:
                        torch.nn.utils.clip_grad_norm_(graph_params, max_norm=float(cfg.grad_clip_graph))

                    lr_scale = cosine_schedule_scale(
                        step=global_step,
                        total_steps=total_steps,
                        warmup_steps=warmup_steps,
                        min_scale=min_lr_scale,
                    ) if cfg.use_scheduler else 1.0

                    if optimizer_head is not None:
                        for pg in optimizer_head.param_groups:
                            pg["lr"] = float(cfg.lr_head) * float(lr_scale)
                    if optimizer_graph is not None:
                        for pg in optimizer_graph.param_groups:
                            pg["lr"] = float(cfg.lr_graph) * float(lr_scale)

                    finite_ok = True
                    for bk, blk in blocks.items():
                        gcat = build_grad_cat_from_params(blk.params, blk.sizes)
                        if not torch.isfinite(gcat).all():
                            finite_ok = False
                            break
                    if optimizer_head is not None:
                        finite_ok = finite_ok and grads_all_finite(head_params)
                    if optimizer_graph is not None:
                        finite_ok = finite_ok and grads_all_finite(graph_params)

                    grad_norm2 = 0.0
                    n_blocks_pecl = 0
                    n_blocks_total = len(blocks)

                    if finite_ok:
                        for bk, blk in blocks.items():
                            if graph_frozen and (blk.group == "graph"):
                                continue
                            grad_cat_b = build_grad_cat_from_params(blk.params, blk.sizes)
                            gnb = float(grad_cat_b.norm().item())
                            grad_norm2 += gnb * gnb
                            unit_pool_b = unit_dir_pool_blocks.get(bk, {})
                            other_cities_b = [c for c in active_cities if (c != city and c in unit_pool_b)]
                            do_pecl_b = (len(other_cities_b) > 0)
                            delta_apply_b, delta_raw_b = proj_adam_blocks[bk].step(
                                city=city,
                                grad_cat=grad_cat_b,
                                lr_scale=float(lr_scale),
                                do_pecl=do_pecl_b,
                                unit_dir_pool_cat=unit_pool_b,
                                other_cities=other_cities_b,
                                lam=float(lam_epoch),
                            )
                            apply_delta_cat_to_params(blk.params, blk.sizes, delta_apply_b)
                            if do_pecl_b:
                                n_blocks_pecl += 1
                            dir_sum_blocks[city][bk].add_(delta_apply_b.detach().float(), alpha=float(B))
                            dir_cnt_blocks[city][bk] += int(B)
                            if dir_cnt_blocks[city][bk] >= int(cfg.dir_update_every) * int(cfg.batch_size):
                                mean_delta = dir_sum_blocks[city][bk] / float(dir_cnt_blocks[city][bk])
                                update_city_direction_pool_cat(
                                    city=city,
                                    raw_dir_pool_cat=raw_dir_pool_blocks[bk],
                                    unit_dir_pool_cat=unit_dir_pool_blocks[bk],
                                    delta_cat_mean=mean_delta,
                                    ema_beta=float(cfg.pecl_dir_ema_beta),
                                )
                                dir_sum_blocks[city][bk].zero_()
                                dir_cnt_blocks[city][bk] = 0

                        if optimizer_head is not None:
                            if scaler.is_enabled():
                                scaler.step(optimizer_head)
                            else:
                                optimizer_head.step()
                        if optimizer_graph is not None:
                            if scaler.is_enabled():
                                scaler.step(optimizer_graph)
                            else:
                                optimizer_graph.step()

                    if scaler.is_enabled():
                        scaler.update()

                    with torch.no_grad():
                        preds_np = preds.detach().float().cpu().numpy()
                        ade, fde, mr = compute_ade_fde_mr(preds_np, batch_future_np, float(cfg.mr_threshold))
                        st = city_stats[city]
                        st["loss"] += float(loss.item()) * B
                        st["ade"] += float(ade) * B
                        st["fde"] += float(fde) * B
                        st["mr"] += float(mr) * B
                        st["n"] += B
                        g_loss += float(loss.item()) * B
                        g_ade += float(ade) * B
                        g_fde += float(fde) * B
                        g_mr += float(mr) * B
                        g_n += B

                    global_step += 1

                for bk in blocks.keys():
                    if graph_frozen and (blocks[bk].group == "graph"):
                        dir_sum_blocks[city][bk].zero_()
                        dir_cnt_blocks[city][bk] = 0
                        continue
                    if dir_cnt_blocks[city][bk] > 0:
                        mean_delta = dir_sum_blocks[city][bk] / float(dir_cnt_blocks[city][bk])
                        update_city_direction_pool_cat(
                            city=city,
                            raw_dir_pool_cat=raw_dir_pool_blocks[bk],
                            unit_dir_pool_cat=unit_dir_pool_blocks[bk],
                            delta_cat_mean=mean_delta,
                            ema_beta=float(cfg.pecl_dir_ema_beta),
                        )
                        dir_sum_blocks[city][bk].zero_()
                        dir_cnt_blocks[city][bk] = 0

        for c in active_cities:
            for bk in blocks.keys():
                if graph_frozen and (blocks[bk].group == "graph"):
                    dir_sum_blocks[c][bk].zero_()
                    dir_cnt_blocks[c][bk] = 0
                    continue
                if dir_cnt_blocks[c][bk] > 0:
                    mean_delta = dir_sum_blocks[c][bk] / float(dir_cnt_blocks[c][bk])
                    update_city_direction_pool_cat(
                        city=c,
                        raw_dir_pool_cat=raw_dir_pool_blocks[bk],
                        unit_dir_pool_cat=unit_dir_pool_blocks[bk],
                        delta_cat_mean=mean_delta,
                        ema_beta=float(cfg.pecl_dir_ema_beta),
                    )
                    dir_sum_blocks[c][bk].zero_()
                    dir_cnt_blocks[c][bk] = 0

        epoch_time = time.time() - epoch_start

        for city in active_cities:
            st = city_stats[city]
            if st["n"] <= 0:
                continue
            logger.info(
                f"[{city}] Train: loss={st['loss']/st['n']:.4f} | ADE={st['ade']/st['n']:.4f} | "
                f"FDE={st['fde']/st['n']:.4f} | MR={100.0*(st['mr']/st['n']):.1f}%"
            )

        if g_n > 0:
            logger.info(
                f"EPOCH {epoch} | Global: loss={g_loss / g_n:.4f} | ADE={g_ade / g_n:.4f} | "
                f"FDE={g_fde / g_n:.4f} | MR={100.0 * (g_mr / g_n):.1f}% | Time={epoch_time:.1f}s"
            )

        t_loss_sum = t_ade_sum = t_fde_sum = t_mr_sum = 0.0
        t_n_sum = 0

        for spec in cfg.cities:
            city = spec.name
            test_loader = test_loaders.get(city, None)
            n_test = int(test_sizes.get(city, 0))
            if test_loader is None or n_test <= 0:
                continue
            loss_t, ade_t, fde_t, mr_t = evaluate_loader(
                model, test_loader, device, float(cfg.mr_threshold), float(cfg.lambda_pred_l2)
            )
            t_loss_sum += float(loss_t) * n_test
            t_ade_sum += float(ade_t) * n_test
            t_fde_sum += float(fde_t) * n_test
            t_mr_sum += float(mr_t) * n_test
            t_n_sum += n_test

        do_save = (int(cfg.ckpt_every_epochs) > 0) and (int(epoch) % int(cfg.ckpt_every_epochs) == 0)
        if bool(cfg.ckpt_save_last) and int(epoch) == int(cfg.epochs):
            do_save = True

        if do_save:
            ckpt_path = os.path.join(cfg.model_dir, f"TrajMind_epoch{epoch:03d}.pt")
            cfg_dict = asdict(cfg)
            raw_cpu = {bk: {c: v.detach().float().cpu() for c, v in raw_dir_pool_blocks[bk].items()} for bk in raw_dir_pool_blocks.keys()}
            unit_cpu = {bk: {c: v.detach().float().cpu() for c, v in unit_dir_pool_blocks[bk].items()} for bk in unit_dir_pool_blocks.keys()}
            torch.save(
                {
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "model_state": model.state_dict(),
                    "config_dict": cfg_dict,
                    "pecl_raw_dir_pool_blocks": raw_cpu,
                    "pecl_unit_dir_pool_blocks": unit_cpu,
                },
                ckpt_path,
            )

def main():
    cfg = MultiCityTrainingConfig(
        # Please change the data path to run the code.
        cities=[
            CitySpec("porto", train_npy="./", test_npy="./"),
            CitySpec("BJ", train_npy="./", test_npy="./"),
            CitySpec("CD", train_npy="./", test_npy="./"),
            CitySpec("XA", train_npy="./", test_npy="./"),
        ],
        # Please change the store path to run the code.
        model_dir="/",
        graph_dir="/",
        graph_cache_dir="/",
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=128,
        ckpt_every_epochs=5,
        epochs=15,
        pecl_constrain_lora=True,
        pecl_constrain_head=False,
        pecl_constrain_graph=False,
        freeze_graph=False,
        graph_step_freeze=80000,
        city_round_shuffle=True,
        rounds_per_epoch=400,
        city_block_min_batches=4,
        city_block_max_batches=6,
        use_scheduler=True,
        warmup_ratio=0.0,
        min_lr_scale=0.08,
        train_ratio=1.0,
        train_cap=0,
        test_ratio=1.0,
        test_cap=0,
        tanh_scale=1.0,
        lambda_pred_l2=2e-5,
        lambda_orth=1.0,
        pecl_dir_ema_beta=0.9,
        dir_update_every=8,
        lr_lora=1e-4,
        lr_head=3.0e-5,
        lr_graph=1e-5,
        wd_lora=0.0,
        wd_head=1e-6,
        wd_graph=1e-6,
        force_fp32_trainables=True,
        warm_start_head_last_linear=True,
        head_last_init_std=1e-3,
        grad_clip_lora=1.0,
        grad_clip_head=0.5,
        grad_clip_graph=0.5,
        log_every=500,
    )
    run_multi_city_training(cfg)

if __name__ == "__main__":
    main()
