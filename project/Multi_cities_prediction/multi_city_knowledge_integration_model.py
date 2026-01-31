from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, List

import math
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model


DEFAULT_LLM_MODEL_NAME = "LLM_MODEL_NAME"


@dataclass
class PromptAlignedGraphSample:
    traj_tokens: torch.Tensor
    traj_valid_len: int
    word_tokens: torch.Tensor
    prompt_global: torch.Tensor
    edge_index_temporal: torch.Tensor
    edge_index_semantic: torch.Tensor
    edge_index_global: torch.Tensor
    prompt_text: str = ""


def _assert_finite(x: torch.Tensor, name: str):
    if not torch.isfinite(x).all():
        raise FloatingPointError(f"{name} has NaN/Inf. shape={tuple(x.shape)}")


def _clamp_len(v: int, T: int) -> int:
    return max(0, min(int(v), int(T)))


def _prefer_llm_dtype_bf16() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def _inv_sigmoid(x: float, eps: float = 1e-6) -> float:
    x = max(eps, min(1.0 - eps, float(x)))
    return math.log(x / (1.0 - x))


class DiWeightedGCNLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.ln = nn.LayerNorm(self.dim)
        self.msg = nn.Linear(self.dim, self.dim, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout))

        nn.init.xavier_uniform_(self.msg.weight)
        nn.init.zeros_(self.msg.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor] = None,
        dst_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N = int(num_nodes)
        if x.numel() == 0:
            return x

        h = self.ln(x)
        m = self.msg(h)

        if edge_index is None or edge_index.numel() == 0:
            return x + self.drop(self.act(m))

        src = edge_index[0].long()
        dst = edge_index[1].long()

        if edge_weight is None:
            w = torch.ones((src.numel(),), device=x.device, dtype=torch.float32)
        else:
            w = edge_weight.to(device=x.device, dtype=torch.float32)

        out = torch.zeros((N, self.dim), device=x.device, dtype=m.dtype)
        msg = m[src] * w.unsqueeze(-1).to(dtype=m.dtype)
        out.index_add_(0, dst, msg)

        deg = torch.zeros((N,), device=x.device, dtype=torch.float32)
        deg.index_add_(0, dst, w)
        out = out / deg.clamp(min=1.0).unsqueeze(-1)

        if dst_scale is not None:
            out = out * dst_scale.to(device=x.device, dtype=out.dtype).unsqueeze(-1)

        y = x + self.drop(self.act(out))
        return y


class DiWeightedGCN(nn.Module):
    def __init__(self, dim: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([DiWeightedGCNLayer(dim=dim, dropout=dropout) for _ in range(int(layers))])

    def forward(self, x, edge_index, num_nodes, edge_weight=None, dst_scale=None):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, num_nodes, edge_weight=edge_weight, dst_scale=dst_scale)
        return h


class LinearConvLinearHead(nn.Module):
    def __init__(
        self,
        d_text: int,
        pred_len: int,
        out_dim: int,
        d_mid: int = 256,
        kernel: int = 3,
        dropout: float = 0.1,
        tanh_temp: float = 2.0,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)
        self.d_mid = int(d_mid)
        self.tanh_temp = float(tanh_temp)

        k = int(kernel)
        p = k // 2

        self.ln_in = nn.LayerNorm(int(d_text))
        self.down = nn.Linear(int(d_text), self.d_mid, bias=True)

        self.conv = nn.Conv1d(self.d_mid, self.d_mid, kernel_size=k, padding=p, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout))

        self.ln_mid = nn.LayerNorm(self.d_mid)
        self.out = nn.Linear(self.d_mid, self.pred_len * self.out_dim, bias=True)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)

        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, hidden: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
        B, L, _ = hidden.shape

        x = self.down(self.ln_in(hidden))
        x = x.transpose(1, 2).contiguous()
        x = self.conv(x)
        x = self.act(x)
        x = self.drop(x)
        x = x.transpose(1, 2).contiguous()

        x = self.ln_mid(x)

        lengths_t = torch.tensor([int(l) for l in lengths], device=hidden.device, dtype=torch.long)
        lengths_t = lengths_t.clamp(min=1, max=L)
        last_idx = lengths_t - 1
        batch_idx = torch.arange(B, device=hidden.device)

        feat = x[batch_idx, last_idx]
        raw = self.out(feat).view(B, self.pred_len, self.out_dim)

        _assert_finite(raw, "head_output")
        return raw


class TrajPromptGraphEncoder(nn.Module):
    MAX_CACHE_SIZE = 64

    def __init__(
        self,
        d_traj: int,
        d_text: int,
        dropout: float = 0.1,
        semantic_tau: float = 1.0,
        device: Optional[torch.device] = None,
        local_layers: int = 2,
        global_layers: int = 1,
        init_prompt_inj: float = 0.05,
        init_traj_inj: float = 0.10,
        init_ctx_scale: float = 0.05,
        prompt_inj_max: float = 0.30,
        traj_inj_max: float = 0.50,
        ctx_scale_max: float = 0.30,
        traj_token_scale_init: float = 0.20,
        clamp_traj_scale: bool = True,
        dst_scale_max: float = 5.0,
        dst_scale_min: float = 0.2,
    ):
        super().__init__()
        self.base_device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.d_traj = int(d_traj)
        self.d_text = int(d_text)
        self.semantic_tau = float(semantic_tau)
        self.clamp_traj_scale = bool(clamp_traj_scale)

        self.dst_scale_max = float(dst_scale_max)
        self.dst_scale_min = float(dst_scale_min)

        self.traj_proj = nn.Sequential(
            nn.Linear(self.d_traj, self.d_text, bias=True),
            nn.GELU(),
            nn.Linear(self.d_text, self.d_text, bias=True),
        )
        nn.init.xavier_uniform_(self.traj_proj[0].weight)
        nn.init.zeros_(self.traj_proj[0].bias)
        nn.init.xavier_uniform_(self.traj_proj[2].weight)
        nn.init.zeros_(self.traj_proj[2].bias)

        self.traj_ln = nn.LayerNorm(self.d_text)
        self.traj_token_scale = nn.Parameter(torch.tensor(float(traj_token_scale_init)))

        self.local_gnn = DiWeightedGCN(dim=self.d_text, layers=int(local_layers), dropout=float(dropout))
        self.global_gnn = DiWeightedGCN(dim=self.d_text, layers=int(global_layers), dropout=float(dropout))

        self.ctx_proj = nn.Linear(self.d_text, self.d_text, bias=True)
        nn.init.normal_(self.ctx_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.ctx_proj.bias)

        self.delta_ln = nn.LayerNorm(self.d_text)
        self.delta_proj = nn.Linear(self.d_text, self.d_text, bias=True)
        nn.init.normal_(self.delta_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta_proj.bias)

        self.prompt_inj_max = float(prompt_inj_max)
        self.traj_inj_max = float(traj_inj_max)
        self.ctx_scale_max = float(ctx_scale_max)

        p0 = float(init_prompt_inj) / max(self.prompt_inj_max, 1e-6)
        t0 = float(init_traj_inj) / max(self.traj_inj_max, 1e-6)
        c0 = float(init_ctx_scale) / max(self.ctx_scale_max, 1e-6)

        self.prompt_inj_logit = nn.Parameter(torch.tensor(_inv_sigmoid(p0)))
        self.traj_inj_logit = nn.Parameter(torch.tensor(_inv_sigmoid(t0)))
        self.ctx_scale_logit = nn.Parameter(torch.tensor(_inv_sigmoid(c0)))

        self._prompt_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._edge_cache: Dict[Tuple[int, int, str], Tuple[torch.Tensor, int]] = {}

        self.to(self.base_device)
        for p in self.parameters():
            if p.is_floating_point():
                p.data = p.data.float()

    def _evict_if_needed(self, cache: dict):
        if len(cache) >= self.MAX_CACHE_SIZE:
            oldest = next(iter(cache))
            del cache[oldest]

    def _get_prompt_tensors(
        self, sample: PromptAlignedGraphSample, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_text = getattr(sample, "prompt_text", "")
        key = (prompt_text, str(device))
        if key in self._prompt_cache:
            return self._prompt_cache[key]
        self._evict_if_needed(self._prompt_cache)

        wt = sample.word_tokens.to(device=device, dtype=torch.float32).detach()
        pg = sample.prompt_global.to(device=device, dtype=torch.float32).detach()
        self._prompt_cache[key] = (wt, pg)
        return wt, pg

    def _get_local_edges(
        self,
        sample: PromptAlignedGraphSample,
        device: torch.device,
        T_full: int,
        M: int,
    ) -> Tuple[torch.Tensor, int]:
        key = (int(T_full), int(M), str(device))
        if key in self._edge_cache:
            return self._edge_cache[key]
        self._evict_if_needed(self._edge_cache)

        e_temp = sample.edge_index_temporal
        e_sem = sample.edge_index_semantic
        E_temp = int(e_temp.size(1))

        e_all = torch.cat(
            [e_temp.to(dtype=torch.long, device=device), e_sem.to(dtype=torch.long, device=device)],
            dim=1,
        )
        self._edge_cache[key] = (e_all, E_temp)
        return e_all, E_temp

    def forward(self, sample: PromptAlignedGraphSample) -> Tuple[torch.Tensor, int]:
        device = self.base_device

        traj_tokens = sample.traj_tokens.to(device=device, dtype=torch.float32)
        T_full = int(traj_tokens.size(0))

        vlen_raw = getattr(sample, "traj_valid_len", T_full)
        valid_len = max(1, _clamp_len(int(vlen_raw), T_full))

        word_tokens, _ = self._get_prompt_tensors(sample, device)
        M = int(word_tokens.size(0))

        prompt_inj = self.prompt_inj_max * torch.sigmoid(self.prompt_inj_logit)
        traj_inj = self.traj_inj_max * torch.sigmoid(self.traj_inj_logit)
        ctx_scale = self.ctx_scale_max * torch.sigmoid(self.ctx_scale_logit)

        w0 = word_tokens
        z0 = self.traj_ln(self.traj_proj(traj_tokens))
        if self.clamp_traj_scale:
            z0 = z0 * self.traj_token_scale.clamp(min=0.05, max=2.0)
        else:
            z0 = z0 * self.traj_token_scale

        _assert_finite(w0, "prompt_tokens_w0")
        _assert_finite(z0, "traj_tokens_z0")

        h_init = torch.cat([z0, w0], dim=0)

        e_all, E_temp = self._get_local_edges(sample, device, T_full, M)

        if E_temp > 0:
            idx = torch.arange(E_temp, device=device)
            w_temp = (idx < max(valid_len - 1, 0)).float()
        else:
            w_temp = torch.zeros((0,), device=device, dtype=torch.float32)

        if M > 0:
            z_norm = F.normalize(z0[:valid_len], dim=-1)
            w_norm = F.normalize(w0, dim=-1)
            sim = (z_norm @ w_norm.transpose(0, 1)) / max(self.semantic_tau, 1e-6)
            alpha_valid = torch.softmax(sim, dim=1)

            if valid_len < T_full:
                pad = torch.zeros((T_full - valid_len, M), device=device, dtype=alpha_valid.dtype)
                alpha_full = torch.cat([alpha_valid, pad], dim=0)
            else:
                alpha_full = alpha_valid

            w_sem = alpha_full.reshape(-1).to(dtype=torch.float32)

            beta = alpha_valid.sum(dim=0)
            inv_beta = 1.0 / (beta.detach() + 1e-6)
            inv_beta = inv_beta.clamp(min=self.dst_scale_min, max=self.dst_scale_max)

            dst_scale_vec = torch.ones((T_full + M,), device=device, dtype=torch.float32)
            dst_scale_vec[T_full : T_full + M] = inv_beta
        else:
            w_sem = torch.zeros((0,), device=device, dtype=torch.float32)
            dst_scale_vec = None

        w_all = torch.cat([w_temp, w_sem], dim=0)
        _assert_finite(w_all, "edge_weights_all")

        h_loc = self.local_gnn(
            h_init,
            e_all,
            T_full + M,
            edge_weight=w_all,
            dst_scale=dst_scale_vec,
        )

        h_traj_loc = h_loc[:T_full]
        h_text_loc = h_loc[T_full:]

        g_traj = h_traj_loc[:valid_len].mean(dim=0)
        g_text = h_text_loc.mean(dim=0) if M > 0 else torch.zeros((self.d_text,), device=device, dtype=torch.float32)

        g_nodes = torch.stack([g_traj, g_text], dim=0)
        e_global = sample.edge_index_global.to(device=device, dtype=torch.long)
        h_glob = self.global_gnn(g_nodes, e_global, 2, edge_weight=None, dst_scale=None)

        ctx = self.ctx_proj(h_glob[1]) * ctx_scale
        h_e = h_loc + ctx.unsqueeze(0)

        delta = self.delta_proj(self.delta_ln(h_e - h_init))
        delta_traj = delta[:T_full]
        delta_text = delta[T_full:]

        w_align = w0 + prompt_inj * delta_text if M > 0 else w0
        z_align = z0 + traj_inj * delta_traj
        z_align = z_align[:valid_len]

        seq = torch.cat([w_align, z_align], dim=0)
        total_len = int(M + valid_len)

        _assert_finite(seq, "graph_encoder_output")
        return seq, total_len


class MultiCityKnowledgeIntegrationModel(nn.Module):
    def __init__(
        self,
        llm_model_name: str = DEFAULT_LLM_MODEL_NAME,
        d_traj: int = 16,
        hidden_proj: int = 0,
        pred_len: int = 4,
        out_dim: int = 2,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules=None,
        graph_dropout: float = 0.1,
        semantic_tau: float = 1.0,
        local_layers: int = 2,
        global_layers: int = 1,
        head_conv_kernel: int = 3,
        head_conv_layers: int = 2,
        head_dropout: float = 0.1,
        tanh_scale: float = 2.0,
        use_gradient_checkpointing: bool = False,
        attn_implementation: str = "sdpa",
        lora_weights_fp32: bool = False,
        force_fp32: bool = False,
        base_device=None,
    ):
        super().__init__()
        self.base_device = base_device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)

        llm_dtype = _prefer_llm_dtype_bf16()
        self.llm_dtype = llm_dtype

        fp_kwargs = dict(low_cpu_mem_usage=True)
        sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
        if "dtype" in sig.parameters:
            fp_kwargs["dtype"] = llm_dtype
        else:
            fp_kwargs["torch_dtype"] = llm_dtype
        if "attn_implementation" in sig.parameters:
            fp_kwargs["attn_implementation"] = attn_implementation

        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name, **fp_kwargs).to(self.base_device)
        self.llm.config.use_cache = False

        if use_gradient_checkpointing:
            if hasattr(self.llm, "gradient_checkpointing_enable"):
                self.llm.gradient_checkpointing_enable()
            if hasattr(self.llm, "enable_input_require_grads"):
                self.llm.enable_input_require_grads()
        else:
            if hasattr(self.llm, "gradient_checkpointing_disable"):
                self.llm.gradient_checkpointing_disable()

        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=target_modules,
        )
        self.llm = get_peft_model(self.llm, lora_config)

        for n, p in self.llm.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        for n, p in self.llm.named_parameters():
            if "lora_" in n and p.is_floating_point():
                if lora_weights_fp32:
                    p.data = p.data.float()
                else:
                    p.data = p.data.to(dtype=self.llm_dtype)

        d_text = int(self.llm.config.hidden_size)

        self.graph_encoder = TrajPromptGraphEncoder(
            d_traj=int(d_traj),
            d_text=d_text,
            dropout=float(graph_dropout),
            semantic_tau=float(semantic_tau),
            device=self.base_device,
            local_layers=int(local_layers),
            global_layers=int(global_layers),
        )

        self.reg_head = LinearConvLinearHead(
            d_text=d_text,
            pred_len=self.pred_len,
            out_dim=self.out_dim,
            kernel=int(head_conv_kernel),
        ).to(self.base_device)

        for p in self.graph_encoder.parameters():
            if p.is_floating_point():
                p.data = p.data.float()
        for p in self.reg_head.parameters():
            if p.is_floating_point():
                p.data = p.data.float()

    def _get_transformer_body(self):
        base = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
        if hasattr(base, "model"):
            return base.model
        if hasattr(base, "transformer"):
            return base.transformer
        return base

    def forward(self, samples):
        if isinstance(samples, PromptAlignedGraphSample):
            samples = [samples]
        else:
            samples = list(samples)

        B = len(samples)
        device = self.base_device
        if B == 0:
            return torch.zeros((0, self.pred_len, self.out_dim), device=device, dtype=torch.float32)

        seq_list: List[torch.Tensor] = []
        lengths: List[int] = []
        for s in samples:
            seq, total_len = self.graph_encoder(s)
            seq_list.append(seq)
            lengths.append(int(total_len))

        padded = pad_sequence(seq_list, batch_first=True, padding_value=0.0)
        max_len = int(padded.size(1))

        lengths_t = torch.tensor(lengths, device=device, dtype=torch.long)
        attn_mask = (torch.arange(max_len, device=device).unsqueeze(0) < lengths_t.unsqueeze(1)).long()

        inputs_embeds = padded.to(device=device, dtype=self.llm_dtype)

        transformer = self._get_transformer_body()
        out = transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state

        pred = self.reg_head(hidden.float(), lengths)
        return pred

    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        return {n: p for n, p in self.named_parameters() if p.requires_grad}
