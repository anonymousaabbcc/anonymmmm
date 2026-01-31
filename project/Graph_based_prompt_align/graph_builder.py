# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "meta-llama/Llama-3.2-1b"

T_TOKENS = 64
D_TRAJ = 16

EMB_SUFFIX = "_embed_normalize.npy"


def get_project_root() -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "../.."))


def get_device(prefer_cpu: bool = True) -> torch.device:
    if prefer_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_llm_backbone(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def encode_prompt_tokens_with_llm(
    prompt_text: str,
    tokenizer,
    model,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)

    embed_layer = model.get_input_embeddings()
    token_embs = embed_layer(input_ids).squeeze(0)
    token_embs = token_embs.to(dtype=torch.float32).cpu()

    prompt_global = token_embs.mean(dim=0)
    return token_embs, prompt_global


def build_prediction_prompt(city: str, pred_len: int) -> str:
    return (
        f"Given the historical trajectory embedding of a taxi trip in {city}, "
        f"predict the next {int(pred_len)} steps as 2D locations in the same coordinate system as the input."
    )


def build_temporal_edges_traj_only(T: int) -> torch.Tensor:
    if T <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    src = torch.arange(0, T - 1, dtype=torch.long)
    dst = src + 1
    return torch.stack([src, dst], dim=0).cpu()


def build_semantic_edges_traj_to_prompt(
    T: int,
    L: int,
    traj_offset: int = 0,
    prompt_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if T == 0 or L == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32),
        )

    traj_idx = torch.arange(T, dtype=torch.long).unsqueeze(1).expand(T, L)
    tok_idx = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(T, L)

    src = traj_idx.reshape(-1) + traj_offset
    dst = tok_idx.reshape(-1) + prompt_offset

    edge_index = torch.stack([src, dst], dim=0).cpu()

    per_traj = torch.full((L,), 1.0 / float(L), dtype=torch.float32)
    edge_weight = per_traj.repeat(T).cpu()
    return edge_index, edge_weight


def build_global_edge_oneway() -> torch.Tensor:
    return torch.tensor([[0], [1]], dtype=torch.long)


def load_valid_lens_optional(
    emb_dir: str,
    city: str,
    split: str,
    N: int,
    T: int,
    *,
    strict: bool = False,
) -> np.ndarray:
    cand = os.path.join(emb_dir, f"{city}_{split}_valid_len.npy")
    if os.path.isfile(cand):
        v = np.load(cand)
        if v.shape[0] != N:
            raise ValueError(f"valid_len mismatch: {cand}, got {v.shape}, expect ({N},)")
        v = v.astype(np.int64)
        v = np.clip(v, 0, T)
        return v

    if strict:
        raise FileNotFoundError(f"Missing valid_len file: {cand}")
    return (np.ones((N,), dtype=np.int64) * int(T))


@dataclass
class PromptAlignedGraphSample:
    traj_tokens: torch.Tensor
    traj_valid_len: int
    word_tokens: torch.Tensor

    traj_global: torch.Tensor
    prompt_global: torch.Tensor

    edge_index_temporal: torch.Tensor
    edge_index_semantic: torch.Tensor
    edge_weight_semantic: torch.Tensor

    edge_index_global: torch.Tensor

    prompt_text: str
    city: str
    split: str
    index: int


def build_samples_from_embeddings(
    emb_arr: np.ndarray,
    city: str,
    split: str,
    *,
    pred_len: int,
    word_tokens_llm: torch.Tensor,
    prompt_global: torch.Tensor,
    edge_index_temporal: torch.Tensor,
    edge_index_semantic: torch.Tensor,
    edge_weight_semantic: torch.Tensor,
    edge_index_global: torch.Tensor,
    valid_lens: Optional[np.ndarray] = None,
) -> List[PromptAlignedGraphSample]:
    if emb_arr.ndim != 3:
        raise ValueError(f"emb_arr must be (N,T,d), got {emb_arr.shape}")
    N, T, d_traj = emb_arr.shape
    if T != T_TOKENS or d_traj != D_TRAJ:
        raise ValueError(f"Expect (N,{T_TOKENS},{D_TRAJ}), got {emb_arr.shape}")

    if valid_lens is None:
        valid_lens = np.ones((N,), dtype=np.int64) * int(T)

    prompt_text = build_prediction_prompt(city=city, pred_len=pred_len)

    samples: List[PromptAlignedGraphSample] = []
    for idx in range(N):
        traj_tokens = torch.from_numpy(emb_arr[idx].astype(np.float32)).cpu()
        vlen = int(valid_lens[idx])
        vlen = max(0, min(vlen, T))

        if vlen > 0:
            traj_global = traj_tokens[:vlen].mean(dim=0)
        else:
            traj_global = traj_tokens.mean(dim=0)

        samples.append(
            PromptAlignedGraphSample(
                traj_tokens=traj_tokens,
                traj_valid_len=vlen,
                word_tokens=word_tokens_llm,
                traj_global=traj_global,
                prompt_global=prompt_global,
                edge_index_temporal=edge_index_temporal,
                edge_index_semantic=edge_index_semantic,
                edge_weight_semantic=edge_weight_semantic,
                edge_index_global=edge_index_global,
                prompt_text=prompt_text,
                city=city,
                split=split,
                index=idx,
            )
        )
    return samples


def parse_city_split_from_embed_name(fname: str) -> Optional[Tuple[str, str]]:
    if not fname.endswith(EMB_SUFFIX):
        return None

    base = fname[:-len(EMB_SUFFIX)]
    try:
        city, split = base.rsplit("_", 1)
        return city, split
    except ValueError:
        return None


def construct_unitraj_prompt_graphs(
    pred_len: int = 4,
    model_name: str = MODEL_NAME,
    *,
    prefer_cpu: bool = True,
    strict_valid_len: bool = False,
    emb_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
):
    project_root = get_project_root()
    device = get_device(prefer_cpu=prefer_cpu)

    if emb_dir is None:
        emb_dir = os.path.join(project_root, "CL_embed_combined", "embedding_normalize")  # please change the data path to run
    if out_dir is None:
        out_dir = os.path.join(project_root, "Graph_based_prompt_align", "CL_embed_combined_normalize")  # please change the output path to run

    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(f"embedding dir not found: {emb_dir}")

    files = sorted([f for f in os.listdir(emb_dir) if f.endswith(EMB_SUFFIX)])
    if not files:
        raise FileNotFoundError(f"no *{EMB_SUFFIX} found in: {emb_dir}")

    tokenizer, llm = load_llm_backbone(model_name, device=device)
    d_text = int(llm.config.hidden_size)

    for fname in files:
        parsed = parse_city_split_from_embed_name(fname)
        if parsed is None:
            continue
        city, split = parsed

        emb_path = os.path.join(emb_dir, fname)
        emb_arr = np.load(emb_path)
        if emb_arr.ndim != 3:
            continue

        N, T, d_traj = emb_arr.shape
        if T != T_TOKENS or d_traj != D_TRAJ:
            raise ValueError(f"Expect (N,{T_TOKENS},{D_TRAJ}) but got {emb_arr.shape} from {emb_path}")

        valid_lens = load_valid_lens_optional(
            emb_dir=emb_dir, city=city, split=split, N=N, T=T, strict=strict_valid_len
        )

        prompt_text = build_prediction_prompt(city=city, pred_len=int(pred_len))
        word_tokens_llm, prompt_global = encode_prompt_tokens_with_llm(
            prompt_text=prompt_text, tokenizer=tokenizer, model=llm, device=device
        )

        if int(word_tokens_llm.size(1)) != int(d_text):
            raise RuntimeError(
                f"prompt embedding dim mismatch: got {word_tokens_llm.size(1)} but llm hidden is {d_text}."
            )

        L = int(word_tokens_llm.size(0))

        edge_index_temporal = build_temporal_edges_traj_only(T)
        edge_index_semantic, edge_weight_semantic = build_semantic_edges_traj_to_prompt(
            T=T, L=L, traj_offset=0, prompt_offset=T
        )
        edge_index_global = build_global_edge_oneway()

        samples = build_samples_from_embeddings(
            emb_arr=emb_arr,
            city=city,
            split=split,
            pred_len=int(pred_len),
            word_tokens_llm=word_tokens_llm,
            prompt_global=prompt_global,
            edge_index_temporal=edge_index_temporal,
            edge_index_semantic=edge_index_semantic,
            edge_weight_semantic=edge_weight_semantic,
            edge_index_global=edge_index_global,
            valid_lens=valid_lens,
        )

        out_path = os.path.join(out_dir, f"{city}_{split}_graphs.pt")
        torch.save(samples, out_path)

        del emb_arr


def build_prompt_alignment_graphs(pred_len: int = 4, model_name: str = MODEL_NAME):
    return construct_unitraj_prompt_graphs(pred_len=pred_len, model_name=model_name)


if __name__ == "__main__":
    construct_unitraj_prompt_graphs(pred_len=4, model_name=MODEL_NAME)
