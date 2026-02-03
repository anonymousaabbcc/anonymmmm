from __future__ import annotations

from typing import Tuple, Optional, Dict, Any, List, Sequence, Union

import numpy as np
import random
import torch
from torch.utils.data import Dataset, get_worker_info


def set_seed(seed: int = 42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def to_torch(x: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    t = torch.from_numpy(x)
    if device is not None:
        t = t.to(device)
    return t


def _is_tail_contiguous_mask(mask_bool: np.ndarray) -> bool:
    if mask_bool.ndim != 1:
        return False
    if mask_bool.size == 0:
        return True
    zero_pos = np.where(~mask_bool)[0]
    if zero_pos.size == 0:
        return True
    first_zero = int(zero_pos[0])
    return bool(mask_bool[first_zero:].sum() == 0)


def extract_real_steps(traj: np.ndarray) -> np.ndarray:
    assert traj.ndim == 2 and traj.shape[1] == 4
    mask_bool = traj[:, 3] > 0.5
    if not mask_bool.any():
        return traj[0:0].astype(np.float32, copy=False)

    if _is_tail_contiguous_mask(mask_bool):
        T_real = int(mask_bool.sum())
        real = traj[:T_real].astype(np.float32, copy=False).copy()
    else:
        real = traj[mask_bool].astype(np.float32, copy=False).copy()

    real[:, 3] = 1.0
    return real


def split_hist_future_by_mask(traj: np.ndarray, future_len: int) -> Tuple[np.ndarray, np.ndarray]:
    assert traj.ndim == 2 and traj.shape[1] == 4
    real = extract_real_steps(traj)
    T_real = int(real.shape[0])

    if T_real <= 0:
        return traj[0:1].astype(np.float32, copy=True), traj[0:0].astype(np.float32, copy=True)

    future_len = int(future_len)
    if future_len <= 0:
        return real.copy(), real[0:0].copy()

    T_hist = max(1, T_real - future_len)
    hist = real[:T_hist].copy()
    future = real[T_hist:T_real].copy()
    return hist, future


def segment_pool_to_fixed_len_np(arr: np.ndarray, out_len: int) -> np.ndarray:
    assert arr.ndim == 2 and arr.shape[1] == 4
    out_len = int(out_len)
    if out_len <= 0:
        raise ValueError("out_len must be > 0")

    T = int(arr.shape[0])
    pooled = np.zeros((out_len, 4), dtype=np.float32)
    if T <= 0:
        return pooled

    xyt = arr[:, :3].astype(np.float32, copy=False)
    idx = (np.arange(T, dtype=np.int64) * out_len) // max(1, T)
    idx = np.clip(idx, 0, out_len - 1)

    sums = np.zeros((out_len, 3), dtype=np.float32)
    cnts = np.zeros((out_len,), dtype=np.float32)
    np.add.at(sums, idx, xyt)
    np.add.at(cnts, idx, 1.0)

    nonempty = cnts > 0
    means = sums / np.maximum(cnts[:, None], 1.0)

    if not bool(nonempty.all()):
        if not bool(nonempty.any()):
            means[:] = xyt[0]
        else:
            last = None
            for j in range(out_len):
                if nonempty[j]:
                    last = means[j].copy()
                elif last is not None:
                    means[j] = last
            if not nonempty[0]:
                j0 = int(np.argmax(nonempty))
                means[:j0] = means[j0]

    pooled[:, :3] = means
    pooled[:, 3] = 1.0
    return pooled


def uniform_strided_zero_view(arr: np.ndarray, delta: int) -> np.ndarray:
    assert arr.ndim == 2, f"expected 2D, got {arr.shape}"
    delta = int(delta)
    if delta <= 0:
        raise ValueError("delta must be a positive integer (Z+)")

    T = int(arr.shape[0])
    if T <= 0:
        return arr.astype(np.float32, copy=True)

    period = delta + 1
    keep = (np.arange(T, dtype=np.int64) % period) == 0
    out = arr.astype(np.float32, copy=True)
    out[~keep, :] = 0.0
    return out


def sample_neighbor_subtraj_pair(
    traj_hist: np.ndarray,
    sub_len: int,
    shift: int,
    rng: np.random.Generator,
    avoid_zero_shift: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    T = int(traj_hist.shape[0])
    if T <= 0:
        z = traj_hist.astype(np.float32, copy=True)
        return z, z

    sub_len = int(max(1, min(int(sub_len), T)))

    if avoid_zero_shift and shift == 0 and T > 1:
        shift = 1 if bool(rng.integers(0, 2)) else -1

    max_start = T - sub_len
    a_min = max(0, -shift)
    a_max = min(max_start, max_start - shift)

    if a_max < a_min:
        shift = int(np.clip(shift, -max_start, max_start))
        a_min = max(0, -shift)
        a_max = min(max_start, max_start - shift)
        if a_max < a_min:
            shift = 0
            a_min, a_max = 0, max_start

    anchor_start = int(rng.integers(a_min, a_max + 1)) if a_max > a_min else int(a_min)
    pos_start = anchor_start + shift

    st_anchor = traj_hist[anchor_start:anchor_start + sub_len].astype(np.float32, copy=True)
    st_pos = traj_hist[pos_start:pos_start + sub_len].astype(np.float32, copy=True)
    st_anchor[:, 3] = 1.0
    st_pos[:, 3] = 1.0
    return st_anchor, st_pos


def pad_to_len(arr: np.ndarray, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    T = int(arr.shape[0])
    out = np.zeros((int(max_len), 4), dtype=np.float32)
    mask = np.zeros((int(max_len),), dtype=np.float32)

    if T <= 0 or int(max_len) <= 0:
        return out, mask

    Lc = min(T, int(max_len))
    out[:Lc, :3] = arr[:Lc, :3].astype(np.float32, copy=False)
    out[:Lc, 3] = 1.0
    mask[:Lc] = 1.0
    return out, mask


_NpyPathType = Union[str, Sequence[Union[str, Tuple[str, str]]]]


def _parse_sources(npy_path: _NpyPathType) -> List[Tuple[str, str]]:
    if isinstance(npy_path, str):
        return [("city0", npy_path)]

    if not isinstance(npy_path, (list, tuple)):
        raise TypeError(f"npy_path must be str or list/tuple, got {type(npy_path)}")

    out: List[Tuple[str, str]] = []
    for i, it in enumerate(npy_path):
        if isinstance(it, str):
            out.append((f"city{i}", it))
        elif isinstance(it, (list, tuple)) and len(it) == 2 and isinstance(it[0], str) and isinstance(it[1], str):
            out.append((it[0], it[1]))
        else:
            raise TypeError(f"Unsupported npy source entry: {it}")
    if len(out) == 0:
        raise ValueError("Empty npy_path list.")
    return out


class MultiCityCLDataset(Dataset):
    def __init__(
        self,
        npy_path: _NpyPathType,
        target_len: int = 0,
        prepool_len: int = 0,
        delta_range: Tuple[int, int] = (2, 6),
        sub_len_range: Tuple[int, int] = (16, 64),
        shift_range: Tuple[int, int] = (-2, 2),
        seed: int = 42,
        future_len: int = 4,
        avoid_zero_shift: bool = True,
        mmap: bool = True,
        keep_recent: bool = True,
    ):
        super().__init__()

        self.sources = _parse_sources(npy_path)
        self.city_names = [c for c, _ in self.sources]
        self.paths = [p for _, p in self.sources]
        self.n_cities = len(self.sources)

        self.data_list: List[np.ndarray] = []
        self.N_list: List[int] = []
        self.L_list: List[int] = []

        print(f"[MultiCityCLDataset] Loading {self.n_cities} NPY source(s)...")
        for city, path in self.sources:
            print(f"  - {city}: {path}")
            arr = np.load(path, mmap_mode="r" if mmap else None)
            assert arr.ndim == 3 and arr.shape[2] == 4, f"{city} expected (N,L,4), got {arr.shape}"
            self.data_list.append(arr)
            self.N_list.append(int(arr.shape[0]))
            self.L_list.append(int(arr.shape[1]))

        self.N_total = int(sum(self.N_list))

        self.base_seed = int(seed)
        self.epoch = 0
        self.future_len = int(future_len)

        self.delta_range = (int(delta_range[0]), int(delta_range[1]))
        self.sub_len_range = (int(sub_len_range[0]), int(sub_len_range[1]))
        self.shift_range = (int(shift_range[0]), int(shift_range[1]))
        self.avoid_zero_shift = bool(avoid_zero_shift)

        self.cap_real_len_list: List[int] = []
        for L in self.L_list:
            cap = int(min(L, int(target_len))) if (target_len is not None and int(target_len) > 0) else int(L)
            self.cap_real_len_list.append(cap)

        self.keep_recent = bool(keep_recent)

        self.prepool_len = int(prepool_len) if (prepool_len is not None and int(prepool_len) > 0) else 0
        if self.prepool_len > 0:
            self.max_len = int(self.prepool_len)
        else:
            self.max_len = int(max(self.cap_real_len_list))

        self.cumN = np.cumsum(np.asarray(self.N_list, dtype=np.int64))
        self.base_uid = np.concatenate(([0], self.cumN[:-1])).astype(np.int64)

        mode = f"CPU-prepool({self.prepool_len})" if self.prepool_len > 0 else f"pad({self.max_len})"
        print(f"[MultiCityCLDataset] cities={self.n_cities} | N_total={self.N_total} | mode={mode} | future_len={self.future_len}")
        for i, city in enumerate(self.city_names):
            print(f"    {city:>6s}: N={self.N_list[i]}  L_city={self.L_list[i]}  cap_real_len={self.cap_real_len_list[i]}")
        print(f"[MultiCityCLDataset] density operator: uniform strided zeroing with delta in [{self.delta_range[0]},{self.delta_range[1]}]")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.N_total

    def _map_index(self, idx: int) -> Tuple[int, int]:
        city_id = int(np.searchsorted(self.cumN, idx, side="right"))
        prev = int(self.cumN[city_id - 1]) if city_id > 0 else 0
        local_idx = int(idx - prev)
        return city_id, local_idx

    def _rng_for_index(self, global_idx: int, city_id: int, local_idx: int) -> np.random.Generator:
        wi = get_worker_info()
        base = int(wi.seed) if wi is not None else self.base_seed
        seed = (base + self.epoch * 2000003 + global_idx * 1000003 + city_id * 99991 + local_idx * 37) % (2**32)
        return np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_idx = int(idx)
        city_id, local_idx = self._map_index(global_idx)
        rng = self._rng_for_index(global_idx, city_id, local_idx)

        data = self.data_list[city_id]
        traj = data[local_idx]

        hist, _ = split_hist_future_by_mask(traj, self.future_len)
        Th = int(hist.shape[0])

        cap_real_len = int(self.cap_real_len_list[city_id])
        if cap_real_len > 0 and Th > cap_real_len:
            if self.keep_recent:
                hist = hist[-cap_real_len:].copy()
            else:
                hist = hist[:cap_real_len].copy()
            Th = int(hist.shape[0])

        d0, d1 = self.delta_range
        if d1 < d0:
            d0, d1 = d1, d0
        delta = int(rng.integers(d0, d1 + 1))
        delta = max(1, delta)

        desired_min, desired_max = self.sub_len_range
        desired_min = int(max(1, desired_min))
        desired_max = int(max(desired_min, desired_max))

        if Th <= desired_min:
            sub_len = max(1, Th)
        else:
            sub_len = int(rng.integers(desired_min, min(desired_max, Th) + 1))

        smin, smax = self.shift_range
        smin, smax = int(smin), int(smax)
        if smax < smin:
            smin, smax = smax, smin

        if self.avoid_zero_shift and (smin <= 0 <= smax):
            candidates = list(range(smin, 0)) + list(range(1, smax + 1))
            shift = int(rng.choice(candidates)) if len(candidates) > 0 else 0
        else:
            shift = int(rng.integers(smin, smax + 1))

        st_anchor, st_pos = sample_neighbor_subtraj_pair(
            hist,
            sub_len=sub_len,
            shift=shift,
            rng=rng,
            avoid_zero_shift=self.avoid_zero_shift,
        )

        if self.prepool_len > 0:
            dA_4 = segment_pool_to_fixed_len_np(hist, self.prepool_len)
            dP_4 = uniform_strided_zero_view(dA_4, delta=delta)
            sA_4 = segment_pool_to_fixed_len_np(st_anchor, self.prepool_len)
            sP_4 = segment_pool_to_fixed_len_np(st_pos, self.prepool_len)

            m_dA = np.ones((self.prepool_len,), dtype=np.float32)
            m_dP = np.ones((self.prepool_len,), dtype=np.float32)
            m_sA = np.ones((self.prepool_len,), dtype=np.float32)
            m_sP = np.ones((self.prepool_len,), dtype=np.float32)
        else:
            dA_4, m_dA = pad_to_len(hist, self.max_len)
            dP_4 = uniform_strided_zero_view(dA_4, delta=delta)
            m_dP = m_dA.copy()

            sA_4, m_sA = pad_to_len(st_anchor, self.max_len)
            sP_4, m_sP = pad_to_len(st_pos, self.max_len)

        uid = int(self.base_uid[city_id] + local_idx)

        return {
            "density_anchor": to_torch(dA_4),
            "density_pos": to_torch(dP_4),
            "st_anchor": to_torch(sA_4),
            "st_pos": to_torch(sP_4),
            "mask_density_anchor": torch.from_numpy(m_dA),
            "mask_density_pos": torch.from_numpy(m_dP),
            "mask_st_anchor": torch.from_numpy(m_sA),
            "mask_st_pos": torch.from_numpy(m_sP),
            "traj_index": torch.tensor(uid, dtype=torch.long),
            "city_id": torch.tensor(city_id, dtype=torch.long),
            "delta": torch.tensor(int(delta), dtype=torch.long),
        }


class MultiCityFullViewDataset(Dataset):
    def __init__(
        self,
        npy_path: str,
        target_len: int = 0,
        prepool_len: int = 0,
        future_len: int = 4,
        mmap: bool = True,
        keep_recent: bool = True,
    ):
        super().__init__()
        print(f"[MultiCityFullViewDataset] Loading NPY from {npy_path} ...")
        data = np.load(npy_path, mmap_mode="r" if mmap else None)
        assert data.ndim == 3 and data.shape[2] == 4
        self.data = data
        self.N, self.L, self.D = data.shape
        assert self.D == 4

        self.future_len = int(future_len)

        self.cap_real_len = int(min(self.L, int(target_len))) if (target_len is not None and int(target_len) > 0) else int(self.L)
        self.keep_recent = bool(keep_recent)

        self.prepool_len = int(prepool_len) if (prepool_len is not None and int(prepool_len) > 0) else 0
        self.max_len = int(self.prepool_len) if self.prepool_len > 0 else int(self.cap_real_len)

        mode = f"CPU-prepool({self.prepool_len})" if self.prepool_len > 0 else f"pad({self.max_len})"
        print(f"[MultiCityFullViewDataset] N={self.N}, L_city={self.L}, cap_real_len={self.cap_real_len}, mode={mode}, future_len={self.future_len}")

    def __len__(self):
        return int(self.N)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj = self.data[int(idx)]
        hist, _ = split_hist_future_by_mask(traj, self.future_len)

        Th = int(hist.shape[0])
        if self.cap_real_len > 0 and Th > self.cap_real_len:
            if self.keep_recent:
                hist = hist[-self.cap_real_len:].copy()
            else:
                hist = hist[:self.cap_real_len].copy()

        if self.prepool_len > 0:
            full_4 = segment_pool_to_fixed_len_np(hist, self.prepool_len)
            mask_np = np.ones((self.prepool_len,), dtype=np.float32)
        else:
            full_4, mask_np = pad_to_len(hist, self.max_len)

        return {
            "full": to_torch(full_4),
            "mask": torch.from_numpy(mask_np),
            "traj_index": torch.tensor(int(idx), dtype=torch.long),
        }
