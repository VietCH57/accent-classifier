"""Datasets, samplers, collate and on-GPU augmentation."""

from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from . import audio
from .utils import group_weights

SR = audio.SR


# --------------------------------------------------------------------------- evaluation / extraction
def make_segments(df: pd.DataFrame, seg: float, min_len: float) -> pd.DataFrame:
    """One row per fixed segment of every utterance (deterministic). `utt` = row index in df."""
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        for a, b in audio.segments(float(r.duration), seg, min_len):
            rows.append((i, a, b))
    s = pd.DataFrame(rows, columns=["utt", "start", "end"])
    keep = [c for c in ["audio", "path", "label", "corpus", "speaker", "l1", "gender", "fold", "el_split"]
            if c in df.columns]
    return pd.concat([s, df.iloc[s["utt"].to_numpy()][keep].reset_index(drop=True)], axis=1)


class SegmentDataset(Dataset):
    def __init__(self, segs: pd.DataFrame):
        self.paths = segs["path"].tolist() if "path" in segs.columns else segs["audio"].tolist()
        self.start = segs["start"].to_numpy()
        self.end = segs["end"].to_numpy()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return i, audio.load(self.paths[i], self.start[i], self.end[i])


def length_batches(segs: pd.DataFrame, max_seconds: float, max_batch: int) -> list[list[int]]:
    """Batches of similar-length segments, each holding at most `max_seconds` of padded audio."""
    order = np.argsort(-(segs["end"] - segs["start"]).to_numpy(), kind="stable")
    dur = (segs["end"] - segs["start"]).to_numpy()
    batches, cur = [], []
    for i in order:
        longest = dur[cur[0]] if cur else dur[i]
        if cur and (len(cur) + 1 > max_batch or (len(cur) + 1) * longest > max_seconds):
            batches.append(cur)
            cur = []
        cur.append(int(i))
    if cur:
        batches.append(cur)
    return batches


def collate_padded(items):
    idx = torch.tensor([i for i, _ in items])
    lengths = torch.tensor([len(w) for _, w in items])
    wav = torch.zeros(len(items), int(lengths.max()))
    for j, (_, w) in enumerate(items):
        wav[j, : len(w)] = torch.from_numpy(w)
    return idx, wav, lengths


# --------------------------------------------------------------------------- training
class CropDataset(Dataset):
    """Item = (row, crop seconds): a random crop of that length (whole utterance if shorter)."""

    def __init__(self, df: pd.DataFrame):
        self.paths = df["path"].tolist()
        self.dur = df["duration"].to_numpy()
        self.labels = df["label"].to_numpy()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        i, crop = item
        d = self.dur[i]
        start = random.uniform(0, d - crop) if d > crop else 0.0
        w = audio.load(self.paths[i], start, start + crop)
        n = int(crop * SR)
        if len(w) > n:
            w = w[:n]
        return w, int(self.labels[i])


class BalancedCropSampler(Sampler):
    """Infinite stream of batches [(row, crop_seconds), ...]; rows drawn with replacement using the
    group shares; one crop length ~ U(crop_min, crop_max) per batch."""

    def __init__(self, df: pd.DataFrame, shares: dict, batch_size: int, crop_min: float, crop_max: float,
                 seed: int = 0):
        self.p = group_weights(df, shares, by_duration=True, crop=(crop_min + crop_max) / 2)
        self.p = self.p / self.p.sum()
        self.bs, self.cmin, self.cmax = batch_size, crop_min, crop_max
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            crop = float(self.rng.uniform(self.cmin, self.cmax))
            rows = self.rng.choice(len(self.p), size=self.bs, replace=True, p=self.p)
            yield [(int(r), crop) for r in rows]

    def __len__(self):
        return 2**31


def collate_train(items):
    lengths = torch.tensor([len(w) for w, _ in items])
    wav = torch.zeros(len(items), int(lengths.max()))
    for j, (w, _) in enumerate(items):
        wav[j, : len(w)] = torch.from_numpy(w)
    return wav, lengths, torch.tensor([y for _, y in items])


# --------------------------------------------------------------------------- augmentation
FFT_BLOCK = 4096  # FFT lengths are rounded up to a multiple of this: few distinct cuFFT plans, cheap sizes


def _filter(x: torch.Tensor, gain_fn) -> torch.Tensor:
    """Multiply the spectrum of every row by gain_fn(freqs [F]) -> [B, F] or [F], via a zero-padded FFT.
    Falls back to the CPU if cuFFT fails (it can when the GPU is nearly full)."""
    n = x.shape[1]
    m = -(-n // FFT_BLOCK) * FFT_BLOCK

    def run(t):
        spec = torch.fft.rfft(F.pad(t, (0, m - n)), dim=1)
        g = gain_fn(torch.fft.rfftfreq(m, 1 / SR).to(t.device))
        return torch.fft.irfft(spec * g.to(t.device), n=m, dim=1)[:, :n]

    try:
        return run(x)
    except RuntimeError as e:
        if "cuFFT" not in str(e) and "CUFFT" not in str(e):
            raise
        torch.backends.cuda.cufft_plan_cache.clear()
        return run(x.cpu()).to(x.device)


class Augment:
    """Channel augmentation applied on the GPU to break the corpus/channel shortcut:
    random band-limiting (telephone-like to wideband), additive noise at a random SNR, random gain.
    Accent cues (phones, prosody) are left intact: no speed / pitch changes."""

    def __init__(self, p_band: float = 0.5, p_noise: float = 0.5, snr: tuple = (5.0, 30.0),
                 low_cut: tuple = (0.0, 300.0), high_cut: tuple = (3000.0, 8000.0), gain_db: float = 6.0):
        self.p_band, self.p_noise, self.snr = p_band, p_noise, snr
        self.low_cut, self.high_cut, self.gain_db = low_cut, high_cut, gain_db

    def __call__(self, wav: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, n = wav.shape
        dev = wav.device
        mask = (torch.arange(n, device=dev)[None] < lengths[:, None].to(dev)).float()
        if self.p_band > 0:
            lo = torch.empty(b, device=dev).uniform_(*self.low_cut)
            hi = torch.empty(b, device=dev).uniform_(*self.high_cut)
            on = torch.rand(b, device=dev) < self.p_band

            def band(freqs):
                lo_, hi_, on_ = lo.to(freqs.device), hi.to(freqs.device), on.to(freqs.device)
                keep = (freqs[None] >= lo_[:, None]) & (freqs[None] <= hi_[:, None])
                return (keep | ~on_[:, None]).float()

            wav = _filter(wav, band)
        if self.p_noise > 0:
            power = (wav**2 * mask).sum(1) / mask.sum(1).clamp(min=1)
            snr = torch.empty(b, device=dev).uniform_(*self.snr)
            noise = torch.randn_like(wav)
            if torch.rand(()) < 0.5:  # pinkish noise half of the time
                noise = _filter(noise, lambda f: 1 / f.clamp(min=20).sqrt())
                noise = noise / noise.std(dim=1, keepdim=True).clamp(min=1e-8)
            scale = torch.sqrt(power / (10 ** (snr / 10)))
            on = (torch.rand(b, device=dev) < self.p_noise).float()
            wav = wav + noise * (scale * on)[:, None]
        if self.gain_db > 0:
            g = 10 ** (torch.empty(b, device=dev).uniform_(-self.gain_db, self.gain_db) / 20)
            wav = wav * g[:, None]
        return wav * mask


def steps_per_pass(df: pd.DataFrame, batch_size: int, crop: float) -> int:
    return max(1, math.ceil(float(np.maximum(1, df["duration"] / crop).sum()) / batch_size))
