"""Audio I/O: everything is converted to 16 kHz mono float32."""

from __future__ import annotations

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000


def resample(wav: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    if sr == target:
        return wav
    g = gcd(sr, target)
    return resample_poly(wav, target // g, sr // g).astype(np.float32)


def load(path: str, start: float = 0.0, end: float | None = None) -> np.ndarray:
    """Load [start, end) seconds of a file as 16 kHz mono float32 (only that part is read)."""
    info = sf.info(path)
    sr = info.samplerate
    a = int(round(start * sr))
    b = info.frames if end is None else min(info.frames, int(round(end * sr)))
    wav, _ = sf.read(path, start=a, stop=b, dtype="float32", always_2d=True)
    return resample(wav.mean(axis=1), sr)


def segments(duration: float, seg: float, min_len: float) -> list[tuple[float, float]]:
    """Non-overlapping windows of `seg` s covering the utterance. An utterance shorter than `seg`
    is one segment; a trailing piece shorter than `min_len` is merged into the previous window
    by shifting that window to the end of the file (so every segment is <= seg long)."""
    if duration <= seg:
        return [(0.0, duration)]
    out, t = [], 0.0
    while t + seg <= duration + 1e-6:
        out.append((t, t + seg))
        t += seg
    rest = duration - t
    if rest >= min_len:
        out.append((duration - seg, duration))
    return out
