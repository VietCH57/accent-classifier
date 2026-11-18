from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def setup_logging(log_file: Path | None = None) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        handlers=handlers, force=True)
    return logging.getLogger("accent")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_device(name: str, memory_fraction: float | None):
    import torch

    device = torch.device(name if (name != "cuda" or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        if memory_fraction:
            torch.cuda.set_per_process_memory_fraction(float(memory_fraction), index)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.cufft_plan_cache[index].max_size = 16  # each cached plan holds GPU workspace
    return device


def read_split(metadata_dir: str | Path, fold: int, split: str) -> pd.DataFrame:
    df = pd.read_csv(Path(metadata_dir) / "folds" / f"fold{fold}" / f"{split}.csv")
    if "audio16k" in df.columns:
        df["path"] = df["audio16k"].where(df["audio16k"].notna() & (df["audio16k"] != ""), df["audio"])
    else:
        df["path"] = df["audio"]
    return df.reset_index(drop=True)


def sample_groups(df: pd.DataFrame) -> pd.Series:
    """Sampling group of every row: non_vietnamese / vn_l2arctic / vn_elearning."""
    return np.where(df["label"] == 0, "non_vietnamese", "vn_" + df["corpus"].astype(str))


def parse_shares(text: str) -> dict[str, float]:
    out = {}
    for part in text.split(","):
        k, v = part.split(":")
        out[k.strip()] = float(v)
    return out


def group_weights(df: pd.DataFrame, shares: dict[str, float], by_duration: bool = False,
                  crop: float = 4.0) -> np.ndarray:
    """Per-row sampling weights so that each group gets its share of draws. Groups that are absent
    are dropped and the remaining shares renormalised. Within a group rows are uniform, or
    proportional to the number of `crop`-second crops they hold when `by_duration`."""
    g = sample_groups(df)
    present = {k: v for k, v in shares.items() if (g == k).any() and v > 0}
    total = sum(present.values())
    w = np.zeros(len(df))
    for k, share in present.items():
        m = g == k
        within = np.maximum(1.0, df.loc[m, "duration"].to_numpy() / crop) if by_duration else np.ones(m.sum())
        w[m] = share / total * within / within.sum()
    return w


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))
