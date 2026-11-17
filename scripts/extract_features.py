#!/usr/bin/env python
"""Stage A, step 1: run a frozen encoder once over every utterance (cut into fixed segments) and
store mean+std pooled hidden states of EVERY layer:

    features/<encoder>/feats.npy      float16 [N_segments, L, 2*D]   (memory-mapped)
    features/<encoder>/segments.csv   one row per segment (utterance, start, end, label, speaker, fold ...)
    features/<encoder>/meta.json

All folds share these features (the encoder is not trained), so probe.py can compare encoders and
layers on the 4 folds in minutes.

    python scripts/extract_features.py --encoder xlsr300m
    python scripts/extract_features.py --encoder phonrec:/path/to/runs/fullnative_el2a_os2/best --name phonrec_os2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from accent.data import SegmentDataset, collate_padded, length_batches, make_segments  # noqa: E402
from accent.encoders import build_encoder, safe_name  # noqa: E402
from accent.utils import setup_device, setup_logging  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", required=True)
    p.add_argument("--name", default=None, help="output folder name (default: from --encoder)")
    p.add_argument("--metadata", default="metadata")
    p.add_argument("--out_root", default="features")
    p.add_argument("--seg_sec", type=float, default=4.0)
    p.add_argument("--min_sec", type=float, default=1.0, help="drop trailing pieces shorter than this")
    p.add_argument("--batch_seconds", type=float, default=200.0, help="max padded audio seconds per batch")
    p.add_argument("--max_batch", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu_memory_fraction", type=float, default=0.5)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N utterances")
    a = p.parse_args()

    out = Path(a.out_root) / (a.name or safe_name(a.encoder))
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "extract.log")
    if (out / "meta.json").exists():
        log.info("%s already done (delete meta.json to redo)", out)
        return

    utt = pd.read_csv(Path(a.metadata) / "utterances.csv")
    utt["path"] = utt["audio16k"] if "audio16k" in utt.columns else utt["audio"]
    if a.limit:
        utt = utt.groupby("corpus").head(a.limit).reset_index(drop=True)
    segs = make_segments(utt, a.seg_sec, a.min_sec)
    log.info("%d utterances -> %d segments (%.1f h)", len(utt), len(segs), (segs.end - segs.start).sum() / 3600)

    device = setup_device(a.device, a.gpu_memory_fraction)
    enc = build_encoder(a.encoder).to(device)
    L, D = enc.num_layers, enc.dim
    log.info("encoder %s: %d layers x %d dims -> %.1f GB of features", a.encoder, L, D, len(segs) * L * 2 * D * 2 / 1e9)

    feats = np.lib.format.open_memmap(out / "feats.npy", mode="w+", dtype=np.float16, shape=(len(segs), L, 2 * D))
    batches = length_batches(segs, a.batch_seconds, a.max_batch)
    loader = DataLoader(SegmentDataset(segs), batch_sampler=batches, collate_fn=collate_padded,
                        num_workers=a.num_workers, pin_memory=device.type == "cuda")
    use_bf16 = device.type == "cuda" and not a.no_bf16
    t0, done = time.time(), 0
    for k, (idx, wav, lengths) in enumerate(loader):
        wav, lengths = wav.to(device, non_blocking=True), lengths.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            hs, mask = enc(wav, lengths)
        hs = hs.float()
        m = mask[:, None, :, None].float()
        n = m.sum(2).clamp(min=1)
        mean = (hs * m).sum(2) / n
        std = (((hs - mean[:, :, None]) ** 2 * m).sum(2) / n).clamp(min=0).sqrt()
        feats[idx.numpy()] = torch.cat([mean, std], -1).cpu().numpy().astype(np.float16)
        done += len(idx)
        if k % 50 == 0 or done == len(segs):
            el = time.time() - t0
            log.info("%d/%d segments | %.0f s elapsed | ETA %.0f s", done, len(segs), el, el / done * (len(segs) - done))
    feats.flush()
    segs.to_csv(out / "segments.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(encoder=a.encoder, num_layers=L, dim=D, seg_sec=a.seg_sec,
                                                   min_sec=a.min_sec, n_segments=len(segs),
                                                   n_utterances=len(utt), seconds=time.time() - t0), indent=2))
    log.info("Done: %s", out)


if __name__ == "__main__":
    main()
