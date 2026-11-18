#!/usr/bin/env python
"""Stage A, step 2: probe cached features (from extract_features.py) on the 4 speaker folds.

For every encoder and every fold:
  * --probe linear  : logistic regression on the mean+std features of each single layer
  * --probe mlp     : 1-hidden-layer MLP on each single layer
  * --weighted      : additionally one probe on a learned softmax-weighted sum of ALL layers
Training uses class/corpus-balanced sample weights (train split only); dev and test are complete.
Segment probabilities are averaged per utterance; metrics are utterance-level on L2-ARCTIC
(balanced accuracy, macro-F1, AUC, EER, speaker-level accuracy) + eLearning recall.
Per fold the layer is chosen on dev; test numbers are reported for that layer.

    python scripts/probe.py --features features/xlsr300m features/wavlm_large --weighted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from accent import metrics  # noqa: E402
from accent.utils import group_weights, parse_shares, setup_device, setup_logging  # noqa: E402

N_FOLDS = 4


def fold_splits(segs: pd.DataFrame, k: int, train_corpora: set[str]) -> dict[str, np.ndarray]:
    l2 = segs["corpus"] == "l2arctic"
    fold = segs["fold"].fillna(-1).astype(int)
    el = segs["corpus"] == "elearning"
    split = {
        "train": (l2 & (fold != k) & (fold != (k + 1) % N_FOLDS)) | (el & (segs["el_split"] == "train")),
        "dev": (l2 & (fold == (k + 1) % N_FOLDS)) | (el & (segs["el_split"] == "dev")),
        "test": (l2 & (fold == k)) | (el & (segs["el_split"] == "test")),
    }
    split["train"] &= segs["corpus"].isin(train_corpora)
    return {s: np.flatnonzero(m.to_numpy()) for s, m in split.items()}


class Probe(nn.Module):
    def __init__(self, dim: int, num_layers: int = 1, hidden: int = 0, dropout: float = 0.3):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers)) if num_layers > 1 else None
        self.net = (nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 2)) if hidden else nn.Linear(dim, 2))

    def forward(self, x):
        if self.layer_logits is not None:  # x: [B, L, F]
            x = torch.einsum("l,blf->bf", F.softmax(self.layer_logits, 0), x.float())
        return self.net(x.float())


def fit_predict(get, n: int, dim: int, idx: dict, w: torch.Tensor, y: torch.Tensor, a, num_layers=1):
    """get(rows: LongTensor on CPU) -> standardised features of those rows on the device, [B, F] or [B, L, F].
    Only one mini-batch is on the GPU at a time. Returns P(vn) for all n rows and the model."""
    dev = w.device
    torch.manual_seed(a.seed)
    model = Probe(dim, num_layers, a.hidden if a.probe == "mlp" else 0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    tr = torch.as_tensor(idx["train"])
    gen = torch.Generator().manual_seed(a.seed)
    for _ in range(a.epochs):
        for b in tr[torch.randperm(len(tr), generator=gen)].split(a.batch_size):
            bd = b.to(dev)
            loss = (F.cross_entropy(model(get(b)), y[bd], reduction="none") * w[bd]).sum() / w[bd].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        prob = torch.cat([F.softmax(model(get(b)), -1)[:, 1] for b in torch.arange(n).split(a.eval_batch)])
    return prob.cpu().numpy(), model


def train_stats(x: torch.Tensor, rows: np.ndarray, dev, chunk: int = 2048):
    """Mean / std over the train rows of a CPU tensor [N, ...], computed chunk-wise on the device."""
    s = s2 = None
    for b in torch.as_tensor(rows).split(chunk):
        t = x[b].to(dev).double()
        s = t.sum(0) if s is None else s + t.sum(0)
        s2 = (t * t).sum(0) if s2 is None else s2 + (t * t).sum(0)
    n = len(rows)
    mean = s / n
    std = (s2 / n - mean * mean).clamp(min=0).sqrt().clamp(min=1e-4)
    return mean.float(), std.float()


def evaluate(segs, prob, rows) -> dict:
    utt = metrics.aggregate_segments(segs.iloc[rows], prob[rows])
    return metrics.compute(utt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", nargs="+", required=True, help="feature folders (features/<encoder>)")
    p.add_argument("--out_root", default="results/probe")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--layers", type=int, nargs="*", default=None, help="only these layers (default: all)")
    p.add_argument("--weighted", action="store_true", help="also probe a learned weighted sum of all layers")
    p.add_argument("--probe", choices=["linear", "mlp"], default="linear")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--train_corpora", default="l2arctic,elearning",
                   help="'l2arctic' alone = ablation without eLearning in train")
    p.add_argument("--shares", default="non_vietnamese:0.5,vn_l2arctic:0.25,vn_elearning:0.25")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch", type=int, default=2048, help="rows per prediction batch (lower = less VRAM)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu_memory_fraction", type=float, default=0.5)
    a = p.parse_args()

    device = setup_device(a.device, a.gpu_memory_fraction)
    shares = parse_shares(a.shares)
    corpora = set(a.train_corpora.split(","))
    tag = f"{a.probe}" + ("" if corpora == {"l2arctic", "elearning"} else "_" + "+".join(sorted(corpora)))
    overall = []
    for fdir in map(Path, a.features):
        name = fdir.name
        out = Path(a.out_root) / name
        log = setup_logging(out / f"probe_{tag}.log")
        meta = json.loads((fdir / "meta.json").read_text())
        segs = pd.read_csv(fdir / "segments.csv", low_memory=False)
        segs["utt"] = segs["utt"].astype(int)
        log.info("== %s: %d segments, %d layers, dim %d", name, len(segs), meta["num_layers"], meta["dim"])
        feats = np.load(fdir / "feats.npy", mmap_mode="r")
        L = feats.shape[1]
        layers = a.layers if a.layers is not None else list(range(L))
        y = torch.as_tensor(segs["label"].to_numpy(), device=device)
        rows = []
        splits, weights = {}, {}
        for k in a.folds:
            splits[k] = idx = fold_splits(segs, k, corpora)
            w = torch.zeros(len(segs), device=device)
            w[torch.as_tensor(idx["train"], device=device)] = torch.as_tensor(
                group_weights(segs.iloc[idx["train"]], shares), device=device, dtype=torch.float)
            weights[k] = w
        for layer in layers:
            x_layer = torch.from_numpy(np.ascontiguousarray(feats[:, layer]))  # CPU [N, F] fp16
            for k in a.folds:
                idx = splits[k]
                mean, std = train_stats(x_layer, idx["train"], device)
                prob, _ = fit_predict(lambda b: (x_layer[b].to(device).float() - mean) / std, len(segs),
                                      x_layer.shape[1], idx, weights[k], y, a)
                r = {"fold": k, "layer": layer}
                for s in ("dev", "test"):
                    r.update({f"{s}_{m}": v for m, v in metrics.headline(evaluate(segs, prob, idx[s])).items()})
                rows.append(r)
                log.info("layer %2d fold %d | dev bal_acc %.4f auc %.4f | test bal_acc %.4f auc %.4f el_rec %s",
                         layer, k, r["dev_bal_acc"], r["dev_auc"], r["test_bal_acc"], r["test_auc"],
                         f"{r['test_el_recall']:.3f}" if r["test_el_recall"] is not None else "-")
        if a.weighted and L > 1:
            x_all = torch.from_numpy(np.ascontiguousarray(feats))  # CPU [N, L, F] fp16 (RAM, not VRAM)
            for k in a.folds:
                idx = splits[k]
                mean, std = train_stats(x_all, idx["train"], device)  # [L, F]
                prob, model = fit_predict(lambda b: (x_all[b].to(device).float() - mean) / std, len(segs),
                                          x_all.shape[2], idx, weights[k], y, a, num_layers=L)
                lw = F.softmax(model.layer_logits.detach(), 0).cpu().numpy()
                r = {"fold": k, "layer": "weighted", "layer_weights": json.dumps([round(float(v), 4) for v in lw])}
                for s in ("dev", "test"):
                    res = evaluate(segs, prob, idx[s])
                    r.update({f"{s}_{m}": v for m, v in metrics.headline(res).items()})
                    if s == "test":
                        r["test_per_l1"] = json.dumps({k2: round(v, 4) for k2, v in
                                                       res.get("l2arctic_per_l1_acc", {}).items()})
                rows.append(r)
                log.info("weighted fold %d | dev bal_acc %.4f | test bal_acc %.4f auc %.4f | top layers %s", k,
                         r["dev_bal_acc"], r["test_bal_acc"], r["test_auc"], [int(i) for i in np.argsort(-lw)[:5]])
        torch.cuda.empty_cache() if device.type == "cuda" else None
        df = pd.DataFrame(rows)
        df.to_csv(out / f"probe_{tag}.csv", index=False)

        # per-layer mean over folds
        single = df[df["layer"] != "weighted"].copy()
        per_layer = single.groupby("layer")[["dev_bal_acc", "test_bal_acc", "test_auc", "test_el_recall"]].mean()
        log.info("\nPer-layer mean over folds (%s):\n%s", name, per_layer.round(4).to_string())
        # best layer per fold chosen on dev
        best = single.loc[single.groupby("fold")["dev_bal_acc"].idxmax()]
        summ = {"encoder": name, "probe": tag,
                "best_layer_per_fold": best["layer"].tolist(),
                "best_layer_mean_dev": int(per_layer["dev_bal_acc"].idxmax()),
                "dev_bal_acc": round(float(per_layer["dev_bal_acc"].max()), 4)}
        summ["select_score"] = summ["dev_bal_acc"]  # dev only: used by run_all.sh to rank encoders
        for m in ("bal_acc", "macro_f1", "auc", "eer", "spk_acc", "el_recall"):
            v = best[f"test_{m}"].astype(float)
            summ[f"test_{m}"] = f"{v.mean():.4f} ± {v.std(ddof=0):.4f}"
        if a.weighted:
            wt = df[df["layer"] == "weighted"]
            for m in ("bal_acc", "auc", "el_recall"):
                v = wt[f"test_{m}"].astype(float)
                summ[f"weighted_test_{m}"] = f"{v.mean():.4f} ± {v.std(ddof=0):.4f}"
            if len(wt):
                summ["weighted_dev_bal_acc"] = round(float(wt["dev_bal_acc"].astype(float).mean()), 4)
                summ["select_score"] = max(summ["select_score"], summ["weighted_dev_bal_acc"])
        (out / f"summary_{tag}.json").write_text(json.dumps(summ, indent=2))
        log.info("Summary: %s", json.dumps(summ, indent=2))
        overall.append(summ)

    print("\n" + pd.DataFrame(overall).to_markdown(index=False))


if __name__ == "__main__":
    main()
