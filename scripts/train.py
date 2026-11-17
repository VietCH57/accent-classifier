#!/usr/bin/env python
"""Stage B: train the accent head on top of a frozen encoder for one fold.

  random crop 3-6 s (balanced sampler, train only) -> channel augmentation (GPU)
  -> frozen encoder (all hidden states) -> learned layer weights -> LayerNorm + Linear
  -> attentive statistics pooling -> MLP -> {non-Vietnamese, Vietnamese}

Evaluation: each dev/test utterance is cut into fixed 4 s segments, P(Vietnamese) is averaged per
utterance. The checkpoint with the best dev balanced accuracy on L2-ARCTIC is kept and evaluated on
the full dev and test sets.

    python scripts/train.py --encoder wavlm_large --fold 0 --run wavlm_asp
    for k in 0 1 2 3; do python scripts/train.py --encoder wavlm_large --fold $k --run wavlm_asp; done
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
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from accent import metrics  # noqa: E402
from accent.data import (Augment, BalancedCropSampler, CropDataset, SegmentDataset, collate_padded,  # noqa: E402
                         collate_train, length_batches, make_segments)
from accent.encoders import build_encoder  # noqa: E402
from accent.model import AccentHead  # noqa: E402
from accent.utils import (parse_shares, read_split, save_json, seed_everything, setup_device,  # noqa: E402
                          setup_logging)


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--metadata", default="metadata")
    g.add_argument("--fold", type=int, required=True)
    g.add_argument("--train_corpora", default="l2arctic,elearning")
    g.add_argument("--shares", default="non_vietnamese:0.5,vn_l2arctic:0.25,vn_elearning:0.25",
                   help="share of training draws per group (train only)")
    g.add_argument("--crop_min", type=float, default=3.0)
    g.add_argument("--crop_max", type=float, default=6.0)
    g.add_argument("--eval_seg", type=float, default=4.0)
    g.add_argument("--eval_min_seg", type=float, default=1.0)
    g.add_argument("--dev_max_per_speaker", type=int, default=300,
                   help="dev utterances per speaker used for model selection during training (0 = all)")
    g = p.add_argument_group("model")
    g.add_argument("--encoder", required=True)
    g.add_argument("--layer", type=int, default=None, help="use one fixed layer instead of the learned weighted sum")
    g.add_argument("--pool", choices=["asp", "meanstd"], default="asp")
    g.add_argument("--proj_dim", type=int, default=256)
    g.add_argument("--hidden", type=int, default=256)
    g.add_argument("--dropout", type=float, default=0.2)
    g = p.add_argument_group("optimisation")
    g.add_argument("--steps", type=int, default=6000)
    g.add_argument("--batch_size", type=int, default=32)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--layer_lr", type=float, default=1e-2, help="lr of the layer-weight logits")
    g.add_argument("--weight_decay", type=float, default=1e-2)
    g.add_argument("--warmup", type=int, default=300)
    g.add_argument("--label_smoothing", type=float, default=0.05)
    g.add_argument("--no_aug", action="store_true")
    g.add_argument("--p_band", type=float, default=0.5)
    g.add_argument("--p_noise", type=float, default=0.5)
    g.add_argument("--snr", type=float, nargs=2, default=[5.0, 30.0])
    g = p.add_argument_group("run")
    g.add_argument("--run", required=True, help="run name: runs/<run>/fold<k>/")
    g.add_argument("--runs_root", default="runs")
    g.add_argument("--eval_every", type=int, default=500)
    g.add_argument("--log_every", type=int, default=50)
    g.add_argument("--eval_batch_seconds", type=float, default=200.0)
    g.add_argument("--num_workers", type=int, default=6)
    g.add_argument("--device", default="cuda")
    g.add_argument("--gpu_memory_fraction", type=float, default=0.5)
    g.add_argument("--no_bf16", action="store_true")
    g.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def subsample_dev(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return df
    l2 = df[df.corpus == "l2arctic"].sample(frac=1.0, random_state=seed).groupby("speaker").head(n)
    return pd.concat([l2, df[df.corpus != "l2arctic"]]).reset_index(drop=True)


@torch.no_grad()
def predict(enc, head, df, a, device, use_bf16) -> pd.DataFrame:
    head.eval()
    segs = make_segments(df, a.eval_seg, a.eval_min_seg)
    loader = DataLoader(SegmentDataset(segs), batch_sampler=length_batches(segs, a.eval_batch_seconds, 64),
                        collate_fn=collate_padded, num_workers=a.num_workers)
    prob = np.zeros(len(segs), dtype=np.float32)
    for idx, wav, lengths in loader:
        wav, lengths = wav.to(device, non_blocking=True), lengths.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            hs, mask = enc(wav, lengths)
            logits = head(hs, mask)
        prob[idx.numpy()] = F.softmax(logits.float(), -1)[:, 1].cpu().numpy()
    head.train()
    return metrics.aggregate_segments(segs, prob)


def lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + np.cos(np.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def main():
    a = parse()
    out = Path(a.runs_root) / a.run / f"fold{a.fold}"
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "train.log")
    log.info("args: %s", json.dumps(vars(a)))
    save_json(vars(a), out / "args.json")
    seed_everything(a.seed)
    device = setup_device(a.device, a.gpu_memory_fraction)
    use_bf16 = device.type == "cuda" and not a.no_bf16

    train = read_split(a.metadata, a.fold, "train")
    train = train[train.corpus.isin(a.train_corpora.split(","))].reset_index(drop=True)
    dev = read_split(a.metadata, a.fold, "dev")
    test = read_split(a.metadata, a.fold, "test")
    dev_sel = subsample_dev(dev, a.dev_max_per_speaker, a.seed)
    for n, d in [("train", train), ("dev", dev), ("dev_sel", dev_sel), ("test", test)]:
        log.info("%-7s %s", n, d.groupby(["label_name", "corpus"]).agg(spk=("speaker", "nunique"), utts=("audio", "size"),
                                                                        hours=("duration", lambda x: round(x.sum() / 3600, 2))).to_dict("index"))

    enc = build_encoder(a.encoder).to(device)
    head = AccentHead(enc.num_layers, enc.dim, a.proj_dim, a.hidden, a.pool, a.dropout, a.layer).to(device)
    log.info("encoder %s (%d layers x %d, frozen) | head params %d", a.encoder, enc.num_layers, enc.dim,
             sum(p.numel() for p in head.parameters()))
    opt = torch.optim.AdamW([
        {"params": [p for n, p in head.named_parameters() if not n.startswith("layers.")], "lr": a.lr},
        {"params": head.layers.parameters(), "lr": a.layer_lr, "weight_decay": 0.0},
    ], lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, a.warmup, a.steps))
    aug = None if a.no_aug else Augment(p_band=a.p_band, p_noise=a.p_noise, snr=tuple(a.snr))

    sampler = BalancedCropSampler(train, parse_shares(a.shares), a.batch_size, a.crop_min, a.crop_max, a.seed)
    loader = DataLoader(CropDataset(train), batch_sampler=sampler, collate_fn=collate_train,
                        num_workers=a.num_workers, pin_memory=device.type == "cuda", persistent_workers=a.num_workers > 0)

    best, history = -1.0, []
    t0, run_loss, run_acc, seen_vn = time.time(), 0.0, 0.0, 0.0
    it = iter(loader)
    for step in range(1, a.steps + 1):
        wav, lengths, y = next(it)
        wav, lengths, y = wav.to(device, non_blocking=True), lengths.to(device), y.to(device)
        if aug is not None:
            wav = aug(wav, lengths)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            with torch.no_grad():
                hs, mask = enc(wav, lengths)
            logits = head(hs, mask)
        loss = F.cross_entropy(logits.float(), y, label_smoothing=a.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()
        sched.step()
        run_loss += loss.item()
        run_acc += (logits.argmax(-1) == y).float().mean().item()
        seen_vn += y.float().mean().item()
        if step % a.log_every == 0:
            n = a.log_every
            el = time.time() - t0
            log.info("step %d/%d | loss %.4f | acc %.3f | vn_frac %.2f | lr %.2e | %.2f s/step | ETA %.0f min", step,
                     a.steps, run_loss / n, run_acc / n, seen_vn / n, sched.get_last_lr()[0], el / step,
                     el / step * (a.steps - step) / 60)
            run_loss = run_acc = seen_vn = 0.0
        if step % a.eval_every == 0 or step == a.steps:
            res = metrics.compute(predict(enc, head, dev_sel, a, device, use_bf16))
            h = metrics.headline(res)
            h["step"] = step
            history.append(h)
            lw = head.layers.weights().detach().cpu().numpy()
            log.info("EVAL dev step %d | bal_acc %.4f | macro_f1 %.4f | auc %.4f | eer %.4f | spk_acc %.3f | el_recall %s"
                     " | top layers %s", step, h["bal_acc"], h["macro_f1"], h["auc"], h["eer"], h["spk_acc"],
                     f"{h['el_recall']:.3f}" if h["el_recall"] is not None else "-", [int(i) for i in np.argsort(-lw)[:5]])
            score = h["bal_acc"] + 1e-3 * h["auc"]
            if score > best:
                best = score
                torch.save({"head": head.state_dict(), "step": step, "args": vars(a),
                            "num_layers": enc.num_layers, "dim": enc.dim}, out / "best.pt")
                log.info("  new best (step %d)", step)
            save_json(history, out / "history.json")
    torch.save({"head": head.state_dict(), "step": a.steps, "args": vars(a), "num_layers": enc.num_layers,
                "dim": enc.dim}, out / "last.pt")

    # final evaluation with the best checkpoint on the complete dev and test sets
    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    head.load_state_dict(ck["head"])
    final = {"best_step": ck["step"], "layer_weights": head.layers.weights().detach().cpu().tolist()}
    for name, df in [("dev", dev), ("test", test)]:
        pred = predict(enc, head, df, a, device, use_bf16)
        pred.to_csv(out / f"pred_{name}.csv", index=False)
        res = metrics.compute(pred)
        final[name] = res
        log.info("FINAL %s | %s | per L1 %s", name, {k: round(v, 4) if v is not None else None
                                                      for k, v in metrics.headline(res).items()},
                 {k: round(v, 3) for k, v in res.get("l2arctic_per_l1_acc", {}).items()})
    save_json(final, out / "metrics.json")
    log.info("Done: %s", out)


if __name__ == "__main__":
    main()
