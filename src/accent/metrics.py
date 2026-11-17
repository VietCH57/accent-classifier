"""Metrics. Inputs are utterance-level predictions: columns label, prob (P(vietnamese)), corpus,
speaker, l1. The primary numbers are on L2-ARCTIC only (both classes, unseen speakers);
eLearning is all-Vietnamese, so only its recall is meaningful."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve


def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (probs >= threshold).astype(int)
    out = {"n": int(len(labels)), "acc": float((pred == labels).mean()) if len(labels) else float("nan")}
    if len(np.unique(labels)) == 2:
        out.update(bal_acc=float(balanced_accuracy_score(labels, pred)),
                   macro_f1=float(f1_score(labels, pred, average="macro")),
                   auc=float(roc_auc_score(labels, probs)),
                   eer=eer(labels, probs),
                   vn_recall=float(pred[labels == 1].mean()),
                   nonvn_recall=float(1 - pred[labels == 0].mean()))
    return out


def compute(pred: pd.DataFrame, threshold: float = 0.5) -> dict:
    res: dict = {}
    l2 = pred[pred["corpus"] == "l2arctic"]
    if len(l2):
        res["l2arctic"] = binary_metrics(l2["label"].to_numpy(), l2["prob"].to_numpy(), threshold)
        spk = l2.groupby("speaker").agg(label=("label", "first"), prob=("prob", "mean"), l1=("l1", "first"))
        res["l2arctic_speaker"] = {
            "acc": float(((spk["prob"] >= threshold).astype(int) == spk["label"]).mean()),
            "per_speaker": {s: {"l1": r.l1, "label": int(r.label), "mean_prob": round(float(r.prob), 4),
                                "utt_acc": round(float(((l2[l2.speaker == s]["prob"] >= threshold).astype(int)
                                                        == r.label).mean()), 4)}
                            for s, r in spk.iterrows()},
        }
        res["l2arctic_per_l1_acc"] = {
            l1: float(((g["prob"] >= threshold).astype(int) == g["label"]).mean()) for l1, g in l2.groupby("l1")}
    el = pred[pred["corpus"] == "elearning"]
    if len(el):
        res["elearning_recall"] = float((el["prob"] >= threshold).mean())
        res["elearning_n"] = int(len(el))
    allm = binary_metrics(pred["label"].to_numpy(), pred["prob"].to_numpy(), threshold)
    res["all"] = allm
    return res


def headline(res: dict) -> dict:
    """Flat dict of the numbers we compare runs by."""
    l2 = res.get("l2arctic", {})
    return {"bal_acc": l2.get("bal_acc"), "macro_f1": l2.get("macro_f1"), "auc": l2.get("auc"),
            "eer": l2.get("eer"), "spk_acc": res.get("l2arctic_speaker", {}).get("acc"),
            "el_recall": res.get("elearning_recall")}


def aggregate_segments(seg: pd.DataFrame, prob: np.ndarray, key: str = "utt") -> pd.DataFrame:
    """Average segment probabilities into one prediction per utterance."""
    seg = seg.assign(prob=prob)
    keep = ["label", "corpus", "speaker", "l1", "audio"]
    return seg.groupby(key, sort=False).agg(**{c: (c, "first") for c in keep}, prob=("prob", "mean"),
                                            n_seg=("prob", "size")).reset_index()
