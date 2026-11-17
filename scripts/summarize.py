#!/usr/bin/env python
"""Mean ± std over folds for every run in runs/ (from runs/<run>/fold<k>/metrics.json) and every
probe summary in results/probe/.

    python scripts/summarize.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from accent import metrics  # noqa: E402

COLS = ["bal_acc", "macro_f1", "auc", "eer", "spk_acc", "el_recall"]


def fmt(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return f"{np.mean(v) * 100:.2f} ± {np.std(v) * 100:.2f}" if v else "-"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", default="runs")
    p.add_argument("--probe_root", default="results/probe")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    a = p.parse_args()

    rows, l1rows = [], []
    for run in sorted(Path(a.runs_root).glob("*")):
        files = sorted(run.glob("fold*/metrics.json"))
        if not files:
            continue
        hs, l1 = [], {}
        for f in files:
            res = json.loads(f.read_text())[a.split]
            hs.append(metrics.headline(res))
            for k, v in res.get("l2arctic_per_l1_acc", {}).items():
                l1.setdefault(k, []).append(v)
        rows.append({"run": run.name, "folds": len(files), **{c: fmt([h[c] for h in hs]) for c in COLS}})
        l1rows.append({"run": run.name, **{k: fmt(v) for k, v in sorted(l1.items())}})
    if rows:
        print(f"## Trained heads ({a.split}, %; mean ± std over folds; L2-ARCTIC utterance-level)\n")
        print(pd.DataFrame(rows).to_markdown(index=False))
        print(f"\n## Accuracy per L1 ({a.split}, %)\n")
        print(pd.DataFrame(l1rows).to_markdown(index=False))
    probes = [json.loads(f.read_text()) for f in sorted(Path(a.probe_root).glob("*/summary_*.json"))]
    if probes:
        print("\n## Probes (test, chosen layer per fold on dev)\n")
        print(pd.DataFrame(probes).to_markdown(index=False))
    if not rows and not probes:
        print("nothing found")


if __name__ == "__main__":
    main()
