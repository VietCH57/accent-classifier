#!/usr/bin/env python
"""Find the audio and build the metadata / splits for the L2-English accent classifier
(2 classes: Vietnamese vs. non-Vietnamese speakers).

Sources
-------
* L2-ARCTIC: every ``<SPEAKER>/wav/*.wav`` under ``--l2arctic_root`` (all utterances, not only the
  annotated subset). 24 speakers, 6 L1s x 4 speakers (2 M + 2 F). Label = 1 for the 4 Vietnamese
  speakers, 0 otherwise.
* eLearningMDD: all utterances are Vietnamese (label 1). Its train/dev/test split is already
  speaker-disjoint and is kept as given (``--el_train/--el_dev/--el_test``: CSV or XLSX; the audio
  column may be named audio / Path / path; relative paths are resolved against ``--el_root``; if
  the file has a ``domain`` column only rows with ``--el_domain`` are used).

Splits
------
L2-ARCTIC uses 4-fold speaker cross-validation: fold k holds the k-th speaker of every L1 (6
speakers: 1 Vietnamese + 5 others, gender-balanced 3 M / 3 F). In fold k:
test = speakers of fold k, dev = speakers of fold k+1, train = the remaining 12 speakers.
eLearning keeps its own train/dev/test in every fold.
Class balancing happens only at training time (sampler); dev/test are kept complete.

Outputs (``--out_dir``)
-----------------------
utterances.csv            every usable utterance with speaker / L1 / gender / label / duration / sr
speakers.csv              one row per L2-ARCTIC speaker (+ one per eLearning split) with fold and hours
folds/fold{k}/{train,dev,test}.csv
summary.md                counts and hours per fold / split / class / corpus, plus sanity checks
audio_info.json           cache of (frames, sample_rate, channels) per file

    python scripts/build_metadata.py \
        --l2arctic_root /home/mdd/lamnh/MDD-2026/data/l2arctic_release_v5.0 \
        --el_train elearning_train.xlsx --el_dev elearning_dev.csv --el_test elearning_test.csv \
        --el_root /home/mdd/lamnh/MDD-2026 --out_dir metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import soundfile as sf

log = logging.getLogger("build_metadata")

# L2-ARCTIC speaker table (speaker, L1, gender)
L2ARCTIC = [
    ("ABA", "Arabic", "M"), ("SKA", "Arabic", "F"), ("YBAA", "Arabic", "M"), ("ZHAA", "Arabic", "F"),
    ("BWC", "Mandarin", "M"), ("LXC", "Mandarin", "F"), ("NCC", "Mandarin", "F"), ("TXHC", "Mandarin", "M"),
    ("ASI", "Hindi", "M"), ("RRBI", "Hindi", "M"), ("SVBI", "Hindi", "F"), ("TNI", "Hindi", "F"),
    ("HJK", "Korean", "F"), ("HKK", "Korean", "M"), ("YDCK", "Korean", "F"), ("YKWK", "Korean", "M"),
    ("EBVS", "Spanish", "M"), ("ERMS", "Spanish", "M"), ("MBMPS", "Spanish", "F"), ("NJS", "Spanish", "F"),
    ("HQTV", "Vietnamese", "M"), ("PNV", "Vietnamese", "F"), ("THV", "Vietnamese", "F"), ("TLV", "Vietnamese", "M"),
]
L1_ORDER = ["Vietnamese", "Arabic", "Mandarin", "Hindi", "Korean", "Spanish"]
N_FOLDS = 4
LABEL_NAMES = {1: "vietnamese", 0: "non_vietnamese"}
AUDIO_COLS = ("audio", "Path", "path", "wav", "file")
SPLITS = ("train", "dev", "test")


# --------------------------------------------------------------------------- folds
def assign_folds() -> dict[str, int]:
    """fold k gets the k-th speaker of every L1; orders are M,F,M,F rotated per L1 so that every
    fold has 3 male and 3 female speakers."""
    folds = {}
    for i, l1 in enumerate(L1_ORDER):
        spk = [s for s in L2ARCTIC if s[1] == l1]
        males = sorted(s[0] for s in spk if s[2] == "M")
        females = sorted(s[0] for s in spk if s[2] == "F")
        assert len(males) == 2 and len(females) == 2, l1
        order = [males[0], females[0], males[1], females[1]]
        for k in range(N_FOLDS):
            folds[order[(k + i) % N_FOLDS]] = k
    return folds


def split_of(fold_of_speaker: int, k: int) -> str:
    if fold_of_speaker == k:
        return "test"
    if fold_of_speaker == (k + 1) % N_FOLDS:
        return "dev"
    return "train"


# --------------------------------------------------------------------------- audio info
def _info(path: str):
    try:
        i = sf.info(path)
        return [int(i.frames), int(i.samplerate), int(i.channels)]
    except Exception:  # noqa: BLE001
        return None


def audio_info(paths, cache_file: Path, workers: int) -> dict:
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    todo = sorted({p for p in paths if p not in cache})
    if todo:
        log.info("Reading headers of %d files", len(todo))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p, r in zip(todo, ex.map(_info, todo, chunksize=64)):
                cache[p] = r
        cache_file.write_text(json.dumps(cache))
    return cache


# --------------------------------------------------------------------------- sources
def find_l2arctic(root: Path) -> pd.DataFrame:
    if not root.is_dir():
        sys.exit(f"L2-ARCTIC root not found: {root}")
    dirs = {d.name.upper(): d for d in root.iterdir() if d.is_dir()}
    rows, missing = [], []
    for spk, l1, gender in L2ARCTIC:
        d = dirs.get(spk)
        wavs = sorted((d / "wav").glob("*.wav")) if d is not None and (d / "wav").is_dir() else []
        if not wavs:
            missing.append(spk)
            continue
        rows += [dict(audio=str(w), corpus="l2arctic", speaker=spk, l1=l1, gender=gender,
                      label=int(l1 == "Vietnamese"), utt_id=w.stem) for w in wavs]
    if missing:
        log.warning("L2-ARCTIC speakers without <SPK>/wav/*.wav: %s", missing)
    others = sorted(set(dirs) - {s[0] for s in L2ARCTIC})
    if others:
        log.info("Ignored entries in %s: %s", root, others)
    return pd.DataFrame(rows)


def read_table(path: str) -> pd.DataFrame:
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def find_elearning(files: dict[str, str], root: Path | None, domain: str) -> pd.DataFrame:
    rows = []
    for split, f in files.items():
        df = read_table(f)
        col = next((c for c in AUDIO_COLS if c in df.columns), None)
        if col is None:
            sys.exit(f"{f}: no audio column (expected one of {AUDIO_COLS})")
        if "domain" in df.columns:
            n = len(df)
            df = df[df["domain"] == domain]
            log.info("%s: kept %d / %d rows with domain == %s", f, len(df), n, domain)
        df = df[df[col].notna() & (df[col].astype(str).str.strip() != "")]
        tcol = next((c for c in ("Transcript", "text", "transcript") if c in df.columns), None)
        if tcol is not None:
            empty = df[tcol].isna() | (df[tcol].astype(str).str.strip() == "")
            if empty.any():
                log.info("%s: %d rows with empty %s dropped", f, int(empty.sum()), tcol)
            df = df[~empty]
        paths = df[col].astype(str).str.strip()
        if root is not None:
            paths = paths.map(lambda p: p if Path(p).is_absolute() else str(root / p))
        dup = int(paths.duplicated().sum())
        if dup:
            log.warning("%s: %d duplicated paths dropped", f, dup)
        for p in paths.drop_duplicates():
            rows.append(dict(audio=p, corpus="elearning", speaker=f"elearning_{split}", l1="Vietnamese",
                             gender="", label=1, utt_id=Path(p).stem, el_split=split))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- summary
def table(df: pd.DataFrame) -> str:
    g = df.groupby(["label_name", "corpus"])
    t = pd.DataFrame({"speakers": g["speaker"].nunique(), "utts": g.size(), "hours": g["duration"].sum() / 3600})
    t.loc[("TOTAL", ""), :] = [df["speaker"].nunique(), len(df), df["duration"].sum() / 3600]
    t["speakers"] = t["speakers"].astype(int)
    t["utts"] = t["utts"].astype(int)
    return t.round(2).to_markdown()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--l2arctic_root", default="/home/mdd/lamnh/MDD-2026/data/l2arctic_release_v5.0")
    p.add_argument("--el_train", required=True)
    p.add_argument("--el_dev", required=True)
    p.add_argument("--el_test", required=True)
    p.add_argument("--el_root", default=None, help="prefix for relative eLearning paths")
    p.add_argument("--el_domain", default="el_easy", help="used only if the file has a 'domain' column")
    p.add_argument("--out_dir", default="metadata")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "build_metadata.log", mode="w")])

    l2 = find_l2arctic(Path(args.l2arctic_root))
    el = find_elearning({"train": args.el_train, "dev": args.el_dev, "test": args.el_test},
                        Path(args.el_root) if args.el_root else None, args.el_domain)
    df = pd.concat([l2, el], ignore_index=True)
    if df.empty:
        sys.exit("No audio found")

    info = audio_info(df["audio"].tolist(), out / "audio_info.json", args.workers)
    df["frames"], df["sample_rate"], df["channels"] = zip(*[info.get(a) or (None, None, None) for a in df["audio"]])
    bad = df["frames"].isna() | (df["frames"] == 0)
    if bad.any():
        log.warning("%d missing/unreadable/empty files dropped (by corpus: %s)", int(bad.sum()),
                    df.loc[bad, "corpus"].value_counts().to_dict())
        for a in df.loc[bad, "audio"].head(20):
            log.warning("    %s", a)
        df = df[~bad].copy()
    df["duration"] = df["frames"].astype(float) / df["sample_rate"].astype(float)
    df["sample_rate"] = df["sample_rate"].astype(int)
    df["channels"] = df["channels"].astype(int)
    df["label_name"] = df["label"].map(LABEL_NAMES)
    folds = assign_folds()
    df["fold"] = df["speaker"].map(folds).astype("Int64")
    df = df.drop(columns="frames").sort_values(["corpus", "speaker", "audio"]).reset_index(drop=True)

    cols = ["audio", "label", "label_name", "corpus", "speaker", "l1", "gender", "duration",
            "sample_rate", "channels", "fold", "el_split", "utt_id"]
    df[cols].to_csv(out / "utterances.csv", index=False)

    # speakers table
    sp = df.groupby(["corpus", "speaker"]).agg(l1=("l1", "first"), gender=("gender", "first"),
                                               label=("label", "first"), fold=("fold", "first"),
                                               n_utts=("audio", "size"), hours=("duration", lambda x: x.sum() / 3600),
                                               sample_rates=("sample_rate", lambda x: sorted(set(map(int, x)))))
    sp.round(3).to_csv(out / "speakers.csv")

    # folds
    md = ["# Accent classifier metadata", "",
          f"- L2-ARCTIC root: `{args.l2arctic_root}`",
          f"- eLearning files: train `{args.el_train}`, dev `{args.el_dev}`, test `{args.el_test}`", "",
          "## Speakers per fold (L2-ARCTIC)", "",
          pd.DataFrame([dict(fold=k, **{l1: ", ".join(f"{s}({g})" for s, l, g in L2ARCTIC if l == l1 and folds[s] == k)
                                        for l1 in L1_ORDER}) for k in range(N_FOLDS)]).to_markdown(index=False), "",
          "## All data", "", table(df), "",
          "## Sample rates by corpus", "",
          df.groupby(["corpus", "sample_rate"]).size().rename("utts").to_frame().to_markdown(), "",
          "## Duration (s) by corpus", "",
          df.groupby("corpus")["duration"].describe(percentiles=[.1, .5, .9]).round(2).to_markdown(), ""]
    problems = []
    for k in range(N_FOLDS):
        d = out / "folds" / f"fold{k}"
        d.mkdir(parents=True, exist_ok=True)
        parts = {}
        for s in SPLITS:
            l2 = df[df["corpus"] == "l2arctic"]
            l2s = l2[l2["fold"].astype(int).map(lambda f: split_of(f, k)) == s]
            els = df[(df["corpus"] == "elearning") & (df["el_split"] == s)]
            parts[s] = pd.concat([l2s, els])
            parts[s][cols].to_csv(d / f"{s}.csv", index=False)
        for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
            for key in ("speaker", "audio"):
                inter = set(parts[a][key]) & set(parts[b][key])
                if inter:
                    problems.append(f"fold{k}: {key} overlap {a}/{b}: {sorted(inter)[:5]}")
        for s in SPLITS:
            if parts[s]["label"].nunique() < 2:
                problems.append(f"fold{k}/{s}: only one class")
        md += [f"## Fold {k}", ""]
        for s in SPLITS:
            md += [f"### {s}", "", table(parts[s]), ""]
    test_count = df[df.corpus == "l2arctic"].groupby("speaker")["fold"].first().value_counts()
    if len(test_count) != N_FOLDS or set(test_count) != {6}:
        problems.append(f"L2-ARCTIC speakers per fold != 6: {test_count.to_dict()}")
    missing_spk = sorted({s[0] for s in L2ARCTIC} - set(df["speaker"]))
    if missing_spk:
        problems.append(f"L2-ARCTIC speakers missing: {missing_spk}")
    md += ["## Checks", ""] + ([f"- PROBLEM: {x}" for x in problems] or ["- all checks passed"])
    (out / "summary.md").write_text("\n".join(md) + "\n")

    print("\n".join(md[: md.index("## Fold 0")]))
    print(f"Fold tables: {out / 'summary.md'}")
    for x in problems:
        log.error(x)
    log.info("Done: %s", out.resolve())
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
