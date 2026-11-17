#!/usr/bin/env python
"""Convert every utterance in the metadata to 16 kHz mono 16-bit WAV once (L2-ARCTIC is 44.1 kHz;
resampling it on the fly every epoch would make the CPU the bottleneck), then add an ``audio16k``
column to metadata/utterances.csv and to every fold CSV. Existing files are skipped.

    python scripts/prepare_audio.py --metadata metadata --out_dir /home/mdd/vietch/data/accent16k
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

from accent import audio  # noqa: E402


def target_path(out: Path, row) -> Path:
    group = row.speaker if row.corpus == "l2arctic" else row.el_split
    h = hashlib.md5(row.audio.encode()).hexdigest()[:6]
    return out / row.corpus / str(group) / f"{Path(row.audio).stem}_{h}.wav"


def convert(job):
    src, dst = job
    if Path(dst).exists():
        return None
    try:
        wav = audio.load(src)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        tmp = dst + ".tmp.wav"
        sf.write(tmp, wav, audio.SR, subtype="PCM_16")
        Path(tmp).rename(dst)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{src}: {e}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", default="metadata")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()

    meta, out = Path(a.metadata), Path(a.out_dir)
    utt = pd.read_csv(meta / "utterances.csv")
    utt["audio16k"] = [str(target_path(out, r)) for r in utt.itertuples(index=False)]
    assert utt["audio16k"].is_unique
    jobs = list(zip(utt["audio"], utt["audio16k"]))
    errors = []
    with Pool(a.workers) as pool:
        for k, err in enumerate(pool.imap_unordered(convert, jobs, chunksize=32), 1):
            if err:
                errors.append(err)
            if k % 2000 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} files", flush=True)
    if errors:
        print(f"{len(errors)} errors, e.g.:\n  " + "\n  ".join(errors[:10]))
        sys.exit(1)

    m = dict(zip(utt["audio"], utt["audio16k"]))
    utt.to_csv(meta / "utterances.csv", index=False)
    for f in sorted((meta / "folds").glob("fold*/*.csv")):
        df = pd.read_csv(f)
        df["audio16k"] = df["audio"].map(m)
        assert df["audio16k"].notna().all(), f
        df.to_csv(f, index=False)
    print(f"Done: {len(utt)} files in {out}; added column audio16k to {meta}")


if __name__ == "__main__":
    main()
