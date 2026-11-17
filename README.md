# L2-English accent classifier (Vietnamese vs. non-Vietnamese)

Pretrained speech encoder (XLS-R / WavLM / HuBERT / Whisper / ...) → pooling → small classifier, 2 classes:
`1 = vietnamese`, `0 = non_vietnamese`.

## Data

| corpus | speakers | label | split |
|---|---|---|---|
| L2-ARCTIC v5.0 (full `<SPK>/wav/*.wav`) | 24 = 6 L1 × (2 M + 2 F) | 1 for HQTV, PNV, THV, TLV; else 0 | 4-fold speaker CV |
| eLearningMDD (easy) | — | 1 (all Vietnamese) | official train/dev/test (speaker-disjoint), same in every fold |

L2-ARCTIC fold *k*: **test** = the 6 speakers of fold *k* (1 per L1, 3 M / 3 F), **dev** = fold *k+1*,
**train** = the other 12. Class balancing is done only by the training sampler; dev/test are complete.
Report balanced accuracy / macro-F1 on L2-ARCTIC (mean ± std over 4 folds), eLearning recall separately.

## 1. Build metadata

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/build_metadata.py \
    --l2arctic_root /home/mdd/lamnh/MDD-2026/data/l2arctic_release_v5.0 \
    --el_train <train.xlsx|csv> --el_dev <dev.csv> --el_test <test.csv> \
    --el_root <prefix for relative paths> \
    --out_dir metadata
```

* eLearning files may be the label files (`Path, Canonical, Transcript`) or the combined CTC CSVs
  (`audio, text, domain`; only rows with `domain == --el_domain`, default `el_easy`, are kept).
  Rows with empty transcript and missing/unreadable audio are dropped (and logged).
* Output in `metadata/`: `utterances.csv`, `speakers.csv`, `folds/fold{0..3}/{train,dev,test}.csv`,
  `summary.md` (counts / hours per fold, split, class, corpus + overlap checks), `build_metadata.log`.
  Exit code 1 if any check fails.

CSV columns: `audio,label,label_name,corpus,speaker,l1,gender,duration,sample_rate,channels,fold,el_split,utt_id`.
