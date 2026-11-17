#!/usr/bin/env bash
# =====================================================================================
# Full pipeline, end to end. Every step is resumable: re-running the script skips what is
# already done (downloaded data, 16 kHz wavs, extracted features, probes, finished folds).
#
#   cd /home/mdd/vietch/accent-classifier
#   nohup bash scripts/run_all.sh > run_all.out 2>&1 &
#   tail -f logs/run_all.log
#
# Steps
#   0. environment check (venv, torch + CUDA, disk)
#   1. L2-ARCTIC full release (download from HF + unzip if missing)
#   2. metadata + 4 speaker folds                      (build_metadata.py)
#   3. 16 kHz mono wav cache                           (prepare_audio.py)
#   4. features of EVERY encoder                       (extract_features.py)
#   5. probes of EVERY encoder: per layer + weighted, with and without eLearning in train (probe.py)
#   6. rank all encoders on DEV, then full 4-fold training one encoder after another in rank order:
#      top-1 first, then top-2, ... (TOP_K=0: all of them; TOP_K=n: only the first n).
#      results/final_summary.md is rewritten after every encoder, so you can stop at any time.
#   7. (optional, ABLATIONS=1) best encoder without eLearning in train / with mean+std pooling
#   8. final tables -> results/final_summary.md        (summarize.py)
#
# Everything below can be overridden from the environment, e.g.
#   TOP_K=3 ENCODERS="xlsr300m wavlm_large" bash scripts/run_all.sh
#   PHONREC="phonrec_os2=/home/mdd/vietch/MDD/XLSR-300M-Phoneme-Recognition/runs/fullnative_el2a_os2/best" bash scripts/run_all.sh
# =====================================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
L2ARCTIC_ROOT=${L2ARCTIC_ROOT:-/home/mdd/vietch/data/l2arctic_full}
EL_DIR=${EL_DIR:-/home/mdd/lamnh/MDD-2026/data/stage2_combined/server}
EL_TRAIN=${EL_TRAIN:-$EL_DIR/fullnative_el2a_train_ctc.csv}
EL_DEV=${EL_DEV:-$EL_DIR/fullnative_el2a_dev_ctc.csv}
EL_TEST=${EL_TEST:-$EL_DIR/fullnative_el2a_test_ctc.csv}
EL_ROOT=${EL_ROOT:-}                       # prefix for relative eLearning paths (label files only)
AUDIO16K=${AUDIO16K:-/home/mdd/vietch/data/accent16k}
METADATA=${METADATA:-metadata}
FEATURES=${FEATURES:-features}
# encoders: alias (see src/accent/encoders.py) or name=spec
ENCODERS=${ENCODERS:-"xlsr300m wavlm_large hubert_large wav2vec2_large whisper_small whisper_medium ecapa"}
PHONREC=${PHONREC:-}                       # optional "name=/path/to/checkpoint ..." (reported, not selected: leakage)
TOP_K=${TOP_K:-0}                          # 0 = train every ranked encoder in rank order; n = only the top n
ABLATIONS=${ABLATIONS:-0}                  # 1 = also train the two ablations of step 7
DEVICE=${DEVICE:-cuda}
GPU_FRAC=${GPU_FRAC:-0.5}
WORKERS=${WORKERS:-6}
TRAIN_ARGS=${TRAIN_ARGS:-}                 # extra args for train.py, e.g. "--steps 8000"
PROBE_ARGS=${PROBE_ARGS:-}
NO_EL_SHARES="non_vietnamese:0.5,vn_l2arctic:0.5"

mkdir -p logs results
exec > >(tee -a logs/run_all.log) 2>&1
step() { echo; echo "=================== [$(date '+%F %T')] $* ==================="; }
die() { echo "FATAL: $*"; exit 1; }
FAILED=()

# encoder entry -> name / spec
enc_name() { [[ "$1" == *=* ]] && echo "${1%%=*}" || echo "$1"; }
enc_spec() { [[ "$1" == *=* ]] && echo "${1#*=}" || echo "$1"; }
ALL_ENC=($ENCODERS)
for p in $PHONREC; do ALL_ENC+=("${p%%=*}=phonrec:${p#*=}"); done

# ------------------------------------------------------------------------------------- 0
step "0. environment"
[ -x "$PY" ] || die "$PY not found: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
$PY - <<EOF || die "torch / transformers missing or broken (pip install -r requirements.txt)"
import torch, transformers, sklearn, scipy, soundfile, pandas
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| transformers", transformers.__version__)
if "$DEVICE" == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA not available")
EOF
df -h . | tail -1
avail=$(df -Pk . | tail -1 | awk '{print int($4/1024/1024)}')
[ "$avail" -lt 40 ] && echo "WARNING: only ${avail} GB free; features need ~4 GB per encoder"

# ------------------------------------------------------------------------------------- 1
step "1. L2-ARCTIC full release -> $L2ARCTIC_ROOT"
n_wav=$(ls "$L2ARCTIC_ROOT"/*/wav/*.wav 2>/dev/null | wc -l)
if [ "$n_wav" -lt "${MIN_L2_WAVS:-26000}" ]; then
  echo "found $n_wav wavs -> downloading chikingsley/l2-arctic-release-v5.0"
  mkdir -p "$L2ARCTIC_ROOT"
  $PY -m pip install -q -U huggingface_hub
  $PY - <<EOF || die "download failed"
from huggingface_hub import snapshot_download
snapshot_download("chikingsley/l2-arctic-release-v5.0", repo_type="dataset",
                  local_dir="$L2ARCTIC_ROOT/zips", allow_patterns=["*.zip"], ignore_patterns=["suitcase_corpus.zip"])
EOF
  for z in "$L2ARCTIC_ROOT"/zips/*.zip; do unzip -q -n "$z" -d "$L2ARCTIC_ROOT" || die "unzip $z"; done
  n_wav=$(ls "$L2ARCTIC_ROOT"/*/wav/*.wav | wc -l)
  [ "$n_wav" -ge "${MIN_L2_WAVS:-26000}" ] && rm -rf "$L2ARCTIC_ROOT/zips"
fi
echo "L2-ARCTIC wavs: $n_wav"
[ "$n_wav" -ge "${MIN_L2_WAVS:-26000}" ] || die "expected ~26,867 wavs"

# ------------------------------------------------------------------------------------- 2
step "2. metadata"
EL_ROOT_ARG=(); [ -n "$EL_ROOT" ] && EL_ROOT_ARG=(--el_root "$EL_ROOT")
$PY scripts/build_metadata.py --l2arctic_root "$L2ARCTIC_ROOT" --el_train "$EL_TRAIN" --el_dev "$EL_DEV" \
    --el_test "$EL_TEST" "${EL_ROOT_ARG[@]}" --out_dir "$METADATA" --workers "$WORKERS" \
  || die "build_metadata failed (see $METADATA/build_metadata.log)"

# ------------------------------------------------------------------------------------- 3
step "3. 16 kHz wav cache -> $AUDIO16K"
$PY scripts/prepare_audio.py --metadata "$METADATA" --out_dir "$AUDIO16K" --workers "$WORKERS" \
  || die "prepare_audio failed"

# ------------------------------------------------------------------------------------- 4
step "4. features: ${ALL_ENC[*]}"
for e in "${ALL_ENC[@]}"; do
  name=$(enc_name "$e"); spec=$(enc_spec "$e")
  if [ -f "$FEATURES/$name/meta.json" ]; then echo "[$name] features done"; continue; fi
  if [ "$spec" = "ecapa" ]; then
    $PY -c "import speechbrain" 2>/dev/null || $PY -m pip install -q speechbrain || { FAILED+=("$name: speechbrain install"); continue; }
  fi
  extra=()
  case "$spec" in
    whisper_medium|whisper:*medium*|whisper:*large*) extra=(--max_batch 8) ;;
    whisper*) extra=(--max_batch 16) ;;
  esac
  step "4. extract $name ($spec)"
  $PY scripts/extract_features.py --encoder "$spec" --name "$name" --metadata "$METADATA" --out_root "$FEATURES" \
      --num_workers "$WORKERS" --device "$DEVICE" --gpu_memory_fraction "$GPU_FRAC" "${extra[@]}" \
    || { echo "!! extraction failed for $name"; FAILED+=("$name: extract"); }
done

# ------------------------------------------------------------------------------------- 5
DONE_FEATS=()
for e in "${ALL_ENC[@]}"; do n=$(enc_name "$e"); [ -f "$FEATURES/$n/meta.json" ] && DONE_FEATS+=("$FEATURES/$n"); done
[ ${#DONE_FEATS[@]} -gt 0 ] || die "no features extracted"
for variant in main noel; do
  if [ $variant = main ]; then tag=linear; vargs=(); else tag=linear_l2arctic; vargs=(--train_corpora l2arctic --shares "$NO_EL_SHARES"); fi
  todo=()
  for f in "${DONE_FEATS[@]}"; do [ -f "results/probe/$(basename "$f")/summary_$tag.json" ] || todo+=("$f"); done
  [ ${#todo[@]} -eq 0 ] && { echo "probes ($tag) done"; continue; }
  step "5. probe ($tag): ${todo[*]}"
  $PY scripts/probe.py --features "${todo[@]}" --weighted --device "$DEVICE" --gpu_memory_fraction "$GPU_FRAC" \
      "${vargs[@]}" $PROBE_ARGS || FAILED+=("probe $tag")
done

# ------------------------------------------------------------------------------------- 6
step "6. rank encoders on dev (probe select_score) and train the top $TOP_K"
# candidates: everything except ECAPA (single vector, no frames) and phonrec (saw L2-ARCTIC speakers)
CANDIDATES=()
for e in $ENCODERS; do n=$(enc_name "$e"); [ "$(enc_spec "$e")" = ecapa ] || CANDIDATES+=("$n=$(enc_spec "$e")"); done
RANKED=$($PY - "${CANDIDATES[@]}" <<'EOF'
import json, sys
from pathlib import Path
rows = []
for c in sys.argv[1:]:
    name, spec = c.split("=", 1)
    f = Path("results/probe") / name / "summary_linear.json"
    if f.exists():
        rows.append((json.loads(f.read_text())["select_score"], name, spec))
rows.sort(reverse=True)
for s, name, spec in rows:
    print(f"{name}={spec}={s}")
EOF
)
echo "$RANKED" | awk -F= '{printf "  %-16s dev bal_acc %s\n", $1, $NF}'
if [ "$TOP_K" -gt 0 ]; then TOP=($(echo "$RANKED" | head -n "$TOP_K" | awk -F= '{print $1"="$2}'))
else TOP=($(echo "$RANKED" | awk -F= '{print $1"="$2}')); fi
[ ${#TOP[@]} -gt 0 ] || die "no ranked encoders"
echo "training order: ${TOP[*]}"

write_summary() {
  {
    echo "# Accent classifier: final summary ($(date '+%F %T'))"
    echo; echo "Encoder ranking on dev (probe, best of single layer / weighted):"; echo
    echo "$RANKED" | awk -F= '{printf "%d. %s: %s\n", NR, $1, $NF}'
    echo; $PY scripts/summarize.py --split test
    echo; $PY scripts/summarize.py --split dev | sed -n '/Trained heads/,/Accuracy per L1/p' | sed '$d'
  } > results/final_summary.md
}

run_cv() {  # run_cv <run name> <encoder spec> [train args...]
  local run=$1 spec=$2; shift 2
  for k in 0 1 2 3; do
    if [ -f "runs/$run/fold$k/metrics.json" ]; then echo "[$run] fold $k done"; continue; fi
    rm -f "runs/$run/fold$k/train.log"
    step "6. train $run fold $k"
    $PY scripts/train.py --encoder "$spec" --fold "$k" --run "$run" --metadata "$METADATA" --device "$DEVICE" \
        --gpu_memory_fraction "$GPU_FRAC" --num_workers "$WORKERS" "$@" $TRAIN_ARGS \
      || { echo "!! $run fold $k failed"; FAILED+=("$run fold $k"); }
  done
}
rank=0
for t in "${TOP[@]}"; do
  rank=$((rank + 1))
  step "6. top-$rank / ${#TOP[@]}: ${t%%=*}"
  run_cv "${t%%=*}_asp" "${t#*=}"
  write_summary
  echo "summary updated after top-$rank (results/final_summary.md)"
done

# ------------------------------------------------------------------------------------- 7
best=${TOP[0]}
if [ "$ABLATIONS" = 1 ]; then
  step "7. ablations on the best encoder (${best%%=*})"
  run_cv "${best%%=*}_asp_noel" "${best#*=}" --train_corpora l2arctic --shares "$NO_EL_SHARES"
  run_cv "${best%%=*}_meanstd" "${best#*=}" --pool meanstd
else
  echo "step 7 skipped (ABLATIONS=0)"
fi

# ------------------------------------------------------------------------------------- 8
step "8. summary -> results/final_summary.md"
write_summary
cat results/final_summary.md

step "finished"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Steps that FAILED (re-run the script after fixing; finished parts are skipped):"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi
echo "all steps OK"
