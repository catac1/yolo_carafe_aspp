#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Run the A0-A4 COCO-Seg ablation in series on 3x RTX 6000 (devices 1,2,3).
#
#   bash experiments/scripts/run_ablation.sh              # all five stages
#   bash experiments/scripts/run_ablation.sh A3 A4        # only these
#   DRY_RUN=1 bash experiments/scripts/run_ablation.sh    # print commands, run nothing
#   BATCH=96 bash experiments/scripts/run_ablation.sh     # override a setting
#   PROBE=1 bash experiments/scripts/run_ablation.sh      # time each stage, train nothing
#
# PROBE=1 trains one epoch on PROBE_FRACTION (default 1%) of the data per stage
# and extrapolates a measured epoch time and total run time, so the cost of the
# full sweep is known before committing days of GPU time. Probe runs write to
# <stage>_probe/ and never touch the real run directories.
#
# Stages already holding a weights/best.pt are skipped so the sweep can be
# re-entered after an interruption; FORCE=1 reruns them from scratch. A stage
# with a weights/last.pt but no best.pt is resumed from that checkpoint.
#
# Settings (override by exporting or prefixing the command):
#   DEVICE=1,2,3   GPUs to train on; GPU 0 is occupied (TRAINING_GUIDE.md §6.0)
#   BATCH=48       global batch, 16 images/GPU; must divide by the GPU count (§6.1).
#                  Validation runs at 2x batch per GPU and is the binding limit.
#   EPOCHS=100     epochs per stage
#   IMGSZ=640      image size
#   WORKERS=8      dataloader workers PER GPU
#   SEED=0         fixed so stages stay comparable
#   DATA=coco.yaml dataset; use coco8-seg.yaml for a fast end-to-end check
#   PROJECT=experiments/results
#   PROBE=0        set to 1 to time stages instead of training them
#   PROBE_FRACTION=0.01   data fraction used by PROBE=1

set -uo pipefail
cd "$(dirname "$0")/../.."

# Run python the same way this script runs yolo: through uv, so the project
# virtualenv is used whether or not it happens to be activated. Set PY to use a
# specific interpreter instead (e.g. PY=/path/to/.venv/bin/python).
py() {
    if [ -n "${PY:-}" ]; then
        "$PY" "$@"
    else
        uv run --no-sync python "$@"
    fi
}
DEVICE="${DEVICE:-1,2,3}"
BATCH="${BATCH:-48}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-0}"
SCALE="${SCALE:-s}"
DATA="${DATA:-coco.yaml}"
PROJECT="${PROJECT:-experiments/results}"
# Ultralytics resolves a relative `project` under <runs_dir>/<task>/, so results would
# land in runs/segment/experiments/results/... and the resume checks below would never
# find them. Absolute keeps save_dir exactly at $PROJECT/$name.
mkdir -p "$PROJECT"
PROJECT="$(py -c "import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())" "$PROJECT")"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
PROBE="${PROBE:-0}"
PROBE_FRACTION="${PROBE_FRACTION:-0.01}"

# stage : architecture yaml : warm-start checkpoint
STAGES=(
    "A0:ultralytics/cfg/models/26/yolo26${SCALE}-seg.yaml:checkpoints/yolo26${SCALE}-seg.pt"
    "A1:experiments/configs/yolo26${SCALE}-seg-carafe.yaml:checkpoints/yolo26${SCALE}-seg-carafe_pretrained.pt"
    "A2:experiments/configs/yolo26${SCALE}-seg-aspp.yaml:checkpoints/yolo26${SCALE}-seg-aspp_pretrained.pt"
    "A3:experiments/configs/yolo26${SCALE}-seg-carafe-aspp.yaml:checkpoints/yolo26${SCALE}-seg-carafe-aspp_pretrained.pt"
    "A4:experiments/configs/yolo26${SCALE}-seg-carafe-aspp-deeplabv3plus.yaml:checkpoints/yolo26${SCALE}-seg-carafe-aspp-deeplabv3plus_pretrained.pt"
)

# ---------------------------------------------------------------- preflight --
want=("$@")
[ ${#want[@]} -eq 0 ] && want=(A0 A1 A2 A3 A4)

selected=()
for w in "${want[@]}"; do
    hit=""
    for s in "${STAGES[@]}"; do
        [ "${s%%:*}" = "$w" ] && hit="$s" && break
    done
    [ -z "$hit" ] && { echo "error: unknown stage '$w' (expected one of A0 A1 A2 A3 A4)" >&2; exit 2; }
    selected+=("$hit")
done

if ! py -c "import cv2, torch, ultralytics" 2>/dev/null; then
    echo "error: the ultralytics environment is not importable." >&2
    echo "       See TRAINING_GUIDE.md section 0:" >&2
    echo "         uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e \".[extra]\"" >&2
    exit 1
fi

case "$SCALE" in
    n|s|m|l|x) ;;
    *) echo "error: SCALE must be one of n s m l x, got '$SCALE'" >&2; exit 2 ;;
esac

ngpu=$(awk -F, '{print NF}' <<< "$DEVICE")
if [ $((BATCH % ngpu)) -ne 0 ]; then
    echo "error: BATCH=$BATCH is not divisible by the $ngpu GPUs in DEVICE=$DEVICE." >&2
    echo "       DDP splits the global batch evenly across devices." >&2
    exit 2
fi

missing=0
for s in "${selected[@]}"; do
    IFS=: read -r stage config ckpt <<< "$s"
    # A scaled name (yolo26s-seg-carafe.yaml) is resolved by Ultralytics against the
    # unscaled file (yolo26-seg-carafe.yaml), which is how one config serves every
    # scale, so accept either spelling here.
    unified="${config/yolo26${SCALE}-/yolo26-}"
    if [ ! -f "$config" ] && [ ! -f "$unified" ]; then
        echo "error: missing config $config (nor $unified)" >&2
        missing=1
    fi
    [ -f "$ckpt" ] || { echo "error: missing checkpoint $ckpt" >&2; missing=1; }
done
if [ "$missing" -ne 0 ]; then
    echo "       Rebuild the checkpoints: bash experiments/scripts/setup_checkpoints.sh" >&2
    exit 1
fi

if [ "$PROBE" = "1" ]; then
    echo "TIMING PROBE: ${want[*]} - 1 epoch on ${PROBE_FRACTION} of $DATA, extrapolated to $EPOCHS epochs"
else
    echo "Ablation: ${want[*]}"
fi
echo "  device=$DEVICE (${ngpu} GPUs, $((BATCH / ngpu)) images/GPU)  batch=$BATCH  epochs=$EPOCHS"
echo "  imgsz=$IMGSZ  workers=$WORKERS (per GPU)  seed=$SEED  scale=$SCALE  data=$DATA"
echo "  project=$PROJECT"
echo

# Decide what to do with a stage from its last.pt alone: Ultralytics stamps
# epoch=-1 into the final, optimizer-stripped checkpoint, so that - not the mere
# existence of best.pt - is what "finished" means. best.pt appears as soon as
# epoch 1 improves fitness, so a run interrupted at epoch 50 has both files.
# Resuming is always the default; only FORCE=1 starts a stage over.
stage_state() {  # $1 = run directory -> "done" | "resume" | "fresh"
    local last="$1/weights/last.pt"
    if [ "$FORCE" = "1" ] || [ ! -f "$last" ]; then
        echo fresh
        return
    fi
    py - "$last" <<'PYSTATE'
import sys
import torch
try:
    ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print("done" if ckpt.get("epoch", -1) == -1 else "resume")
except Exception:
    print("resume")  # unreadable or partial checkpoint: let Ultralytics decide
PYSTATE
}

# ------------------------------------------------------------------- driver --
mkdir -p "$PROJECT/logs"
failed=()
probe_rows=()

for s in "${selected[@]}"; do
    IFS=: read -r stage config ckpt <<< "$s"
    if [ "$PROBE" = "1" ]; then
        name="${stage}_${SCALE}_probe"
    else
        name="${stage}_${SCALE}_3gpu_ddp"
    fi
    out="$PROJECT/$name"
    log="$PROJECT/logs/${name}_$(date +%Y%m%d_%H%M%S).log"

    if [ "$PROBE" = "1" ]; then
        rm -rf "$out"
        # Three epochs with val=False. `fraction` shrinks only the TRAIN split -
        # validation always runs on the full val set - so timing one epoch with
        # validation on and scaling by 1/fraction multiplies a fixed cost by 100.
        # Differencing epoch 2 against epoch 1 cancels startup and validation;
        # three epochs because Ultralytics validates on the final epoch whatever
        # val= says. Validation is timed separately and added back once per epoch.
        args=(segment train
            model="$config" pretrained="$ckpt"
            data="$DATA" epochs=3 fraction="$PROBE_FRACTION" batch="$BATCH" imgsz="$IMGSZ"
            device="$DEVICE" workers="$WORKERS" amp=True seed="$SEED" val=False plots=False
            project="$PROJECT" name="$name" exist_ok=True)
        echo "==> $stage: timing probe"
        if [ "$DRY_RUN" = "1" ]; then
            echo "    yolo ${args[*]}"
            continue
        fi
        if uv run --no-sync yolo "${args[@]}" 2>&1 | tee "$log"; then
            vlog="$PROJECT/logs/${name}_val_$(date +%Y%m%d_%H%M%S).log"
            echo "    $stage: timing one validation pass on the full val split"
            uv run --no-sync yolo segment val model="$out/weights/last.pt" data="$DATA"                 batch="$BATCH" imgsz="$IMGSZ" device="$DEVICE" workers="$WORKERS"                 plots=False > "$vlog" 2>&1
            row=$(py - "$out/results.csv" "$vlog" "$PROBE_FRACTION" "$EPOCHS" "$stage" <<'EOF'
import csv, re, sys
csv_path, val_log, frac, epochs, stage = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
try:
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    key = next(k for k in rows[0] if k.strip() == "time")
    times = [float(r[key].strip()) for r in rows]
except Exception as e:
    print(f"{stage}|ERROR|{e}|")
    raise SystemExit(0)
# Epochs 1 and 2 are both non-final, so neither validated.
train_probe = times[1] - times[0] if len(times) > 2 else times[-1]
train_full = train_probe / frac
val_s = 0.0
try:
    text = open(val_log, encoding="utf-8", errors="replace").read()
    ms = sum(float(x) for x in re.findall(r"([\d.]+)ms", re.search(r"Speed:.*per image", text).group(0)))
    images = int(re.search(r"^\s*all\s+(\d+)\s+\d+", text, re.M).group(1))
    val_s = ms * images / 1000
except Exception:
    pass
epoch_s = train_full + val_s
print(f"{stage}|{train_full / 60:.1f} min|{val_s:.0f}s|{epoch_s / 60:.1f} min|{epoch_s * epochs / 86400:.2f} d")
EOF
)
            probe_rows+=("$row")
            echo "    $stage probe: $(cut -d'|' -f4 <<< "$row") per full epoch"
        else
            echo "    $stage probe FAILED, see $log" >&2
            failed+=("$stage")
        fi
        echo
        continue
    fi

    state=$(stage_state "$out")
    if [ "$state" = "done" ]; then
        echo "==> $stage: already finished all $EPOCHS epochs, skipping. FORCE=1 to retrain."
        continue
    fi

    args=(segment train
        data="$DATA" epochs="$EPOCHS" batch="$BATCH" imgsz="$IMGSZ"
        device="$DEVICE" workers="$WORKERS" amp=True seed="$SEED"
        project="$PROJECT" name="$name" exist_ok=True)

    if [ "$state" = "resume" ]; then
        echo "==> $stage: resuming from $out/weights/last.pt"
        args=(segment train resume model="$out/weights/last.pt")
    else
        echo "==> $stage: $config"
        args+=(model="$config" pretrained="$ckpt")
    fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "    yolo ${args[*]}"
        continue
    fi

    started=$SECONDS
    if uv run --no-sync yolo "${args[@]}" 2>&1 | tee "$log"; then
        echo "    $stage done in $(((SECONDS - started) / 60)) min -> $out"
    else
        echo "    $stage FAILED after $(((SECONDS - started) / 60)) min, see $log" >&2
        failed+=("$stage")
    fi
    echo
done

# ------------------------------------------------------------------ summary --
if [ "$PROBE" = "1" ] && [ ${#probe_rows[@]} -ne 0 ]; then
    echo "Measured timings - $DATA, batch=$BATCH on ${ngpu} GPUs, imgsz=$IMGSZ, ${EPOCHS} epochs"
    echo
    printf '  %-6s %14s %8s %14s %14s
' stage "train/epoch" "val" "epoch" "${EPOCHS} epochs"
    printf '  %-6s %14s %8s %14s %14s
' ------ -------------- -------- -------------- --------------
    total=0
    for r in "${probe_rows[@]}"; do
        IFS='|' read -r s tr vl ep tot <<< "$r"
        printf '  %-6s %14s %8s %14s %14s
' "$s" "$tr" "$vl" "$ep" "$tot"
        total=$(py -c "import sys; print(f'{float(sys.argv[1]) + float(sys.argv[2]):.2f}')" "$total" "${tot% d}")
    done
    printf '  %-6s %14s %8s %14s %12s d
' TOTAL "" "" "" "$total"
    echo
    echo "Training time scales with PROBE_FRACTION; validation is measured once on the"
    echo "full val split and counted once per epoch, as a real run pays it."
fi

if [ ${#failed[@]} -ne 0 ]; then
    echo "Ablation finished with failures: ${failed[*]}" >&2
    exit 1
fi
[ "$PROBE" = "1" ] || echo "Ablation complete. Results under $PROJECT/"
