#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Train ONE ablation stage on ONE GPU, in the foreground. No DDP.
#
#   bash experiments/scripts/run_stage.sh <STAGE> <GPU>
#
# Run one of these per terminal, one terminal per GPU:
#
#   terminal 1:  bash experiments/scripts/run_stage.sh A0 1
#   terminal 2:  bash experiments/scripts/run_stage.sh A1 2
#   terminal 3:  bash experiments/scripts/run_stage.sh A2 3
#
# then, as each finishes, start the next stage in that freed terminal:
#
#   terminal 1:  bash experiments/scripts/run_stage.sh A3 1
#   terminal 2:  bash experiments/scripts/run_stage.sh A4 2
#
# TO RESUME: re-run the exact same command. A stage with weights/last.pt and no
# weights/best.pt resumes from that checkpoint; a finished stage is skipped.
# Nothing else to remember, and one crashed stage never affects the others.
#
# Settings (override by prefixing the command, e.g. BATCH=64 bash ...):
#   BATCH=32       batch for this stage; it owns one GPU, so this is per GPU
#   EPOCHS=100     epochs
#   IMGSZ=640      image size
#   WORKERS=6      dataloader workers; every concurrent stage adds its own
#   SEED=0         fixed so stages stay comparable
#   DATA=coco.yaml dataset; use coco8-seg.yaml for a fast end-to-end check
#   PROJECT=experiments/results
#   FORCE=0        set to 1 to retrain a finished stage from scratch
#   PROBE=0        set to 1 to time the stage on PROBE_FRACTION instead of training
#   PROBE_FRACTION=0.01

set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-python}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-6}"
SEED="${SEED:-0}"
DATA="${DATA:-coco.yaml}"
PROJECT="${PROJECT:-experiments/results}"
FORCE="${FORCE:-0}"
PROBE="${PROBE:-0}"
PROBE_FRACTION="${PROBE_FRACTION:-0.01}"

# stage : architecture yaml : warm-start checkpoint
STAGES=(
    "A0:ultralytics/cfg/models/26/yolo26-seg.yaml:checkpoints/yolo26s-seg.pt"
    "A1:experiments/configs/yolo26s-seg-carafe.yaml:checkpoints/yolo26s-seg-carafe_pretrained.pt"
    "A2:experiments/configs/yolo26s-seg-aspp.yaml:checkpoints/yolo26s-seg-aspp_pretrained.pt"
    "A3:experiments/configs/yolo26s-seg-carafe-aspp.yaml:checkpoints/yolo26s-seg-carafe-aspp_pretrained.pt"
    "A4:experiments/configs/yolo26s-seg-carafe-aspp-deeplabv3plus.yaml:checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt"
)

usage() {
    echo "usage: bash experiments/scripts/run_stage.sh <STAGE> <GPU>" >&2
    echo "       STAGE  one of A0 A1 A2 A3 A4" >&2
    echo "       GPU    physical GPU index as nvidia-smi reports it, e.g. 1" >&2
    echo >&2
    echo "example: bash experiments/scripts/run_stage.sh A0 1" >&2
}

# ---------------------------------------------------------------- preflight --
if [ $# -ne 2 ]; then
    usage
    exit 2
fi
stage="$1"
gpu="$2"

config=""
ckpt=""
for s in "${STAGES[@]}"; do
    if [ "${s%%:*}" = "$stage" ]; then
        IFS=: read -r _ config ckpt <<< "$s"
        break
    fi
done
if [ -z "$config" ]; then
    echo "error: unknown stage '$stage'" >&2
    usage
    exit 2
fi

case "$gpu" in
    ''|*[!0-9]*) echo "error: GPU must be a single index, got '$gpu'" >&2; usage; exit 2 ;;
esac

if ! "$PY" -c "import cv2, torch, ultralytics" 2>/dev/null; then
    echo "error: the ultralytics environment is not importable with '$PY'." >&2
    echo "       See TRAINING_GUIDE.md section 0:" >&2
    echo "         uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e '.[extra]'" >&2
    exit 1
fi

[ -f "$config" ] || { echo "error: missing config $config" >&2; exit 1; }
if [ ! -f "$ckpt" ]; then
    echo "error: missing checkpoint $ckpt" >&2
    echo "       Rebuild it: bash experiments/scripts/setup_checkpoints.sh" >&2
    exit 1
fi

# ------------------------------------------------------------------- launch --
if [ "$PROBE" = "1" ]; then name="${stage}_probe"; else name="${stage}_1gpu"; fi
out="$PROJECT/$name"
mkdir -p "$PROJECT/logs"
log="$PROJECT/logs/${name}_$(date +%Y%m%d_%H%M%S).log"

if [ "$PROBE" != "1" ] && [ -f "$out/weights/best.pt" ] && [ "$FORCE" != "1" ]; then
    echo "$stage is already complete: $out/weights/best.pt"
    echo "Pass FORCE=1 to retrain it from scratch."
    exit 0
fi

declare -a args
if [ "$PROBE" = "1" ]; then
    rm -rf "$out"
    args=(segment train model="$config" pretrained="$ckpt"
        data="$DATA" epochs=1 fraction="$PROBE_FRACTION" batch="$BATCH" imgsz="$IMGSZ"
        device=0 workers="$WORKERS" amp=True seed="$SEED"
        project="$PROJECT" name="$name")
    mode="timing probe on $PROBE_FRACTION of $DATA"
elif [ -f "$out/weights/last.pt" ] && [ "$FORCE" != "1" ]; then
    args=(segment train resume model="$out/weights/last.pt")
    mode="resuming from $out/weights/last.pt"
else
    args=(segment train model="$config" pretrained="$ckpt"
        data="$DATA" epochs="$EPOCHS" batch="$BATCH" imgsz="$IMGSZ"
        device=0 workers="$WORKERS" amp=True seed="$SEED"
        project="$PROJECT" name="$name")
    mode="training from $ckpt"
fi

echo "=============================================================="
echo " stage   $stage   ($config)"
echo " gpu     $gpu (pinned via CUDA_VISIBLE_DEVICES; the stage sees it as device 0)"
echo " mode    $mode"
echo " batch   $BATCH   epochs $EPOCHS   imgsz $IMGSZ   workers $WORKERS   seed $SEED"
echo " out     $out"
echo " log     $log"
echo "=============================================================="
echo

# CUDA_VISIBLE_DEVICES pins this process to one physical GPU, so concurrent
# stages in other terminals cannot collide on the same card.
CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync yolo "${args[@]}" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}

echo
if [ "$status" -eq 0 ]; then
    echo "$stage finished. Results: $out"
    if [ "$PROBE" = "1" ] && [ -f "$out/results.csv" ]; then
        "$PY" - "$out/results.csv" "$PROBE_FRACTION" "$EPOCHS" "$stage" <<'PYEOF'
import csv, sys

csv_path, frac, epochs, stage = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
try:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    key = next(k for k in rows[-1] if k.strip() == "time")
    probe_s = float(rows[-1][key].strip())
except Exception:
    raise SystemExit(0)
epoch_s = probe_s / frac
print(f"{stage} measured: {probe_s:.0f}s probe -> {epoch_s / 60:.1f} min/epoch -> "
      f"{epoch_s * epochs / 3600 / 24:.2f} days for {epochs} epochs on one GPU")
PYEOF
    fi
else
    echo "$stage FAILED (exit $status). Log: $log" >&2
    echo "Re-run the same command to resume from the last checkpoint." >&2
fi
exit "$status"
