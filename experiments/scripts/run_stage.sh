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
#   BATCH=16       batch for this stage; it owns one GPU, so this is per GPU.
#                  NOTE validation runs at BATCH*2 and is the binding limit (§6.1)
#   EPOCHS=100     epochs
#   IMGSZ=640      image size
#   WORKERS=6      dataloader workers; every concurrent stage adds its own
#   CACHE=         image cache: 'ram', 'disk', or empty for none. 'ram' needs about
#                  2x the decoded dataset size and silently falls back if short;
#                  'disk' writes .npy beside the images and needs write access
#   SEED=0         fixed so stages stay comparable
#   DATA=coco.yaml dataset; use coco8-seg.yaml for a fast end-to-end check
#   PROJECT=experiments/results
#   FORCE=0        set to 1 to retrain a finished stage from scratch
#   PROBE=0        set to 1 to time the stage on PROBE_FRACTION instead of training
#   PROBE_FRACTION=0.01

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
BATCH="${BATCH:-16}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-6}"
CACHE="${CACHE:-}"
SEED="${SEED:-0}"
SCALE="${SCALE:-s}"
DATA="${DATA:-coco.yaml}"
PROJECT="${PROJECT:-experiments/results}"
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

case "$SCALE" in
    n|s|m|l|x) ;;
    *) echo "error: SCALE must be one of n s m l x, got '$SCALE'" >&2; exit 2 ;;
esac

case "$gpu" in
    ''|*[!0-9]*) echo "error: GPU must be a single index, got '$gpu'" >&2; usage; exit 2 ;;
esac

if ! py -c "import cv2, torch, ultralytics" 2>/dev/null; then
    echo "error: the ultralytics environment is not importable." >&2
    echo "       See TRAINING_GUIDE.md section 0:" >&2
    echo "         uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e '.[extra]'" >&2
    exit 1
fi

# A scaled name (yolo26s-seg-carafe.yaml) is resolved against the unscaled file
# (yolo26-seg-carafe.yaml), which is how one config serves every scale.
unified="${config/yolo26${SCALE}-/yolo26-}"
if [ ! -f "$config" ] && [ ! -f "$unified" ]; then
    echo "error: missing config $config (nor $unified)" >&2
    exit 1
fi
if [ ! -f "$ckpt" ]; then
    echo "error: missing checkpoint $ckpt" >&2
    echo "       Rebuild it: bash experiments/scripts/setup_checkpoints.sh" >&2
    exit 1
fi

# ------------------------------------------------------------------- launch --
if [ "$PROBE" = "1" ]; then name="${stage}_${SCALE}_probe"; else name="${stage}_${SCALE}_1gpu"; fi
# Ultralytics nests a RELATIVE project under <runs_dir>/<task>/, which would put
# results in runs/segment/experiments/results/... and hide them from the resume
# checks. Absolute keeps save_dir exactly at $PROJECT/$name.
mkdir -p "$PROJECT/logs"
PROJECT=$(py -c "import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())" "$PROJECT")
out="$PROJECT/$name"
log="$PROJECT/logs/${name}_$(date +%Y%m%d_%H%M%S).log"

# Decide what to do with this stage from its last.pt alone. Ultralytics stamps
# epoch=-1 into the final, optimizer-stripped checkpoint, so that - not the mere
# existence of best.pt - is what "finished" means: best.pt appears as soon as
# epoch 1 improves fitness, so a run interrupted at epoch 50 has both files.
# Resuming is always the default; only FORCE=1 starts a stage over.
stage_state() {  # -> "done" | "resume" | "fresh"
    local last="$out/weights/last.pt"
    if [ "$FORCE" = "1" ] || [ "$PROBE" = "1" ] || [ ! -f "$last" ]; then
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

state=$(stage_state)
if [ "$state" = "done" ]; then
    echo "$stage already finished all $EPOCHS epochs: $out"
    echo "Pass FORCE=1 to retrain it from scratch."
    exit 0
fi

declare -a args
if [ "$PROBE" = "1" ]; then
    rm -rf "$out"
    # Three epochs with val=False. `fraction` only shrinks the TRAIN split - validation
    # always runs on the full val set - so timing a single epoch with validation on
    # would scale that fixed cost by 1/fraction and wildly overestimate. Differencing
    # epoch 2 against epoch 1 removes both one-time startup and validation, leaving
    # the pure per-epoch training cost; validation is measured once, separately, and
    # added back a single time per epoch. Three epochs, not two, because Ultralytics
    # validates on the FINAL epoch whatever val= says, so the last row is unusable.
    args=(segment train model="$config" pretrained="$ckpt"
        data="$DATA" epochs=3 fraction="$PROBE_FRACTION" batch="$BATCH" imgsz="$IMGSZ"
        device=0 workers="$WORKERS" amp=True seed="$SEED" val=False plots=False
        project="$PROJECT" name="$name" exist_ok=True)
    [ -n "$CACHE" ] && args+=(cache="$CACHE")
    mode="timing probe on $PROBE_FRACTION of $DATA (train timed, val measured separately)"
elif [ "$state" = "resume" ]; then
    args=(segment train resume model="$out/weights/last.pt")
    mode="resuming from $out/weights/last.pt"
else
    args=(segment train model="$config" pretrained="$ckpt"
        data="$DATA" epochs="$EPOCHS" batch="$BATCH" imgsz="$IMGSZ"
        device=0 workers="$WORKERS" amp=True seed="$SEED"
        project="$PROJECT" name="$name" exist_ok=True)
    [ -n "$CACHE" ] && args+=(cache="$CACHE")
    mode="training from $ckpt"
fi

echo "=============================================================="
echo " stage   $stage   ($config)"
echo " gpu     $gpu (pinned via CUDA_VISIBLE_DEVICES; the stage sees it as device 0)"
echo " mode    $mode"
echo " batch   $BATCH   epochs $EPOCHS   imgsz $IMGSZ   workers $WORKERS   seed $SEED${CACHE:+   cache $CACHE}"
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
        # Time one validation pass on the full val split, the cost every real epoch pays.
        vlog="$PROJECT/logs/${name}_val_$(date +%Y%m%d_%H%M%S).log"
        echo
        echo "Measuring one validation pass on the full val split ..."
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync yolo segment val \
            model="$out/weights/last.pt" data="$DATA" batch="$BATCH" imgsz="$IMGSZ" \
            device=0 workers="$WORKERS" plots=False > "$vlog" 2>&1
        echo
        py - "$out/results.csv" "$vlog" "$PROBE_FRACTION" "$EPOCHS" "$stage" <<'PYEOF'
import csv, re, sys

csv_path, val_log, frac, epochs, stage = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]

# Per-epoch training cost: epoch 2 minus epoch 1 drops one-time startup, and val=False
# kept validation out of both. Scaling by 1/fraction gives a full-data training epoch.
try:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    key = next(k for k in rows[0] if k.strip() == "time")
    times = [float(r[key].strip()) for r in rows]
except Exception as e:
    print(f"{stage}: could not read {csv_path} ({e})")
    raise SystemExit(0)
# Rows are cumulative; epochs 1 and 2 are both non-final so neither validated.
train_probe = times[1] - times[0] if len(times) > 2 else times[-1]
train_full = train_probe / frac

# Validation cost from the per-image speed Ultralytics reports, which excludes the
# startup and dataset scan a standalone val run pays but a training epoch does not.
val_s = 0.0
try:
    text = open(val_log, encoding="utf-8", errors="replace").read()
    ms = sum(float(x) for x in re.findall(r"([\d.]+)ms", re.search(r"Speed:.*per image", text).group(0)))
    images = int(re.search(r"^\s*all\s+(\d+)\s+\d+", text, re.M).group(1))
    val_s = ms * images / 1000
except Exception:
    pass

epoch_s = train_full + val_s
total_d = epoch_s * epochs / 86400
print(f"{stage} measured on one GPU:")
print(f"  train  {train_probe:7.1f}s on {frac:g} of the data -> {train_full / 60:6.1f} min/epoch full data")
print(f"  val    {val_s:7.1f}s per epoch on the full val split" + ("" if val_s else "  (not measured - see log)"))
print(f"  epoch  {epoch_s / 60:7.1f} min")
print(f"  {epochs} epochs: {total_d:.2f} days ({epoch_s * epochs / 3600:.1f} h)")
PYEOF
    fi
else
    echo "$stage FAILED (exit $status). Log: $log" >&2
    echo "Re-run the same command to resume from the last checkpoint." >&2
fi
exit "$status"
