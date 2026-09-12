#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Run the A0-A4 COCO-Seg ablation in series on 3x RTX 3090 24GB (devices 0,1,2).
#
#   bash experiments/scripts/run_ablation.sh              # all five stages
#   bash experiments/scripts/run_ablation.sh A3 A4        # only these
#   DRY_RUN=1 bash experiments/scripts/run_ablation.sh    # print commands, run nothing
#   BATCH=48 bash experiments/scripts/run_ablation.sh     # override a setting
#
# Stages already holding a weights/best.pt are skipped so the sweep can be
# re-entered after an interruption; FORCE=1 reruns them from scratch. A stage
# with a weights/last.pt but no best.pt is resumed from that checkpoint.
#
# Settings (override by exporting or prefixing the command):
#   DEVICE=0,1,2   GPUs to train on (TRAINING_GUIDE.md §6.0)
#   BATCH=24       global batch, 8 images/GPU; must divide by the GPU count (§6.1)
#   EPOCHS=100     epochs per stage
#   IMGSZ=640      image size
#   WORKERS=8      dataloader workers PER GPU
#   SEED=0         fixed so stages stay comparable
#   DATA=coco.yaml dataset; use coco8-seg.yaml for a fast end-to-end check
#   PROJECT=experiments/results

set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-python}"
DEVICE="${DEVICE:-0,1,2}"
BATCH="${BATCH:-24}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-0}"
DATA="${DATA:-coco.yaml}"
PROJECT="${PROJECT:-experiments/results}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

# stage : architecture yaml : warm-start checkpoint
STAGES=(
    "A0:ultralytics/cfg/models/26/yolo26-seg.yaml:checkpoints/yolo26s-seg.pt"
    "A1:experiments/configs/yolo26s-seg-carafe.yaml:checkpoints/yolo26s-seg-carafe_pretrained.pt"
    "A2:experiments/configs/yolo26s-seg-aspp.yaml:checkpoints/yolo26s-seg-aspp_pretrained.pt"
    "A3:experiments/configs/yolo26s-seg-carafe-aspp.yaml:checkpoints/yolo26s-seg-carafe-aspp_pretrained.pt"
    "A4:experiments/configs/yolo26s-seg-carafe-aspp-deeplabv3plus.yaml:checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt"
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

if ! "$PY" -c "import cv2, torch, ultralytics" 2>/dev/null; then
    echo "error: the ultralytics environment is not importable with '$PY'." >&2
    echo "       See TRAINING_GUIDE.md section 0:" >&2
    echo "         uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e \".[extra]\"" >&2
    exit 1
fi

ngpu=$(awk -F, '{print NF}' <<< "$DEVICE")
if [ $((BATCH % ngpu)) -ne 0 ]; then
    echo "error: BATCH=$BATCH is not divisible by the $ngpu GPUs in DEVICE=$DEVICE." >&2
    echo "       DDP splits the global batch evenly across devices." >&2
    exit 2
fi

missing=0
for s in "${selected[@]}"; do
    IFS=: read -r stage config ckpt <<< "$s"
    [ -f "$config" ] || { echo "error: missing config $config" >&2; missing=1; }
    [ -f "$ckpt" ] || { echo "error: missing checkpoint $ckpt" >&2; missing=1; }
done
if [ "$missing" -ne 0 ]; then
    echo "       Rebuild the checkpoints: bash experiments/scripts/setup_checkpoints.sh" >&2
    exit 1
fi

echo "Ablation: ${want[*]}"
echo "  device=$DEVICE (${ngpu} GPUs, $((BATCH / ngpu)) images/GPU)  batch=$BATCH  epochs=$EPOCHS"
echo "  imgsz=$IMGSZ  workers=$WORKERS (per GPU)  seed=$SEED  data=$DATA"
echo "  project=$PROJECT"
echo

# ------------------------------------------------------------------- driver --
mkdir -p "$PROJECT/logs"
failed=()

for s in "${selected[@]}"; do
    IFS=: read -r stage config ckpt <<< "$s"
    name="${stage}_3gpu_ddp"
    out="$PROJECT/$name"
    log="$PROJECT/logs/${name}_$(date +%Y%m%d_%H%M%S).log"

    if [ -f "$out/weights/best.pt" ] && [ "$FORCE" != "1" ]; then
        echo "==> $stage: already complete ($out/weights/best.pt), skipping. FORCE=1 to rerun."
        continue
    fi

    args=(segment train
        data="$DATA" epochs="$EPOCHS" batch="$BATCH" imgsz="$IMGSZ"
        device="$DEVICE" workers="$WORKERS" amp=True seed="$SEED"
        project="$PROJECT" name="$name")

    if [ -f "$out/weights/last.pt" ] && [ "$FORCE" != "1" ]; then
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
if [ ${#failed[@]} -ne 0 ]; then
    echo "Ablation finished with failures: ${failed[*]}" >&2
    exit 1
fi
echo "Ablation complete. Results under $PROJECT/"
