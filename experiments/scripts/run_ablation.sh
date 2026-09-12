#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Run the A0-A4 COCO-Seg ablation in series on 3x RTX 3090 24GB (devices 0,1,2).
#
#   bash experiments/scripts/run_ablation.sh              # all five stages
#   bash experiments/scripts/run_ablation.sh A3 A4        # only these
#   DRY_RUN=1 bash experiments/scripts/run_ablation.sh    # print commands, run nothing
#   BATCH=48 bash experiments/scripts/run_ablation.sh     # override a setting
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
#   DEVICE=0,1,2   GPUs to train on (TRAINING_GUIDE.md §6.0)
#   BATCH=24       global batch, 8 images/GPU; must divide by the GPU count (§6.1)
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

if [ "$PROBE" = "1" ]; then
    echo "TIMING PROBE: ${want[*]} - 1 epoch on ${PROBE_FRACTION} of $DATA, extrapolated to $EPOCHS epochs"
else
    echo "Ablation: ${want[*]}"
fi
echo "  device=$DEVICE (${ngpu} GPUs, $((BATCH / ngpu)) images/GPU)  batch=$BATCH  epochs=$EPOCHS"
echo "  imgsz=$IMGSZ  workers=$WORKERS (per GPU)  seed=$SEED  data=$DATA"
echo "  project=$PROJECT"
echo

# ------------------------------------------------------------------- driver --
mkdir -p "$PROJECT/logs"
failed=()
probe_rows=()

for s in "${selected[@]}"; do
    IFS=: read -r stage config ckpt <<< "$s"
    if [ "$PROBE" = "1" ]; then
        name="${stage}_probe"
    else
        name="${stage}_3gpu_ddp"
    fi
    out="$PROJECT/$name"
    log="$PROJECT/logs/${name}_$(date +%Y%m%d_%H%M%S).log"

    if [ "$PROBE" = "1" ]; then
        rm -rf "$out"
        args=(segment train
            model="$config" pretrained="$ckpt"
            data="$DATA" epochs=1 fraction="$PROBE_FRACTION" batch="$BATCH" imgsz="$IMGSZ"
            device="$DEVICE" workers="$WORKERS" amp=True seed="$SEED"
            project="$PROJECT" name="$name")
        echo "==> $stage: timing probe"
        if [ "$DRY_RUN" = "1" ]; then
            echo "    yolo ${args[*]}"
            continue
        fi
        if uv run --no-sync yolo "${args[@]}" 2>&1 | tee "$log"; then
            row=$("$PY" - "$out/results.csv" "$PROBE_FRACTION" "$EPOCHS" "$stage" <<'EOF'
import csv, sys
csv_path, frac, epochs, stage = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
try:
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    key = next(k for k in rows[-1] if k.strip() == "time")
    probe_s = float(rows[-1][key].strip())
except Exception as e:
    print(f"{stage}|ERROR|{e}|")
    raise SystemExit(0)
epoch_s = probe_s / frac
total_h = epoch_s * epochs / 3600
print(f"{stage}|{probe_s:.0f}s|{epoch_s / 60:.1f} min|{total_h / 24:.2f} d")
EOF
)
            probe_rows+=("$row")
            echo "    $stage probe: $(cut -d'|' -f3 <<< "$row") per full epoch"
        else
            echo "    $stage probe FAILED, see $log" >&2
            failed+=("$stage")
        fi
        echo
        continue
    fi

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
if [ "$PROBE" = "1" ] && [ ${#probe_rows[@]} -ne 0 ]; then
    echo "Measured timings - $DATA, batch=$BATCH on ${ngpu} GPUs, imgsz=$IMGSZ, ${EPOCHS} epochs"
    echo
    printf '  %-6s %10s %14s %14s
' stage "probe" "per epoch" "${EPOCHS} epochs"
    printf '  %-6s %10s %14s %14s
' ------ ---------- -------------- --------------
    total=0
    for r in "${probe_rows[@]}"; do
        IFS='|' read -r s p e tot <<< "$r"
        printf '  %-6s %10s %14s %14s
' "$s" "$p" "$e" "$tot"
        total=$("$PY" -c "print(f'{$total + float('${tot% d}'):.2f}')")
    done
    printf '  %-6s %10s %14s %13s d
' TOTAL "" "" "$total"
    echo
    echo "Extrapolated from ${PROBE_FRACTION} of the data; real runs carry extra"
    echo "per-epoch validation and checkpoint overhead, so treat these as a floor."
fi

if [ ${#failed[@]} -ne 0 ]; then
    echo "Ablation finished with failures: ${failed[*]}" >&2
    exit 1
fi
[ "$PROBE" = "1" ] || echo "Ablation complete. Results under $PROJECT/"
