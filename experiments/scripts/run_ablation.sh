#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Run the A0-A4 COCO-Seg ablation with ONE STAGE PER GPU, concurrently.
#
# Each stage trains on a single dedicated GPU rather than across all three with
# DDP. Stages are dealt round-robin onto the available GPUs and each GPU works
# through its own queue in series, so five stages on three GPUs run as two waves:
#
#   GPU 1: A0 -> A3
#   GPU 2: A1 -> A4
#   GPU 3: A2
#
# Versus sequential DDP this trades per-stage speed for total throughput: each
# stage is slower (one GPU instead of three) but the sweep finishes in about two
# stage-times instead of five. Use the DDP driver on the rtx6000 branch if you
# need one particular stage finished as early as possible.
#
#   bash experiments/scripts/run_ablation.sh              # all five stages
#   bash experiments/scripts/run_ablation.sh A3 A4        # only these
#   DRY_RUN=1 bash experiments/scripts/run_ablation.sh    # print the plan, run nothing
#   PROBE=1 bash experiments/scripts/run_ablation.sh      # time each stage, train nothing
#   BATCH=64 bash experiments/scripts/run_ablation.sh     # override a setting
#
# Stages already holding a weights/best.pt are skipped so the sweep can be
# re-entered after an interruption; FORCE=1 reruns them from scratch. A stage
# with a weights/last.pt but no best.pt is resumed from that checkpoint.
#
# Settings (override by exporting or prefixing the command):
#   GPUS=1,2,3     GPUs to spread stages over, one stage per GPU at a time
#   BATCH=32       PER-STAGE batch; each stage owns one GPU, so this is per GPU
#   EPOCHS=100     epochs per stage
#   IMGSZ=640      image size
#   WORKERS=6      dataloader workers per stage; total = WORKERS x concurrent stages
#   SEED=0         fixed so stages stay comparable
#   DATA=coco.yaml dataset; use coco8-seg.yaml for a fast end-to-end check
#   PROJECT=experiments/results
#   PROBE=0        set to 1 to time stages instead of training them
#   PROBE_FRACTION=0.01   data fraction used by PROBE=1

set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-python}"
GPUS="${GPUS:-1,2,3}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-6}"
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
    echo "         uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e '.[extra]'" >&2
    exit 1
fi

IFS=, read -r -a gpu_list <<< "$GPUS"
ngpu=${#gpu_list[@]}
[ "$ngpu" -eq 0 ] && { echo "error: GPUS is empty" >&2; exit 2; }

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

# ------------------------------------------------------------------ planning --
# Deal stages round-robin onto GPUs; each GPU runs its own queue in series.
declare -a queue
for i in "${!gpu_list[@]}"; do queue[i]=""; done
for i in "${!selected[@]}"; do
    slot=$(( i % ngpu ))
    queue[slot]="${queue[slot]} ${selected[i]}"
done

if [ "$PROBE" = "1" ]; then
    echo "TIMING PROBE (parallel): 1 epoch on ${PROBE_FRACTION} of $DATA per stage, extrapolated to $EPOCHS epochs"
else
    echo "Parallel ablation: ${want[*]}"
fi
echo "  one stage per GPU, batch=$BATCH per stage  epochs=$EPOCHS  imgsz=$IMGSZ"
echo "  workers=$WORKERS per stage (up to $((WORKERS * ngpu)) loader processes)  seed=$SEED  data=$DATA"
echo "  project=$PROJECT"
echo
waves=0
for i in "${!gpu_list[@]}"; do
    n=$(wc -w <<< "${queue[i]}")
    [ "$n" -gt "$waves" ] && waves=$n
    printf '  GPU %-3s ' "${gpu_list[i]}"
    for s in ${queue[i]}; do printf '%s ' "${s%%:*}"; done
    echo
done
echo "  -> $waves wave(s)"
echo

# ------------------------------------------------------------------- driver --
mkdir -p "$PROJECT/logs"
status_dir=$(mktemp -d)
trap 'rm -rf "$status_dir"' EXIT

parse_probe() {  # $1=results.csv  $2=stage -> "<stage>|<probe>|<per-epoch>|<days>"
    "$PY" - "$1" "$PROBE_FRACTION" "$EPOCHS" "$2" <<'PYEOF'
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
print(f"{stage}|{probe_s:.0f}s|{epoch_s / 60:.1f} min|{epoch_s * epochs / 3600 / 24:.2f}")
PYEOF
}

run_stage() {  # $1=gpu  $2=stage  $3=config  $4=ckpt
    local gpu="$1" stage="$2" config="$3" ckpt="$4"
    local name out log started
    if [ "$PROBE" = "1" ]; then name="${stage}_probe"; else name="${stage}_1gpu"; fi
    out="$PROJECT/$name"
    log="$PROJECT/logs/${name}_$(date +%Y%m%d_%H%M%S).log"

    local -a args
    if [ "$PROBE" = "1" ]; then
        rm -rf "$out"
        args=(segment train model="$config" pretrained="$ckpt"
            data="$DATA" epochs=1 fraction="$PROBE_FRACTION" batch="$BATCH" imgsz="$IMGSZ"
            device=0 workers="$WORKERS" amp=True seed="$SEED"
            project="$PROJECT" name="$name")
    elif [ -f "$out/weights/best.pt" ] && [ "$FORCE" != "1" ]; then
        echo "[GPU $gpu] $stage: already complete, skipping. FORCE=1 to rerun."
        return 0
    elif [ -f "$out/weights/last.pt" ] && [ "$FORCE" != "1" ]; then
        echo "[GPU $gpu] $stage: resuming from $out/weights/last.pt"
        args=(segment train resume model="$out/weights/last.pt")
    else
        args=(segment train model="$config" pretrained="$ckpt"
            data="$DATA" epochs="$EPOCHS" batch="$BATCH" imgsz="$IMGSZ"
            device=0 workers="$WORKERS" amp=True seed="$SEED"
            project="$PROJECT" name="$name")
    fi

    # CUDA_VISIBLE_DEVICES pins the process to one physical GPU, which becomes
    # its device 0 - a stage cannot touch a GPU another stage is training on.
    if [ "$DRY_RUN" = "1" ]; then
        echo "[GPU $gpu] CUDA_VISIBLE_DEVICES=$gpu yolo ${args[*]}"
        return 0
    fi

    echo "[GPU $gpu] $stage: starting -> $log"
    started=$SECONDS
    if CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync yolo "${args[@]}" > "$log" 2>&1; then
        echo "[GPU $gpu] $stage: done in $(((SECONDS - started) / 60)) min"
        if [ "$PROBE" = "1" ]; then
            parse_probe "$out/results.csv" "$stage" > "$status_dir/$stage.probe"
        fi
    else
        echo "[GPU $gpu] $stage: FAILED after $(((SECONDS - started) / 60)) min, see $log" >&2
        echo "$stage" > "$status_dir/$stage.failed"
    fi
}

for i in "${!gpu_list[@]}"; do
    (
        for s in ${queue[i]}; do
            IFS=: read -r stage config ckpt <<< "$s"
            run_stage "${gpu_list[i]}" "$stage" "$config" "$ckpt"
        done
    ) &
done
wait

# ------------------------------------------------------------------ summary --
echo
shopt -s nullglob
probe_files=("$status_dir"/*.probe)
failed_files=("$status_dir"/*.failed)
shopt -u nullglob

if [ "$PROBE" = "1" ] && [ ${#probe_files[@]} -gt 0 ]; then
    echo "Measured timings - $DATA, batch=$BATCH on 1 GPU per stage, imgsz=$IMGSZ, ${EPOCHS} epochs"
    echo
    printf '  %-6s %10s %14s %14s\n' stage probe "per epoch" "${EPOCHS} epochs"
    printf '  %-6s %10s %14s %14s\n' ------ ---------- -------------- --------------
    cat "${probe_files[@]}" | sort | awk -F'|' -v waves="$waves" '
        { printf "  %-6s %10s %14s %12s d\n", $1, $2, $3, $4; if ($4 + 0 > slow) slow = $4 + 0 }
        END { printf "\n  slowest stage %.2f d x %d wave(s) = about %.2f d of wall time\n", slow, waves, slow * waves }
    '
    echo "  Wall time is set by the busiest GPU, not the sum of all stages."
    echo "  Extrapolated from ${PROBE_FRACTION} of the data, measured under the same"
    echo "  parallel I/O contention the real sweep will see."
fi

if [ ${#failed_files[@]} -gt 0 ]; then
    names=$(cat "${failed_files[@]}" | tr '\n' ' ')
    echo "Ablation finished with failures: $names" >&2
    exit 1
fi
[ "$PROBE" = "1" ] || echo "Parallel ablation complete. Results under $PROJECT/"
