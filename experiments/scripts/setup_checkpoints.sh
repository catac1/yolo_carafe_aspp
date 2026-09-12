#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Rebuild the A0-A4 checkpoint set. The *.pt files are excluded by .gitignore,
# so a fresh clone has no checkpoints/ directory and training fails with
# FileNotFoundError. Run this once after cloning.
#
#   bash experiments/scripts/setup_checkpoints.sh
#
# Produces, from the auto-downloaded yolo26s-seg.pt baseline:
#   checkpoints/yolo26s-seg.pt                                      (A0 baseline)
#   checkpoints/yolo26s-seg-carafe_pretrained.pt                    (A1)
#   checkpoints/yolo26s-seg-aspp_pretrained.pt                      (A2)
#   checkpoints/yolo26s-seg-carafe-aspp_pretrained.pt               (A3)
#   checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt (A4)

set -euo pipefail
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

# Fail early with an actionable message instead of a traceback from deep inside
# the import chain when the virtualenv was never created or activated.
if ! py -c "import cv2, torch, ultralytics" 2>/dev/null; then
    echo "error: the ultralytics environment is not importable." >&2
    echo "       Create and activate it first (see TRAINING_GUIDE.md section 0):" >&2
    echo "         uv venv --python 3.12 && source .venv/bin/activate" >&2
    echo "         uv pip install -e \".[extra]\"" >&2
    exit 1
fi

mkdir -p checkpoints experiments/notes

# A0: baseline, downloaded from the Ultralytics GitHub release assets
if [ ! -f checkpoints/yolo26s-seg.pt ]; then
    echo "==> Downloading A0 baseline yolo26s-seg.pt"
    py -c "
from pathlib import Path
from ultralytics.utils.downloads import attempt_download_asset
Path('checkpoints').mkdir(exist_ok=True)
attempt_download_asset('checkpoints/yolo26s-seg.pt')
"
else
    echo "==> A0 baseline already present, skipping download"
fi

# A1-A4: warm-start transfers off the A0 baseline. The ASPP stages insert a
# block at index 11, so every later layer shifts by one and needs --layer-shift.
transfer() {
    local stage="$1" config="$2" output="$3"
    shift 3
    echo "==> ${stage}: ${config}"
    py experiments/scripts/transfer_weights.py \
        --config "experiments/configs/${config}.yaml" \
        --checkpoint checkpoints/yolo26s-seg.pt \
        --output "checkpoints/${output}.pt" \
        --report "experiments/notes/${stage}_weight_transfer_report.md" \
        "$@"
}

transfer A1 yolo26s-seg-carafe yolo26s-seg-carafe_pretrained
transfer A2 yolo26s-seg-aspp yolo26s-seg-aspp_pretrained --layer-shift-after 10 --layer-shift 1
transfer A3 yolo26s-seg-carafe-aspp yolo26s-seg-carafe-aspp_pretrained --layer-shift-after 10 --layer-shift 1
transfer A4 yolo26s-seg-carafe-aspp-deeplabv3plus yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained \
    --layer-shift-after 10 --layer-shift 1

echo
echo "==> Done. Checkpoints:"
ls -la checkpoints/
