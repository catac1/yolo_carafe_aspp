# COCO-Seg Dataset & Training Guide

This guide covers how to download and verify the **COCO 2017 Instance Segmentation (COCO-Seg)** dataset, and how to train the custom **YOLO26s-Seg** models (Stages A0 through A4).

**Target environment:** Linux, NVIDIA RTX 6000 GPUs, headless (no display server). The package depends on `opencv-python-headless`, so no X11/Qt libraries are required.

---

## 1. COCO-Seg Dataset Overview

In Ultralytics, the standard dataset configuration [`ultralytics/cfg/datasets/coco.yaml`](ultralytics/cfg/datasets/coco.yaml) is configured to automatically download and process instance segmentation polygon annotations:

- **Image Archives**:
    - `train2017.zip` (~19 GB, 118,287 images)
    - `val2017.zip` (~1 GB, 5,000 images)
- **Segmentation Label Archive**:
    - `coco2017labels-segments.zip` (~30 MB, polygon coordinate text files for each image)
- **Total Uncompressed Disk Space**: **~25 GB**

---

## 2. Dataset Storage Location & Customization

### Default Download Location

The default `datasets_dir` setting is **`/workspace/coco_dataset`**, so COCO-Seg lands at:

```text
/workspace/coco_dataset/coco/
├── images/
│   ├── train2017/     (118,287 images)
│   └── val2017/       (5,000 images)
├── labels/
│   ├── train2017/     (118,287 polygon .txt files)
│   └── val2017/       (5,000 polygon .txt files)
├── train2017.txt
└── val2017.txt
```

Make sure the directory exists and is writeable by the training user before the first download:

```bash
sudo mkdir -p /workspace/coco_dataset && sudo chown "$USER" /workspace/coco_dataset
```

> **Note:** the default only applies when no Ultralytics settings file exists yet. If the machine has already run Ultralytics, its saved `datasets_dir` is preserved across upgrades — override it explicitly with Method 1 below.

### How to Customize the Dataset Location

#### Method 1: Change Ultralytics Global Setting (Recommended)

- **Via Python:**
    ```python
    from ultralytics import settings

    settings.update({"datasets_dir": "/workspace/coco_dataset"})
    ```
- **Via CLI:**
    ```bash
    uv run --no-sync yolo settings datasets_dir=/workspace/coco_dataset
    ```

#### Method 2: Custom Location in Python Script

Set the location dynamically right before triggering Option B:

```python
from ultralytics import settings
from ultralytics.data.utils import check_det_dataset

settings.update({"datasets_dir": "/mnt/data/datasets"})
check_det_dataset("coco.yaml", autodownload=True)
```

#### Method 3: Custom YAML with Absolute Path

Create a YAML file (e.g. `experiments/configs/coco_custom.yaml`) specifying an absolute path:

```yaml
path: /mnt/data/datasets/coco # Absolute root path
train: train2017.txt
val: val2017.txt
```

Because `path` is absolute, Ultralytics ignores `datasets_dir` and downloads directly into `/mnt/data/datasets/coco`.

---

## 3. How to Download COCO-Seg

### Option A: Auto-Download via Validation (Recommended)

```bash
uv run --no-sync yolo segment val model=checkpoints/yolo26s-seg.pt data=coco.yaml imgsz=640 device=0
```

### Option B: Programmatic Download via Python

```bash
uv run --no-sync python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('coco.yaml', autodownload=True)"
```

### Option C: Instant Testing Subsets (Zero Wait)

To debug training without waiting for a 25 GB download:

- `data=coco8-seg.yaml` (8 images, ~1 MB)
- `data=coco128-seg.yaml` (128 images, ~7 MB)

---

## 4. Dataset Verification

```bash
uv run --no-sync python experiments/scripts/check_coco.py --data coco.yaml --samples 100
```

This checks:

- Image counts (~118,287 train, ~5,000 val)
- Corresponding `.txt` polygon label existence
- Normalized coordinate bounds ($0.0 \le x, y \le 1.0$)
- Class index bounds ($0 \le \text{class} < 80$)

---

## 5. Model Architectures & Checkpoints

| Stage  | Model Description                                          | Architecture Config YAML                                              | Pretrained Weights Checkpoint                                    |
| :----- | :--------------------------------------------------------- | :-------------------------------------------------------------------- | :--------------------------------------------------------------- |
| **A0** | Baseline YOLO26s-Seg                                        | `ultralytics/cfg/models/26/yolo26-seg.yaml`                            | `checkpoints/yolo26s-seg.pt`                                      |
| **A1** | CARAFE Upsampling                                           | `experiments/configs/yolo26s-seg-carafe.yaml`                          | `checkpoints/yolo26s-seg-carafe_pretrained.pt`                    |
| **A2** | ASPP Context Block                                          | `experiments/configs/yolo26s-seg-aspp.yaml`                            | `checkpoints/yolo26s-seg-aspp_pretrained.pt`                      |
| **A3** | CARAFE + ASPP                                               | `experiments/configs/yolo26s-seg-carafe-aspp.yaml`                     | `checkpoints/yolo26s-seg-carafe-aspp_pretrained.pt`               |
| **A4** | **Full Architecture** (CARAFE + ASPP + DeepLabV3+ Decoder)  | `experiments/configs/yolo26s-seg-carafe-aspp-deeplabv3plus.yaml`       | `checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt` |

---

## 6. How to Train the Model

### 6.1 Quick Smoke Test

Verify the pipeline end to end on 8 images before committing a GPU to a full run:

```bash
uv run --no-sync yolo segment train \
  model=checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt \
  data=coco8-seg.yaml \
  epochs=1 \
  batch=2 \
  imgsz=640 \
  device=0 \
  amp=True \
  project=experiments/results \
  name=A4_smoke_test
```

### 6.2 Single-GPU Full Training (1× RTX 6000)

```bash
uv run --no-sync yolo segment train \
  model=checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt \
  data=coco.yaml \
  epochs=100 \
  batch=64 \
  imgsz=640 \
  device=0 \
  workers=16 \
  amp=True \
  project=experiments/results \
  name=A4_single_gpu
```

`batch=64` targets a 48 GB RTX 6000 Ada. On a 96 GB RTX PRO 6000 Blackwell try `batch=128`; if you hit CUDA OOM, halve it. `batch=-1` lets Ultralytics auto-pick a batch size for ~60% VRAM utilization.

### 6.3 Multi-GPU DDP Training (3× RTX 6000)

Keep the global batch size divisible by the GPU count (`batch=96` = 32 images/GPU):

```bash
uv run --no-sync yolo segment train \
  model=experiments/configs/yolo26s-seg-carafe-aspp-deeplabv3plus.yaml \
  pretrained=checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt \
  data=coco.yaml \
  epochs=100 \
  batch=96 \
  imgsz=640 \
  device=0,1,2 \
  workers=8 \
  amp=True \
  project=experiments/results \
  name=A4_3gpu_ddp
```

`workers` is per-GPU; keep `workers × GPUs` at or below the host's physical core count.

---

## 7. Python API Training Interface

```python
from ultralytics import YOLO

# Load model with warm-started pretrained weights
model = YOLO("checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt")

# Train model
results = model.train(
    data="coco.yaml",
    epochs=100,
    batch=96,  # 32 images/GPU on 3 GPUs, or 64 for a single RTX 6000
    imgsz=640,
    device="0,1,2",  # or device=0 for single GPU
    workers=8,
    amp=True,
    project="experiments/results",
    name="A4_training_run",
)
```

---

## 8. Resuming Interrupted Runs

```bash
uv run --no-sync yolo segment train resume model=experiments/results/A4_3gpu_ddp/weights/last.pt
```

For long unattended runs, launch under `nohup` or `tmux` so an SSH disconnect does not kill training:

```bash
nohup uv run --no-sync yolo segment train ... > train.log 2>&1 &
```
