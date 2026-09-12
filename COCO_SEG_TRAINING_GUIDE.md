# COCO-Seg Dataset & Training Guide

This guide covers how to download and verify the **COCO 2017 Instance Segmentation (COCO-Seg)** dataset, and how to train the custom **YOLO26s-Seg** models (Stages A0 through A4).

---

## 1. COCO-Seg Dataset Overview

In Ultralytics, the standard dataset configuration [`ultralytics/cfg/datasets/coco.yaml`](file:///E:/project4/yolo_carafe_aspp/ultralytics/cfg/datasets/coco.yaml) is configured to automatically download and process instance segmentation polygon annotations:

- **Image Archives**:
  - `train2017.zip` (~19 GB, 118,287 images)
  - `val2017.zip` (~1 GB, 5,000 images)
- **Segmentation Label Archive**:
  - `coco2017labels-segments.zip` (~30 MB, polygon coordinate text files for each image)
- **Total Uncompressed Disk Space**: **~25 GB**

---

## 2. Dataset Storage Location

Ultralytics uses the global `datasets_dir` configuration setting to locate and download datasets.

### Check Current Location
```bash
uv run --no-sync yolo settings
```
*Current configuration on this machine:* `C:\Users\kchpr\Team2\datasets` (Drive `C:` has **>980 GB free**).

### (Optional) Redirect to Another Drive/Folder
To change where datasets are downloaded (e.g. to drive `E:`):
```bash
uv run --no-sync yolo settings datasets_dir="E:/datasets"
```

---

## 3. How to Download COCO-Seg

Choose the method best suited for your workflow:

### Option A: Auto-Download via Validation (Recommended)
Running validation triggers Ultralytics to check for existing files; if missing, it downloads, unzips, and indexes the full COCO-Seg dataset automatically:
```bash
uv run --no-sync yolo segment val model=checkpoints/yolo26s-seg.pt data=coco.yaml imgsz=640 device=0
```

### Option B: Programmatic Download via Python
Download and extract the dataset without running inference:
```bash
uv run --no-sync python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('coco.yaml', autodownload=True)"
```

### Option C: Instant Testing Subsets (Zero Wait)
If you want to debug or test training immediately without waiting for a 25 GB download, use the pre-packaged mini subsets already available in `datasets_dir`:
- `data=coco8-seg.yaml` (8 images, ~1 MB)
- `data=coco128-seg.yaml` (128 images, ~7 MB)

---

## 4. Dataset Verification

Once downloaded, verify that the images and polygon labels are valid using [`experiments/scripts/check_coco.py`](file:///E:/project4/yolo_carafe_aspp/experiments/scripts/check_coco.py):

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

Five model stages are ready in this repository:

| Stage | Model Description | Architecture Config YAML | Pretrained Weights Checkpoint |
| :--- | :--- | :--- | :--- |
| **A0** | Baseline YOLO26s-Seg | `ultralytics/cfg/models/26/yolo26-seg.yaml` | [`checkpoints/yolo26s-seg.pt`](file:///E:/project4/yolo_carafe_aspp/checkpoints/yolo26s-seg.pt) |
| **A1** | CARAFE Upsampling | [`experiments/configs/yolo26s-seg-carafe.yaml`](file:///E:/project4/yolo_carafe_aspp/experiments/configs/yolo26s-seg-carafe.yaml) | [`checkpoints/yolo26s-seg-carafe_pretrained.pt`](file:///E:/project4/yolo_carafe_aspp/checkpoints/yolo26s-seg-carafe_pretrained.pt) |
| **A2** | ASPP Context Block | [`experiments/configs/yolo26s-seg-aspp.yaml`](file:///E:/project4/yolo_carafe_aspp/experiments/configs/yolo26s-seg-aspp.yaml) | [`checkpoints/yolo26s-seg-aspp_pretrained.pt`](file:///E:/project4/yolo_carafe_aspp/checkpoints/yolo26s-seg-aspp_pretrained.pt) |
| **A3** | CARAFE + ASPP | [`experiments/configs/yolo26s-seg-carafe-aspp.yaml`](file:///E:/project4/yolo_carafe_aspp/experiments/configs/yolo26s-seg-carafe-aspp.yaml) | [`checkpoints/yolo26s-seg-carafe-aspp_pretrained.pt`](file:///E:/project4/yolo_carafe_aspp/checkpoints/yolo26s-seg-carafe-aspp_pretrained.pt) |
| **A4** | **Full Architecture** (CARAFE + ASPP + DeepLabV3+ Decoder) | [`experiments/configs/yolo26s-seg-carafe-aspp-deeplabv3plus.yaml`](file:///E:/project4/yolo_carafe_aspp/experiments/configs/yolo26s-seg-carafe-aspp-deeplabv3plus.yaml) | [`checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt`](file:///E:/project4/yolo_carafe_aspp/checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt) |

---

## 6. How to Train the Model

### 6.1 Quick Smoke Test (Local GPU Verification)
For local testing (e.g. Quadro P2200 5GB VRAM), run a 1-epoch test on `coco8-seg.yaml` or a 1% fraction of COCO:
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

### 6.2 Single-GPU Full Training (e.g. 1× RTX 3090, 24GB VRAM)
```bash
uv run --no-sync yolo segment train \
  model=checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt \
  data=coco.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  workers=8 \
  amp=True \
  project=experiments/results \
  name=A4_single_gpu
```

### 6.3 Multi-GPU 3-GPU DDP Training (Plan Target: 3× RTX 3090)
As specified in the experiment plan, use a global batch size divisible by 3 (`batch=24` = 8 images/GPU):
```bash
uv run --no-sync yolo segment train \
  model=experiments/configs/yolo26s-seg-carafe-aspp-deeplabv3plus.yaml \
  pretrained=checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt \
  data=coco.yaml \
  epochs=100 \
  batch=24 \
  imgsz=640 \
  device=0,1,2 \
  workers=4 \
  amp=True \
  project=experiments/results \
  name=A4_3gpu_ddp
```

---

## 7. Python API Training Interface

You can run or automate training inside Python scripts:

```python
from ultralytics import YOLO

# Load model with warm-started pretrained weights
model = YOLO("checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt")

# Train model
results = model.train(
    data="coco.yaml",
    epochs=100,
    batch=24,           # 8 images/GPU on 3 GPUs, or 16 for single 24GB GPU
    imgsz=640,
    device="0,1,2",     # or device=0 for single GPU
    workers=4,
    amp=True,
    project="experiments/results",
    name="A4_training_run",
)
```

---

## 8. Resuming Interrupted Runs

If training is paused or interrupted, resume from the latest checkpoint:
```bash
uv run --no-sync yolo segment train resume model=experiments/results/A4_3gpu_ddp/weights/last.pt
```
