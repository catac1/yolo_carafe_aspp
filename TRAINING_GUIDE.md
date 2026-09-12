# COCO-Seg Dataset & Training Guide

This guide covers how to download and verify the **COCO 2017 Instance Segmentation (COCO-Seg)** dataset, and how to train the custom **YOLO26s-Seg** models (Stages A0 through A4).

**Target environment:** single headless Linux node, Python 3.12, training on **3× NVIDIA RTX 6000** (devices **1, 2, 3** — GPU 0 is occupied and OOMs, see §6.0).
The package depends on `opencv-python-headless`, so no X11/Qt libraries are required, and training runs entirely on PyTorch `.pt` weights — no export toolchains are installed.

### First run on a new machine, in order

Three things must exist before training, and none of them arrive with a `git clone` — `.venv/`, `uv.lock`, and `*.pt` are all gitignored:

```bash
uv venv --python 3.12 && source .venv/bin/activate   # §0  no env  -> ModuleNotFoundError: No module named 'cv2'
uv pip install -e ".[extra]"                          # §0
bash experiments/scripts/setup_checkpoints.sh         # §5.1 no weights -> FileNotFoundError: checkpoints/...pt
uv run --no-sync yolo segment train ... data=coco8-seg.yaml epochs=1   # §6.2 smoke test before the real run
```

The dataset itself downloads on demand (§3), so it needs no separate step.

---

## 0. Environment Setup

Do this first on a new machine. Every later command uses `uv run --no-sync`, and **`--no-sync` deliberately skips dependency installation** — it runs against an environment that already exists. Without this section you get:

```text
ModuleNotFoundError: No module named 'cv2'
```

The project requires **Python 3.12** (`requires-python = ">=3.12,<3.13"`).

```bash
# 1. Create the virtualenv with the pinned interpreter
uv venv --python 3.12
source .venv/bin/activate

# 2. Install the package plus the training extras
#    (albumentations for augmentation, faster-coco-eval for COCO mAP)
uv pip install -e ".[extra]"
```

`.venv/` and `uv.lock` are both gitignored, so this step is per-machine and never arrives with the clone.

### Verify the install

```bash
uv run --no-sync python -c "
import torch, cv2, ultralytics
print('ultralytics', ultralytics.__version__)
print('cv2        ', cv2.__version__)
print('torch      ', torch.__version__, 'cuda', torch.version.cuda)
print('GPUs       ', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f'   {i} {torch.cuda.get_device_name(i)}  {free / 1e9:.1f}/{total / 1e9:.1f} GB free')
"
```

Expect to see the RTX 6000s listed. Devices 1, 2 and 3 are the training GPUs; GPU 0 is excluded (§6.0).

### CUDA build of PyTorch

The default PyPI `torch` wheel may not carry kernels for your GPU. If `torch.cuda.is_available()` is `False`, or training dies with `no kernel image is available for execution on the device`, install a matching CUDA build — for example for Blackwell (`sm_120`, RTX PRO 6000), which needs CUDA 12.8 or newer:

```bash
uv pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Check your architecture against the installed build with:

```bash
uv run --no-sync python -c "import torch; print(torch.cuda.get_arch_list())"
```

### A note on OpenCV

The dependency is `opencv-python-headless`, which is correct for a headless training server and pulls no X11/Qt libraries. Do not install plain `opencv-python` alongside it — the two provide the same `cv2` module and installing both produces an unpredictable mix. If `cv2` is missing after setup, reinstall the package rather than adding OpenCV by hand:

```bash
uv pip install --force-reinstall opencv-python-headless
```

---

## 1. COCO-Seg Dataset Overview

In Ultralytics, the standard dataset configuration [`ultralytics/cfg/datasets/coco.yaml`](ultralytics/cfg/datasets/coco.yaml) is configured to automatically download and process instance segmentation polygon annotations:

- **Image Archives**:
    - `train2017.zip` (~19 GB, 118,287 images)
    - `val2017.zip` (~1 GB, 5,000 images)
- **Segmentation Label Archive**:
    - `coco2017labels-segments.zip` (~30 MB, polygon coordinate text files for each image)
- **Disk space required**: **~45 GB**. The extracted dataset is ~25 GB, but Ultralytics downloads with `delete=False`, so the ~20 GB of `.zip` archives are kept alongside it. They can be removed by hand once extraction succeeds (see §3.1).

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
uv run --no-sync yolo segment val model=checkpoints/yolo26s-seg.pt data=coco.yaml imgsz=640 device=1
```

### Option B: Programmatic Download via Python

```bash
uv run --no-sync python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('coco.yaml', autodownload=True)"
```

### Option C: Instant Testing Subsets (Zero Wait)

To debug training without waiting for a 25 GB download:

- `data=coco8-seg.yaml` (8 images, ~1 MB)
- `data=coco128-seg.yaml` (128 images, ~7 MB)


### 3.1 If the Dataset Is Missing

```text
ls: cannot access '/workspace/coco_dataset/coco/val2017.txt': No such file or directory
```

`val2017.txt` ships inside the labels archive, so this means the download has not run (or ran somewhere else). Check where Ultralytics is actually pointing first — a settings file written before `datasets_dir` was changed keeps its old value, because the schema migration preserves existing settings:

```bash
uv run --no-sync yolo settings | grep datasets_dir
```

If it is not `/workspace/coco_dataset`, set it explicitly:

```bash
uv run --no-sync yolo settings datasets_dir=/workspace/coco_dataset
```

Then confirm the directory is writeable by the training user and has ~45 GB free:

```bash
mkdir -p /workspace/coco_dataset && df -h /workspace/coco_dataset
```

Now download (~20 GB over the network, expect 20-60 minutes):

```bash
uv run --no-sync python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('coco.yaml', autodownload=True)"
```

The expected result — note the labels archive is what creates the `.txt` index files:

```bash
ls /workspace/coco_dataset/coco/
# images/  labels/  train2017.txt  val2017.txt  LICENSE  README.txt
```

Once `experiments/scripts/check_coco.py` passes (§4), reclaim ~20 GB by deleting the archives that Ultralytics leaves behind:

```bash
rm -f /workspace/coco_dataset/coco/images/*.zip /workspace/coco_dataset/*.zip
```

To train immediately without waiting for the download, use `data=coco8-seg.yaml` — it fetches in seconds and exercises the identical code path.

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

### 5.1 Rebuilding the Checkpoints on a New Machine

`.gitignore` excludes `*.pt`, so **none of the checkpoints are tracked in git** — a fresh clone has no `checkpoints/` directory and training fails with:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt'
```

Rebuild the full A0–A4 set on the training machine before the first run:

```bash
bash experiments/scripts/setup_checkpoints.sh
```

This downloads the `yolo26s-seg.pt` baseline from the Ultralytics release assets, then warm-starts A1–A4 from it with `transfer_weights.py` and rewrites the reports in `experiments/notes/`. It takes about a minute and needs no GPU.

Verify the result against the committed transfer reports — the numbers are deterministic, so any deviation means a config or baseline mismatch:

| Stage | Transferred parameters | Matched tensors | Newly initialized tensors |
| :---- | :--------------------- | :-------------- | :------------------------ |
| A1    | 11,546,542 (98.94%)    | 844             | 16                        |
| A2    | 11,546,542 (84.06%)    | 844             | 36                        |
| A3    | 11,546,542 (83.06%)    | 844             | 52                        |
| A4    | 10,775,127 (79.43%)    | 798             | 86                        |

The A2–A4 configs insert an ASPP block at index 11, so every later layer shifts by one — those three transfers need `--layer-shift-after 10 --layer-shift 1`, which the script passes for you. Running `transfer_weights.py` on them without it silently matches only 240 tensors instead of 844 and warm-starts roughly 40% of the weights instead of 80%+, so always check the table above rather than assuming success.

Alternatively, copy an existing `checkpoints/` directory across (~131 MB) with `rsync` or `scp` — the results are identical, since the baseline download is byte-for-byte the same file.

---

## 6. How to Train the Model

All commands below assume the plan target: **3× RTX 6000, single node, Linux**, training on **GPUs 1, 2 and 3** — GPU 0 is reserved and will OOM (see §6.0).

### 6.0 Selecting GPUs (skip GPU 0)

GPU 0 on this host is not available for training — it is already occupied and a run placed on it dies with CUDA OOM. Every command in this guide therefore targets **devices 1, 2, 3**.

Confirm what is free before launching; the `memory.used` column should be near zero on 1, 2 and 3:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv   # what is holding GPU 0
```

There are two ways to select them, and they differ in how the GPUs are numbered:

```bash
# A. Ultralytics device argument - indices are PHYSICAL, as nvidia-smi reports them
yolo segment train ... device=1,2,3

# B. CUDA_VISIBLE_DEVICES - remaps them, so the visible set is renumbered from 0
CUDA_VISIBLE_DEVICES=1,2,3 yolo segment train ... device=0,1,2
```

Both run on the same hardware. Do not combine them carelessly: with `CUDA_VISIBLE_DEVICES=1,2,3` set, `device=1,2,3` refers to the *second, third and fourth visible* GPUs — only one of which exists — and the run fails with an invalid device ordinal. This guide uses form A throughout and sets no `CUDA_VISIBLE_DEVICES`.

Form B is the safer choice if anything else on the box might touch GPU 0, because the training process then cannot address it at all.

### 6.1 Pick the Batch Size for Your Card

DDP splits the `batch` argument across GPUs, so **the global batch must be divisible by 3**. Ultralytics reports the per-GPU split at startup — confirm it matches the table before letting a 100-epoch run proceed.

| Card                       | VRAM  | Per-GPU batch | `batch` (3 GPUs) |
| :------------------------- | :---- | :------------ | :--------------- |
| RTX 6000 (Turing)          | 24 GB | 16            | `48`             |
| RTX 6000 Ada               | 48 GB | 32            | `96`             |
| RTX PRO 6000 Blackwell     | 96 GB | 64            | `192`            |

These are starting points for YOLO26s-Seg at `imgsz=640` with `amp=True`. Segmentation masks make memory scale worse than detection, and the A4 DeepLabV3+ decoder is the heaviest of the five stages — if A4 hits CUDA OOM, drop one row and keep every stage on the same batch size so the ablation stays comparable.

Confirm what you actually have before choosing:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
```

### 6.2 Quick Smoke Test

Verify the pipeline end to end on 8 images before committing 3 GPUs to a long run. Use a single GPU here — DDP startup noise only obscures real errors:

```bash
uv run --no-sync yolo segment train \
  model=checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt \
  data=coco8-seg.yaml \
  epochs=1 \
  batch=2 \
  imgsz=640 \
  device=1 \
  amp=True \
  project=experiments/results \
  name=A4_smoke_test
```

Then smoke-test DDP itself across all three training GPUs, still on 8 images:

```bash
uv run --no-sync yolo segment train \
  model=checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt \
  data=coco8-seg.yaml \
  epochs=1 \
  batch=3 \
  imgsz=640 \
  device=1,2,3 \
  amp=True \
  project=experiments/results \
  name=A4_ddp_smoke
```

### 6.3 Full Training (3× RTX 6000 DDP)

`batch=96` below assumes 48 GB cards — substitute your row from §6.1:

```bash
uv run --no-sync yolo segment train \
  model=experiments/configs/yolo26s-seg-carafe-aspp-deeplabv3plus.yaml \
  pretrained=checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt \
  data=coco.yaml \
  epochs=100 \
  batch=96 \
  imgsz=640 \
  device=1,2,3 \
  workers=8 \
  amp=True \
  seed=0 \
  project=experiments/results \
  name=A4_3gpu_ddp
```

Notes specific to the 3-GPU setup:

- **`workers` is per-GPU.** With `workers=8` on 3 GPUs you get 24 dataloader processes; keep `workers × 3` at or below the host's physical core count (`nproc`). Too many workers starves the GPUs rather than feeding them.
- **Ultralytics spawns DDP itself.** Run the plain `yolo` command above — do not wrap it in `torchrun` or `python -m torch.distributed.run`, which produces nested process groups.
- **Scale the learning rate with the global batch.** `lr0` defaults are tuned around batch 64; tripling the batch usually wants a proportionally higher `lr0` plus warmup. Try `lr0=0.01 warmup_epochs=5` if loss plateaus early.
- **Rank 0 is the first device in the list, not GPU 0.** With `device=1,2,3` the rank-0 process runs on physical GPU 1. Checkpoints, plots, and `results.csv` are written once under `experiments/results/<name>/`, not three times.
- **A killed run can leak GPU memory.** If a DDP run dies uncleanly, clear orphans before relaunching.

### 6.4 Running the Full A0-A4 Ablation

The five stages differ only in `model` and `pretrained`; every other argument is held fixed so the comparison stays clean. They run in series — three GPUs serve one stage at a time, not one stage per GPU.

```bash
bash experiments/scripts/run_ablation.sh
```

The script validates the environment, configs, and checkpoints before starting, so a typo fails in seconds rather than three stages in. It refuses a `BATCH` that is not divisible by the GPU count, writes a timestamped log per stage under `experiments/results/logs/`, and reports which stages failed instead of stopping the sweep at the first one.

Common variations:

```bash
bash experiments/scripts/run_ablation.sh A3 A4        # only selected stages
DRY_RUN=1 bash experiments/scripts/run_ablation.sh    # print the commands, run nothing
BATCH=192 bash experiments/scripts/run_ablation.sh    # 96 GB cards (§6.1)
DEVICE=1,2 BATCH=64 bash experiments/scripts/run_ablation.sh   # two GPUs
DATA=coco8-seg.yaml EPOCHS=1 BATCH=3 bash experiments/scripts/run_ablation.sh   # end-to-end check
PROBE=1 bash experiments/scripts/run_ablation.sh      # measure time per stage, train nothing
```

Settings, overridable by prefixing the command: `DEVICE=1,2,3`, `BATCH=96`, `EPOCHS=100`, `IMGSZ=640`, `WORKERS=8` (per GPU), `SEED=0`, `DATA=coco.yaml`, `PROJECT=experiments/results`, `PROBE=0`, `PROBE_FRACTION=0.01`.

**Measure the cost before committing to it.** `PROBE=1` trains one epoch on 1% of the data per stage and extrapolates a per-epoch and total run time, so you know what the sweep costs before starting it:

```bash
PROBE=1 bash experiments/scripts/run_ablation.sh
```

It prints a measured table:

```text
  stage       probe      per epoch     100 epochs
  ------ ---------- -------------- --------------
  A0           ...            ...            ...
  ...
  TOTAL                                      ... d
```

Probe runs write to `<stage>_probe/` and never touch the real run directories, so they are safe to repeat. `PROBE_FRACTION=0.02` samples more data for a steadier number. The extrapolation is a **floor** — it scales one epoch's training time linearly and does not model per-epoch validation on the full 5,000-image val set, which the real runs pay every epoch.

Use it to sanity-check the batch size too: if the probe shows a per-epoch time far above expectation, the dataloader is starving the GPUs (§6.5) before you have spent days finding out.

**Re-entrancy.** A sweep of this length will be interrupted. Re-running the script:

- **skips** a stage that already has `weights/best.pt` — it finished
- **resumes** a stage that has `weights/last.pt` but no `best.pt` — it was interrupted
- `FORCE=1` retrains a stage from scratch regardless

So after a crash, reboot, or OOM, just run the same command again. Launch it detached, since the full sweep is five multi-day runs:

```bash
tmux new -s ablation 'bash experiments/scripts/run_ablation.sh'
```

`seed=0` makes the runs reproducible. Keep `batch` identical across all five stages, including A4 — changing it mid-ablation invalidates the comparison.

### 6.5 Monitoring

```bash
watch -n 5 nvidia-smi                                   # utilization and VRAM headroom on 1,2,3
tail -f experiments/results/A4_3gpu_ddp/results.csv     # per-epoch metrics
```

If GPU utilization sits well below ~90%, the dataloader is the bottleneck — raise `workers`, or add `cache=ram` if the host has enough spare RAM (COCO-Seg at 640 px needs roughly 30+ GB).

---

## 7. Python API Training Interface

```python
from ultralytics import YOLO

# Load model with warm-started pretrained weights
model = YOLO("checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt")

# Train model on 3x RTX 6000
results = model.train(
    data="coco.yaml",
    epochs=100,
    batch=96,  # global batch, 32 images/GPU across 3 GPUs; must be divisible by 3
    imgsz=640,
    device="1,2,3",
    workers=8,  # per-GPU; 8 x 3 = 24 dataloader processes
    amp=True,
    seed=0,
    project="experiments/results",
    name="A4_training_run",
)
```

Launch this as a script (`python train.py`), not from an interactive interpreter or notebook — Ultralytics re-executes the entry file in each DDP worker.

---

## 8. Resuming Interrupted Runs

`resume` restores the optimizer state, epoch counter, and the full argument set from the checkpoint, so no other flags are needed — and none are honored:

```bash
uv run --no-sync yolo segment train resume model=experiments/results/A4_3gpu_ddp/weights/last.pt
```

The resumed run reuses the saved `device=1,2,3` and `batch`, so it re-forms the same 3-GPU DDP group. To change the batch size or GPU count you must start a fresh run rather than resume.

A 100-epoch COCO-Seg run takes well over a day, so launch it detached under `tmux` (preferred — you can reattach to a live console) or `nohup`:

```bash
tmux new -s a4 'uv run --no-sync yolo segment train ... 2>&1 | tee train.log'
# detach with Ctrl-B D, reattach with: tmux attach -t a4
```

```bash
nohup uv run --no-sync yolo segment train ... > train.log 2>&1 &
```

If a run dies from OOM partway through, resume from `last.pt` only after confirming no orphaned processes still hold GPU memory:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```
