# COCO-Seg Dataset & Training Guide

This guide covers how to download and verify the **COCO 2017 Instance Segmentation (COCO-Seg)** dataset, and how to train the custom **YOLO26s-Seg** models (Stages A0 through A4).

**Target environment:** single headless Linux node, Python 3.12, training on **3× NVIDIA RTX 6000** (devices **1, 2, 3** — GPU 0 is occupied and OOMs, see §6.0).
**This branch runs one stage per GPU, one terminal each, with no DDP** (§6.4).
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

### 6.1 Pick the Batch Size (per stage)

Each stage owns one GPU, so `batch` is simply what fits on a single card — there is no global batch to divide and no divisibility constraint. Ultralytics reports peak VRAM per epoch; check it after the first epoch.

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

Verify the pipeline end to end on 8 images before committing a GPU to a long run:

```bash
DATA=coco8-seg.yaml EPOCHS=1 BATCH=2   bash experiments/scripts/run_stage.sh A4 1
```

That exercises the exact path the real run takes — same script, same GPU pinning, same output layout — in about a minute. A4 is the heaviest stage, so if it passes the others will.

Then confirm three stages coexist on three GPUs, still on 8 images, by running these in three terminals at once:

```bash
DATA=coco8-seg.yaml EPOCHS=1 BATCH=2 bash experiments/scripts/run_stage.sh A0 1
DATA=coco8-seg.yaml EPOCHS=1 BATCH=2 bash experiments/scripts/run_stage.sh A1 2
DATA=coco8-seg.yaml EPOCHS=1 BATCH=2 bash experiments/scripts/run_stage.sh A2 3
```

This is worth doing once: it surfaces host-level limits — shared memory (§6.6), dataloader worker count, open file descriptors — that only appear when several stages run together, and it surfaces them in a minute rather than on day two.

### 6.3 One Stage, One GPU

A single stage is just the script with a stage and a GPU:

```bash
bash experiments/scripts/run_stage.sh A4 1
```

which runs, with `CUDA_VISIBLE_DEVICES=1` so the process sees only that card:

```bash
yolo segment train   model=experiments/configs/yolo26s-seg-carafe-aspp-deeplabv3plus.yaml   pretrained=checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt   data=coco.yaml epochs=100 batch=32 imgsz=640   device=0 workers=6 amp=True seed=0   project=experiments/results name=A4_1gpu
```

Notes for single-GPU stages:

- **No DDP anywhere.** One process, one GPU, `device=0` inside the pinned environment. Nothing to wrap in `torchrun`, no process group to hang, no rank-0 bookkeeping — and a stage that dies leaves nothing behind for the others to trip over.
- **`workers` is per stage.** Every terminal you open adds another `WORKERS` dataloader processes on the same host. Three stages at the default 6 is 18; keep the total at or below `nproc`.
- **`lr0` defaults suit this batch.** They are tuned around batch 64, so at `batch=32` on one GPU the stock values are reasonable. Leave them alone unless you change the batch.
- **Everything is written once**, under `experiments/results/<stage>_1gpu/`.

### 6.4 Running the Full A0-A4 Ablation (one stage per terminal)

No DDP. Each stage trains on one dedicated GPU, launched by hand in its own terminal, so the stages are fully independent processes: one crashing, OOMing, or being Ctrl-C'd never touches the others.

```bash
bash experiments/scripts/run_stage.sh <STAGE> <GPU>
```

**Wave 1** - open three terminals (or three `tmux` windows) and run one each:

```bash
# terminal 1
bash experiments/scripts/run_stage.sh A0 1

# terminal 2
bash experiments/scripts/run_stage.sh A1 2

# terminal 3
bash experiments/scripts/run_stage.sh A2 3
```

**Wave 2** - as each finishes, start the next stage in that freed terminal:

```bash
bash experiments/scripts/run_stage.sh A3 1
bash experiments/scripts/run_stage.sh A4 2
```

Each stage is pinned with `CUDA_VISIBLE_DEVICES=<gpu>` and runs with `device=0`, so it cannot touch a card another terminal is using. Output streams to the terminal and is tee'd to `experiments/results/logs/<stage>_1gpu_<timestamp>.log`. Results go to `<stage>_1gpu/`.

Use `tmux` rather than plain SSH sessions, so a dropped connection does not kill a multi-day stage:

```bash
tmux new -s a0    # then run the stage inside; detach with Ctrl-B D
tmux attach -t a0
```

#### Resuming

**Re-run the identical command.** That is the whole procedure:

```bash
bash experiments/scripts/run_stage.sh A0 1
```

The script decides what to do from what is on disk:

| State on disk | What happens |
| :-------------------------------- | :--------------------------------------------- |
| nothing yet | trains from the warm-start checkpoint |
| `weights/last.pt`, no `best.pt` | resumes from `last.pt` |
| `weights/best.pt` present | reports complete and exits 0, changing nothing |

So after a crash, an OOM, a reboot, or a closed laptop, you re-run the same line in each terminal and every stage picks up where it left off. `FORCE=1` retrains a finished stage from scratch.

The banner at the top of each run states which of the three it chose, so you can confirm at a glance:

```text
 mode    resuming from experiments/results/A0_1gpu/weights/last.pt
```

Note that `resume` restores the batch size, epoch count, and device from the checkpoint and ignores the environment overrides — that is what keeps a resumed stage comparable with the others. To change those, start a fresh run with `FORCE=1`.

#### Settings

Override by prefixing the command, e.g. `BATCH=64 bash experiments/scripts/run_stage.sh A4 2`:

`BATCH=32` (per stage - it owns one GPU), `EPOCHS=100`, `IMGSZ=640`, `WORKERS=6`, `SEED=0`, `DATA=coco.yaml`, `PROJECT=experiments/results`, `FORCE=0`, `PROBE=0`, `PROBE_FRACTION=0.01`.

**`batch` is per stage, not global.** `BATCH=32` means 32 images on that one card - the same per-GPU load as a DDP `batch=96` split three ways. Do not carry a DDP number over.

**Keep every setting identical across the five stages.** The ablation is only meaningful if `batch`, `imgsz`, `epochs`, and `seed` match; the only things that should differ are `model` and `pretrained`.

#### Measure the cost first

`PROBE=1` measures the stage and reports what the real run will cost, in a few minutes instead of days:

```bash
PROBE=1 bash experiments/scripts/run_stage.sh A4 1
```

```text
A4 measured on one GPU:
  train     38.2s on 0.01 of the data ->   63.7 min/epoch full data
  val       21.4s per epoch on the full val split
  epoch     64.0 min
  100 epochs: 4.45 days (106.7 h)
```

**How it measures**, because the naive version of this is badly wrong: `fraction` shrinks only the **train** split — validation always runs on the full val set. Timing one epoch with validation on and scaling by `1/fraction` therefore multiplies a fixed cost by 100. Instead the probe trains three epochs with `val=False` and differences epoch 2 against epoch 1, which cancels both one-time startup and validation; it uses three epochs rather than two because Ultralytics validates on the final epoch whatever `val=` says. Validation is then timed once on its own and added back a single time per epoch.

Probe A0 and A4 — the lightest and heaviest stages — to bracket the sweep. Run the probes in three terminals simultaneously if you want numbers that include the I/O contention three concurrent stages actually cause.

#### Host load

Three concurrent stages multiply every host-side cost:

- **Dataloader processes**: `WORKERS` per stage, so 6 x 3 = 18. Keep the total at or below `nproc`.
- **Disk**: three independent readers over the same 20 GB of COCO. On a network mount or spinning disk this, not the GPUs, sets your epoch time.
- **Shared memory**: see §6.6 - this is the one most likely to kill a long run.

### 6.5 Monitoring

```bash
watch -n 5 nvidia-smi                                   # utilization and VRAM headroom on 1,2,3
tail -f experiments/results/A4_3gpu_ddp/results.csv     # per-epoch metrics
```

If GPU utilization sits well below ~90%, the dataloader is the bottleneck — raise `workers`, or add `cache=ram` if the host has enough spare RAM (COCO-Seg at 640 px needs roughly 30+ GB).

### 6.6 Shared Memory (/dev/shm)

DataLoader workers hand batches to the training process through POSIX shared memory. Containers default `/dev/shm` to 64MB, and one 640px segmentation batch already exceeds that, so training fails — usually partway in, not at startup:

```text
RuntimeError: unable to allocate shared memory(shm) for file </torch_50121_1510199154_15>: No space left on device (28)
DataLoader worker (pid 12345) is killed by signal: Bus error.
```

Check it:

```bash
df -h /dev/shm
```

Rough requirement: `workers x prefetch(4) x batch x 3 x imgsz^2` bytes. At `WORKERS=6`, `BATCH=32`, `imgsz=640` that is **~900MB per stage**, and three concurrent stages need ~2.7GB. The dataloader warns at startup when `/dev/shm` is short of what it is about to demand.

**There is no library-side workaround.** Both of torch's sharing strategies allocate through `shm_open` under `/dev/shm`; `file_system` only changes how regions are named and reclaimed, so `set_sharing_strategy("file_system")` does **not** move batches off the mount and does not help here. Only these do:

**1. Enlarge it (best).** Inside the container as root, if it has `CAP_SYS_ADMIN`:

```bash
mount -o remount,size=16G /dev/shm
```

At container start:

```bash
docker run --shm-size=16g ...   # or --ipc=host
```

```yaml
services:
  train:
    shm_size: '16gb'
```

**2. Load data in-process.** `workers=0` uses no shared memory at all, because there are no worker processes:

```bash
WORKERS=0 bash experiments/scripts/run_stage.sh A0 1
```

This always works and needs no privileges, but the dataloader then competes with training for the main process and will underfeed the GPU. Acceptable for a probe; costly for a 100-epoch run.

**3. Shrink the demand.** Fewer workers and a smaller batch reduce it proportionally, but 64MB is roughly one batch, so no realistic setting fits — this only helps if `/dev/shm` is merely tight rather than tiny.

Fix `/dev/shm` before starting the sweep. Option 2 will get a probe through today; option 1 is what you want for the real runs.

---

## 7. Python API Training Interface

```python
from ultralytics import YOLO

# Load model with warm-started pretrained weights
model = YOLO("checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt")

# Train one stage on one GPU. Set CUDA_VISIBLE_DEVICES=<gpu> before launching
# python so this process sees only that card, then address it as device=0.
results = model.train(
    data="coco.yaml",
    epochs=100,
    batch=32,  # this stage owns one GPU, so batch is what fits on that card
    imgsz=640,
    device=0,
    workers=6,
    amp=True,
    seed=0,
    project="experiments/results",
    name="A4_1gpu",
)
```

Prefer `run_stage.sh` (§6.4) for real runs — it handles GPU pinning, logging, and resume. Use the Python API only for one-off experiments.

---

## 8. Resuming Interrupted Runs

`resume` restores the optimizer state, epoch counter, and the full argument set from the checkpoint, so no other flags are needed — and none are honored:

```bash
uv run --no-sync yolo segment train resume model=experiments/results/A4_3gpu_ddp/weights/last.pt
```

The resumed run reuses the saved `batch` and settings from the checkpoint, which is what keeps a resumed stage comparable with the others. To change them, start a fresh run with `FORCE=1`.

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
