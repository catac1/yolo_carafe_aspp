# PLAN.md — Custom YOLO26s-Seg with CARAFE + DeepLabV3+ Context/Decoder

## 0. Scope and fixed assumptions

This project modifies **Ultralytics YOLO26s instance segmentation** while keeping the training data unchanged.

### Fixed baseline

- Base model: `yolo26s-seg.pt`
- Task: instance segmentation
- Dataset: COCO 2017 instance segmentation / Ultralytics COCO-Seg
- Image size for the first controlled experiments: `640`
- Number of classes: `80`
- Training hardware: **3 × NVIDIA RTX 3090 24 GB**
- Environment/package manager: **uv**
- Training mode: Linux + CUDA + Ultralytics multi-GPU DDP
- Baseline pretrained weights: reuse every compatible `yolo26s-seg.pt` tensor
- New CARAFE / ASPP / DeepLabV3+ decoder parameters: randomly initialized
- No new images, masks, classes, or annotations are required.

### COCO-Seg

Use the same COCO 2017 instance-segmentation data family used for YOLO26 segmentation pretraining:

```text
train2017: 118,287 images
val2017:     5,000 images
classes:        80
```

The Ultralytics `coco.yaml` download definition retrieves the train and validation images plus segmentation labels. The held-out COCO test set is not required for this project.

---

# 1. Project objective

Build the architecture in controlled stages rather than introducing every modification at once.

```text
A0  YOLO26s-Seg baseline
 |
 +-- A1  YOLO26s-Seg + CARAFE
 |
 +-- A2  YOLO26s-Seg + ASPP
 |
 +-- A3  YOLO26s-Seg + CARAFE + ASPP
 |
 +-- A4  YOLO26s-Seg
          + CARAFE top-down upsampling
          + DeepLabV3+-style ASPP context
          + low-level decoder/prototype refinement
```

The experiment is successful only if we can determine **which component caused an improvement or regression**.

Do not begin with A4.

---

# 2. Architectural idea

## 2.1 Baseline concept

The relevant high-level path is approximately:

```text
Backbone
  |
  +-- P2 / stride 4
  +-- P3 / stride 8
  +-- P4 / stride 16
  +-- P5 / stride 32
                  |
             high-level context
                  |
             top-down neck
             /           \
       Upsample         Upsample
         P5->P4           P4->P3
                  |
             Segment head
             /          \
         detection     masks
```

## 2.2 CARAFE modification

Replace only the selected **top-down fixed upsampling** operations.

```text
Before:

P5
 |
nn.Upsample x2
 |
P4 fusion
 |
nn.Upsample x2
 |
P3 fusion
```

```text
After:

P5
 |
CARAFE x2
 |
P4 fusion
 |
CARAFE x2
 |
P3 fusion
```

CARAFE must:

- preserve the input channel count initially;
- change only spatial size;
- support scale factor `2`;
- be differentiable;
- work with AMP;
- work under DDP;
- avoid custom CUDA code for the first version.

The first implementation should use ordinary PyTorch operations such as:

- `Conv2d`
- `softmax`
- `unfold`
- `reshape/view`
- `permute`
- `pixel_shuffle` or an equivalent rearrangement

Optimize later only after the reference implementation is correct.

## 2.3 ASPP / DeepLabV3+ context modification

Add an ASPP-style context block on the high-level feature path.

Conceptually:

```text
                       +-- 1x1 conv ----------------+
                       |                            |
P5/context feature ----+-- dilated conv rate r1 ----+
                       |                            |
                       +-- dilated conv rate r2 ----+--> concat --> projection
                       |                            |
                       +-- dilated conv rate r3 ----+
                       |                            |
                       +-- global pooling ----------+
```

For an ASPP placed on a stride-32 feature map, do **not** blindly copy DeepLab dilation rates intended for output-stride 16.

Initial experiment:

```text
P5 output stride: 32
initial dilation candidates: [3, 6, 9]
```

A later controlled experiment may tap a stride-16 feature and test rates such as:

```text
[6, 12, 18]
```

The ASPP output should initially preserve the expected channel interface so the existing YOLO neck can remain unchanged.

## 2.4 Final DeepLabV3+-style decoder idea

A4 should borrow the important DeepLabV3+ decoder principle:

> combine semantically strong high-level context with a shallow, higher-resolution feature.

Use a low-level YOLO feature such as **P2 / stride 4**.

Concept:

```text
P5 / stride 32
     |
    ASPP
     |
 CARAFE / learned upsampling
     |
     +-----------------------------+
                                   |
P2 / stride 4                      |
     |                             |
1x1 projection                     |
     |                             |
     +---------- concat <----------+
                    |
               refinement
                    |
              mask prototypes
                    |
             instance masks
```

The first A4 implementation should refine the **segmentation/prototype path**, not redesign the complete detection head.

This makes the experiment segmentation-specific while preserving as much pretrained detection functionality as possible.

---

# 3. Repository strategy

Because model parsing and internal modules will be modified, use a **source checkout of Ultralytics**, not only a published wheel.

Recommended approach:

1. Fork `ultralytics/ultralytics`.
2. Clone the fork.
3. Use the cloned repository itself as the uv project.
4. Keep custom modules isolated and clearly named.
5. Keep upstream changes easy to merge.

Example:

```bash
git clone https://github.com/<YOUR_ACCOUNT>/ultralytics.git custom-yolo26-seg
cd custom-yolo26-seg

git remote add upstream https://github.com/ultralytics/ultralytics.git
git checkout -b experiment/carafe-deeplabv3plus
```

Do not copy the whole Ultralytics package into another project unless there is a strong reason. A fork keeps parser changes, YAML changes, tests, and upstream history together.

---

# 4. uv environment setup

## 4.1 Pin Python

Use Python 3.12 unless the CUDA/PyTorch combination on the training server requires a different supported version.

```bash
uv python install 3.12
uv python pin 3.12
```

The cloned Ultralytics repository already contains `pyproject.toml`, so do not run `uv init` over it.

## 4.2 Create/sync the uv environment

```bash
uv sync
```

`uv` installs the local project in editable mode by default, which is exactly what is wanted here: edits under `ultralytics/` become visible without repeatedly reinstalling the package.

Create/update the lock file explicitly when desired:

```bash
uv lock
```

Commit these reproducibility files:

```text
pyproject.toml
uv.lock
.python-version
```

## 4.3 Add project-only development dependencies if needed

Only add packages that are actually used.

Example:

```bash
uv add --dev pytest
```

Do not install random duplicate Torch/OpenCV packages manually after `uv sync`.

---

# 5. GPU/CUDA preflight — 3 × RTX 3090

Before downloading COCO or touching custom architecture code, verify the training environment.

## 5.1 OS-level GPU check

```bash
nvidia-smi
```

Expected:

```text
GPU 0  RTX 3090
GPU 1  RTX 3090
GPU 2  RTX 3090
```

All three cards should show approximately 24 GB VRAM and should not already be occupied by unrelated jobs.

## 5.2 PyTorch CUDA check through uv

```bash
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, round(p.total_memory / 1024**3, 2), "GiB")
PY
```

Required result:

```text
cuda available: True
gpu count: 3
```

Do not start training if `torch.cuda.device_count()` is less than 3.

## 5.3 PyTorch CUDA build mismatch

If `uv sync` resolved a Torch build that cannot use the server CUDA setup:

1. check NVIDIA driver version;
2. choose an officially supported PyTorch CUDA wheel/index;
3. declare that choice in the uv project;
4. regenerate `uv.lock`;
5. repeat the GPU test.

Do not fix this with ad-hoc `pip install` commands outside uv because that destroys environment reproducibility.

---

# 6. Create project directories

Add a small experiment layer without disturbing upstream layout.

```bash
mkdir -p experiments/configs
mkdir -p experiments/scripts
mkdir -p experiments/results
mkdir -p experiments/notes
mkdir -p checkpoints
mkdir -p tests/custom
```

Recommended additions:

```text
custom-yolo26-seg/
├── ultralytics/
│   ├── nn/
│   │   └── modules/
│   │       └── custom_seg.py
│   └── cfg/
│       └── models/
│           └── 26/
├── experiments/
│   ├── configs/
│   │   ├── yolo26s-seg-carafe.yaml
│   │   ├── yolo26s-seg-aspp.yaml
│   │   ├── yolo26s-seg-carafe-aspp.yaml
│   │   └── yolo26s-seg-carafe-deeplabv3plus.yaml
│   ├── scripts/
│   │   ├── inspect_model.py
│   │   ├── transfer_weights.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── results/
│   └── notes/
├── checkpoints/
├── tests/
│   └── custom/
└── PLAN.md
```

The exact Ultralytics source files to edit should be confirmed against the checked-out revision before implementing parser changes.

---

# 7. Record the source revision

Before modifications:

```bash
git rev-parse HEAD
uv run yolo version
uv run python -c "import ultralytics; print(ultralytics.__version__, ultralytics.__file__)"
```

Save this information in:

```text
experiments/notes/environment.md
```

Also record:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Every experiment must be traceable back to:

- Git commit;
- `uv.lock`;
- checkpoint;
- model YAML;
- dataset;
- training arguments.

---

# 8. Download the YOLO26s-Seg baseline checkpoint

Ultralytics model files download automatically on first use.

Trigger only the model download:

```bash
uv run python - <<'PY'
from ultralytics import YOLO

model = YOLO("yolo26s-seg.pt")
print(model)
PY
```

After successful download, keep a stable local copy:

```bash
mv yolo26s-seg.pt checkpoints/yolo26s-seg.pt
```

If the file already exists, do not overwrite it blindly.

Optional checksum:

```bash
sha256sum checkpoints/yolo26s-seg.pt
```

Record the checksum in `experiments/notes/environment.md`.

---

# 9. COCO-Seg dataset location

For a GPU server, do not put the full dataset inside the Git repository.

Example preferred location:

```text
/home/user/yolo_custom/coco_dataset/coco
```

or another fast local NVMe path.

Set the Ultralytics dataset directory:

```bash
uv run yolo settings datasets_dir=/home/user/yolo_custom/coco_dataset
```

Check:

```bash
uv run yolo settings
```

Expected final structure is conceptually:

```text
/home/user/yolo_custom/coco_dataset/
└── coco/
    ├── images/
    │   ├── train2017/
    │   └── val2017/
    ├── labels/
    │   ├── train2017/
    │   └── val2017/
    ├── train2017.txt
    └── val2017.txt
```

The exact layout should follow the `coco.yaml` version in the checked-out Ultralytics revision.

---

# 10. Download COCO-Seg and validate the baseline at the same time

Use the official `coco.yaml`.

A validation run is useful because it:

1. resolves/downloads missing COCO-Seg files;
2. confirms labels can be read;
3. confirms the checkpoint is valid;
4. confirms segmentation validation works;
5. produces the baseline metric.

Run initially on one GPU:

```bash
uv run yolo segment val \
  model=checkpoints/yolo26s-seg.pt \
  data=coco.yaml \
  imgsz=640 \
  device=0 \
  nms=False
```

The first invocation may download roughly the full train/val COCO material required by the dataset definition.

Do not download `test2017` unless a later official COCO test-dev submission is planned.

---

# 11. Verify dataset integrity

After download:

```bash
find /home/user/yolo_custom/coco_dataset/coco/images/train2017 -type f | wc -l
find /home/user/yolo_custom/coco_dataset/coco/images/val2017 -type f | wc -l
```

Expected image counts:

```text
train2017 ≈ 118287
val2017   ≈ 5000
```

Then inspect random image/label pairs.

Create a small script:

```text
experiments/scripts/check_coco.py
```

It should verify:

- image path exists;
- segmentation label exists when expected;
- polygon coordinates parse correctly;
- class IDs fall within `0..79`;
- no NaN/Inf coordinates;
- visualization can render several polygons over their images.

Do not begin full training until at least a few dozen random samples have been visually checked.

---

# 12. Baseline reproduction before custom code

Before modifying the architecture, establish A0.

## 12.1 Baseline validation

Save:

```text
box mAP50-95
mask mAP50-95
precision
recall
validation loss
GPU memory
validation runtime
```

Output directory:

```text
experiments/results/A0_baseline_val/
```

## 12.2 Baseline short training test

Do not start a full COCO baseline training immediately.

First confirm training/DDP with a short subset run.

Example:

```bash
uv run yolo segment train \
  model=checkpoints/yolo26s-seg.pt \
  data=coco.yaml \
  imgsz=640 \
  epochs=1 \
  fraction=0.01 \
  batch=24 \
  device=0,1,2 \
  workers=4 \
  amp=True \
  project=experiments/results \
  name=A0_ddp_smoke
```

Important:

```text
batch=24 = global batch
         = 8 images/GPU with 3 GPUs
```

For Ultralytics multi-GPU DDP, use an explicit global batch size that is a multiple of `3`.

`workers=4` is per rank, so this can create approximately 12 data-loader workers across three ranks.

Increase workers only if CPU and storage can support it.

---

# 13. Locate the exact YOLO26-Seg architecture

Before writing a custom YAML:

```bash
find ultralytics/cfg/models/26 -maxdepth 1 -type f | sort | grep -i seg
```

Read the YOLO26 segmentation config used by the current revision.

Do not assume a configuration from an old YOLO11/YOLOv8 tutorial is identical.

Identify:

```text
P2 feature source
P3 feature source
P4 feature source
P5 feature source
two top-down nn.Upsample modules
SPPF / attention/context modules
Segment26 or equivalent segmentation head
prototype/mask branch implementation
parse_model() handling
```

Record a layer-index map in:

```text
experiments/notes/yolo26s_layer_map.md
```

Example format:

```text
index   output stride   role
-----   -------------   -----------------
...     4               P2 low-level
...     8               P3
...     16              P4
...     32              P5
...     -               first top-down upsample
...     -               second top-down upsample
...     -               segmentation head
```

This file becomes the source of truth for custom YAML edits.

---

# 14. Implement CARAFE first

Create:

```text
ultralytics/nn/modules/custom_seg.py
```

Initial public API:

```python
class CARAFE(nn.Module):
    def __init__(
        self,
        c1: int,
        scale: int = 2,
        kernel_size: int = 5,
        encoder_kernel: int = 3,
        compressed_channels: int = 64,
    ):
        ...
```

Initial contract:

```text
input : [B, C, H, W]
output: [B, C, H*2, W*2]
```

For version 1:

- `c_out == c_in`;
- scale fixed/tested at 2;
- pure PyTorch implementation;
- no CUDA extension;
- no channel reduction on output;
- predicted reassembly weights normalized with softmax;
- unit tests compare exact shape and gradient flow.

## 14.1 CARAFE unit tests

Required tests:

```text
[B=1, C=64, H=20, W=20] -> [1, 64, 40, 40]
[B=2, C=128, H=40, W=40] -> [2, 128, 80, 80]
[B=2, C=256, H=80, W=80] -> [2, 256, 160, 160]
```

Also test:

```python
loss = output.mean()
loss.backward()
```

Confirm all learnable CARAFE parameters receive gradients.

Test AMP:

```text
FP32
CUDA AMP FP16
```

Run:

```bash
uv run pytest tests/custom -q
```

---

# 15. Register CARAFE with the Ultralytics model parser

Depending on the checked-out source revision, this normally requires some combination of:

```text
ultralytics/nn/modules/__init__.py
ultralytics/nn/tasks.py
parse_model()
```

Requirements:

1. YAML name `CARAFE` resolves to the Python class.
2. Parser knows the input channel count.
3. Parser knows the CARAFE output channel count remains unchanged.
4. Scale factor affects H/W only.
5. `model.info()` succeeds.
6. forward pass succeeds at 640×640.

Do not modify generic parser behavior more than necessary.

---

# 16. Create A1 — YOLO26s-Seg + CARAFE

Copy the currently resolved YOLO26 segmentation architecture into:

```text
experiments/configs/yolo26s-seg-carafe.yaml
```

Change only the two selected top-down upsampling layers:

```text
nn.Upsample
    ->
CARAFE
```

Everything else remains baseline-compatible.

## A1 acceptance test

```bash
uv run python experiments/scripts/inspect_model.py \
  experiments/configs/yolo26s-seg-carafe.yaml
```

It must print:

```text
model successfully constructed
task = segment
scale = s
forward output valid
no channel mismatch
```

Verify that the custom YAML truly builds the **s** scale.

Do not rely solely on the filename. Compare parameter count and model summary to the expected YOLO26s-scale baseline and investigate any large unexpected difference.

---

# 17. Implement ASPP

Add to `custom_seg.py`:

```python
class ASPP(nn.Module):
    ...
```

Initial branches:

```text
1x1 conv
3x3 dilated conv r=3
3x3 dilated conv r=6
3x3 dilated conv r=9
global average pooling branch
concat
projection
```

Initial contract:

```text
input : [B, C, H, W]
output: [B, C, H, W]
```

Preserving the channel interface makes weight transfer and YAML integration easier.

ASPP unit tests:

```text
shape preservation
gradient flow
AMP
odd/even H,W
global-pool branch interpolation
```

---

# 18. Create A2 — YOLO26s-Seg + ASPP

Use baseline upsampling.

Insert ASPP only at the chosen high-level context point.

Initial preferred experiment:

```text
P5/high-level feature
        |
       ASPP
        |
existing YOLO top-down neck
```

Do not add CARAFE yet.

Run:

```text
construction test
forward test
1-GPU smoke train
3-GPU smoke train
```

Only after A2 is stable should A3 be built.

---

# 19. Create A3 — YOLO26s-Seg + CARAFE + ASPP

Combine the already-tested A1 and A2 changes.

No new idea is introduced here.

```text
P5
 |
ASPP
 |
CARAFE
 |
P4 fusion
 |
CARAFE
 |
P3 fusion
 |
existing segment head
```

This experiment answers:

> Do CARAFE and ASPP work together, or does one erase the benefit of the other?

---

# 20. A4 — DeepLabV3+-style low-level segmentation decoder

Implement this only after A0–A3 train correctly.

## 20.1 Goal

Preserve YOLO26 detection behavior as much as possible while enhancing the **mask prototype path**.

Use:

```text
high-level context = ASPP(P5 or selected high-level feature)
low-level detail   = P2 / stride 4 feature
```

## 20.2 Low-level projection

Project the P2 feature to a much smaller channel dimension before fusion.

Concept:

```text
P2
 |
1x1 Conv
 |
low-level projection
```

Do not feed the full raw P2 channel tensor directly into an expensive decoder.

## 20.3 High-level upsampling

Upsample the ASPP output toward the low-level spatial resolution.

Prefer reusing the tested CARAFE implementation.

If several 2× stages are required:

```text
stride 32 -> 16 -> 8 -> 4
```

use staged CARAFE rather than immediately inventing an arbitrary x8 operator.

## 20.4 Decoder fusion

```text
ASPP high-level feature
        |
     CARAFE
        |
     CARAFE
        |
     CARAFE
        |
        +----------------+
                         |
P2 -> 1x1 projection ----+--> concat
                              |
                         3x3 refinement
                              |
                         3x3 refinement
                              |
                         mask prototype path
```

This is the DeepLabV3+-style part of the project.

## 20.5 Preserve compatible pretrained weights

Prefer extending/subclassing the existing YOLO26 segmentation head so that:

- compatible detection-head parameter names remain loadable;
- compatible mask coefficient branches remain loadable when possible;
- only new ASPP/decoder/prototype layers are new/random;
- transfer statistics are logged.

Avoid rewriting the complete head from scratch unless required.

---

# 21. Pretrained-weight transfer strategy

For custom YAML models, use the supported pattern:

```python
from ultralytics import YOLO

model = YOLO("experiments/configs/yolo26s-seg-carafe.yaml")
model.load("checkpoints/yolo26s-seg.pt")
```

or the equivalent CLI `pretrained=` flow.

For every variant record:

```text
number of parameters loaded
number of parameters newly initialized
unexpected keys
missing keys
shape-mismatched keys
```

Expected logic:

```text
unchanged YOLO layers -> pretrained
CARAFE              -> random init
ASPP                -> random init
new decoder         -> random init
new prototype parts -> random init
```

A huge drop in transferred parameters is a warning that the custom architecture or naming changed too much.

---

# 22. Smoke-test ladder

Do not jump directly to 118k COCO training images.

Use this ladder for **every architecture variant**.

## Stage S1 — CPU/GPU construction test

```text
build model
print model
model.info()
one random tensor
one forward pass
```

## Stage S2 — COCO8-Seg

Use the tiny built-in segmentation dataset:

```bash
uv run yolo segment train \
  model=experiments/configs/<MODEL>.yaml \
  pretrained=checkpoints/yolo26s-seg.pt \
  data=coco8-seg.yaml \
  imgsz=640 \
  epochs=2 \
  batch=4 \
  device=0 \
  workers=2 \
  project=experiments/results \
  name=<MODEL>_coco8_smoke
```

Purpose:

```text
parser
data
forward
loss
backward
validation
checkpoint saving
```

## Stage S3 — COCO 1% single GPU

```bash
uv run yolo segment train \
  model=experiments/configs/<MODEL>.yaml \
  pretrained=checkpoints/yolo26s-seg.pt \
  data=coco.yaml \
  imgsz=640 \
  epochs=1 \
  fraction=0.01 \
  batch=-1 \
  device=0 \
  workers=4 \
  amp=True \
  project=experiments/results \
  name=<MODEL>_single_gpu_profile
```

This single-GPU run is also used to estimate a viable per-GPU batch size.

## Stage S4 — COCO 1% on 3 GPUs

Start conservatively:

```bash
uv run yolo segment train \
  model=experiments/configs/<MODEL>.yaml \
  pretrained=checkpoints/yolo26s-seg.pt \
  data=coco.yaml \
  imgsz=640 \
  epochs=1 \
  fraction=0.01 \
  batch=24 \
  device=0,1,2 \
  workers=4 \
  amp=True \
  project=experiments/results \
  name=<MODEL>_3gpu_smoke
```

Only after S4 passes is the architecture allowed into full COCO training.

---

# 23. Batch sizing for 3 × RTX 3090

Ultralytics DDP requires an explicit global batch size for multi-GPU training.

The global batch should be divisible by three.

Useful candidates:

```text
global  per GPU
------  -------
12         4
24         8
36        12
48        16
60        20
```

Do not assume that because the baseline fits batch 48, the CARAFE + DeepLabV3+ model will also fit it.

CARAFE and high-resolution decoder features can significantly increase activation memory.

## Recommended procedure

1. Run single-GPU auto-batch/profiling on the exact variant.
2. Observe peak VRAM during real segmentation batches.
3. Select a lower per-GPU number with safety margin.
4. Multiply by 3.
5. Run a 3-GPU 1% COCO smoke test.
6. Increase only if VRAM has sufficient margin.

Initial conservative A3/A4 full-run candidate:

```text
batch=24
```

If stable with substantial free VRAM:

```text
24 -> 36 -> 48
```

Do not depend on automatic OOM recovery in DDP. Multi-GPU OOM should be treated as a failed configuration and the global batch reduced manually.

---

# 24. Data-loader settings

Start with:

```text
workers=4
```

With three DDP ranks this means roughly:

```text
4 workers/rank × 3 ranks = 12 workers
```

If the GPUs wait on data and the server has enough CPU/NVMe throughput:

```text
workers=6
workers=8
```

may be tested.

Do not increase workers just because GPUs exist. Measure:

```text
GPU utilization
CPU utilization
I/O wait
VRAM
RAM
samples/sec
```

Start with:

```text
cache=False
```

COCO is large enough that `cache=ram` should not be enabled unless the server's system RAM is known to be sufficient.

---

# 25. Full training commands

Create a script rather than manually typing different commands for every run.

Recommended:

```text
experiments/scripts/train.py
```

It should accept:

```text
--variant
--batch
--epochs
--device
--workers
--fraction
--resume
```

The equivalent conceptual full run for A3:

```bash
uv run yolo segment train \
  model=experiments/configs/yolo26s-seg-carafe-aspp.yaml \
  pretrained=checkpoints/yolo26s-seg.pt \
  data=coco.yaml \
  imgsz=640 \
  epochs=100 \
  batch=24 \
  device=0,1,2 \
  workers=4 \
  amp=True \
  cache=False \
  project=experiments/results \
  name=A3_carafe_aspp
```

A4:

```bash
uv run yolo segment train \
  model=experiments/configs/yolo26s-seg-carafe-deeplabv3plus.yaml \
  pretrained=checkpoints/yolo26s-seg.pt \
  data=coco.yaml \
  imgsz=640 \
  epochs=100 \
  batch=24 \
  device=0,1,2 \
  workers=4 \
  amp=True \
  cache=False \
  project=experiments/results \
  name=A4_carafe_deeplabv3plus
```

Do not permanently fix `batch=24` until the memory smoke tests are finished.

---

# 26. Baseline fairness

A meaningful architecture comparison requires keeping these fixed:

```text
dataset
train/val split
image size
epochs
optimizer policy
learning-rate policy
augmentation
seed policy
AMP mode
validation settings
pretrained checkpoint
evaluation split
```

The only intended variable should be the architecture.

If batch size must differ because A4 uses more memory, record it explicitly. If practical, use gradient accumulation or another controlled strategy so effective optimization conditions remain comparable.

---

# 27. Experiment matrix

Minimum ablation:

| ID | Model | CARAFE | ASPP | DeepLabV3+ low-level decoder | COCO |
|---|---|---:|---:|---:|---:|
| A0 | YOLO26s-Seg | No | No | No | Same |
| A1 | custom | Yes | No | No | Same |
| A2 | custom | No | Yes | No | Same |
| A3 | custom | Yes | Yes | No | Same |
| A4 | custom | Yes | Yes | Yes | Same |

Do not claim A4 works because it beats A0 alone.

The important comparisons are:

```text
A1 - A0 = CARAFE contribution
A2 - A0 = ASPP contribution
A3 - A1 = ASPP contribution when CARAFE exists
A3 - A2 = CARAFE contribution when ASPP exists
A4 - A3 = DeepLabV3+ low-level decoder contribution
```

---

# 28. Metrics to save

For every experiment:

## Accuracy

```text
box mAP50
box mAP50-95
mask mAP50
mask mAP50-95
precision
recall
```

## Complexity

```text
parameters
GFLOPs
checkpoint size
```

## Training performance

```text
images/sec
epoch duration
peak VRAM per GPU
GPU utilization
CPU utilization
```

## Inference performance

Test the same hardware/settings:

```text
latency
FPS
VRAM
batch=1 performance
```

## Segmentation-specific visual review

Create a fixed validation image list containing:

```text
small objects
thin structures
people
vehicles
overlapping objects
occluded objects
large objects
boundary-heavy scenes
```

Save exactly the same images for A0–A4 so qualitative differences can be inspected side by side.

---

# 29. Optional boundary metrics

COCO mask mAP remains the primary metric.

For additional analysis, derive boundary-focused metrics from the existing masks; no additional labeling is necessary.

Possible later metrics:

```text
Boundary IoU
boundary F-score
mask edge error
small-object mask AP
medium-object mask AP
large-object mask AP
```

These are useful because CARAFE/DeepLab-style refinement may improve boundaries without a dramatic change in overall mask mAP.

Do not introduce these metrics until the standard COCO evaluation is correct.

---

# 30. Checkpoint/resume policy

COCO training is long enough that resume support must be tested.

Before full training, verify:

```bash
uv run yolo segment train resume model=<path-to-last.pt>
```

or the exact resume syntax supported by the checked-out revision.

Save periodic checkpoints:

```text
best.pt
last.pt
periodic epoch checkpoints
```

Do not keep only `best.pt`; debugging custom architecture training sometimes requires examining intermediate epochs.

---

# 31. DDP-specific failure checklist

If 3-GPU training hangs or fails:

1. confirm all three GPUs are visible;
2. confirm no GPU is already out of memory;
3. test the same model on one GPU;
4. reduce global batch;
5. reduce `workers`;
6. inspect DDP subprocess error, not only the parent exception;
7. confirm custom module has no device-hardcoded tensors;
8. confirm all temporary tensors are created on `x.device`;
9. ensure no Python state depends on a single rank;
10. ensure shapes are identical across ranks for equivalent batches.

Never write custom code such as:

```python
torch.zeros(...).cuda(0)
```

Instead use:

```python
x.new_zeros(...)
```

or:

```python
torch.zeros(..., device=x.device)
```

---

# 32. CARAFE-specific failure checklist

Watch for:

```text
huge temporary tensors from unfold()
incorrect softmax dimension
incorrect pixel-shuffle ordering
channel/scale mismatch
OOM at P3/P2 resolution
non-contiguous tensor/view errors
AMP numerical issues
slow Python/tensor rearrangement
```

The pure PyTorch CARAFE reference implementation is for correctness first.

If it becomes the speed bottleneck, optimize only after the architecture experiment has shown value.

---

# 33. ASPP-specific failure checklist

Watch for:

```text
dilation too large for a 20x20 P5 feature
BatchNorm behavior in global-pooling branch
channel explosion after concat
incorrect interpolation mode/align_corners
large memory increase
context feature overwhelming local detail
```

Keep ASPP output channels bounded.

Do not concatenate huge full-width branches and leave them unprojected.

---

# 34. DeepLabV3+ decoder-specific failure checklist

Watch for:

```text
P2 feature makes memory usage explode
high-level and low-level feature sizes do not match
too many P2 channels
prototype resolution changes unexpectedly
mask coefficient/prototype dimensions mismatch
pretrained Segment26 head weights stop loading
box accuracy drops because mask changes leaked into detect branch
```

A4 should first alter the segmentation pathway as locally as possible.

---

# 35. Recommended implementation milestones

## Milestone M0 — Environment and data

- [ ] fork/clone Ultralytics
- [ ] `uv python pin`
- [ ] `uv sync`
- [ ] all 3 RTX 3090 GPUs visible
- [ ] CUDA Torch verified
- [ ] `yolo26s-seg.pt` downloaded
- [ ] COCO-Seg downloaded
- [ ] COCO sample polygons visually verified
- [ ] A0 validation metric saved
- [ ] 3-GPU baseline smoke test passes

## Milestone M1 — CARAFE

- [ ] pure PyTorch CARAFE implemented
- [ ] shape tests pass
- [ ] gradient test passes
- [ ] AMP test passes
- [ ] registered in parser
- [ ] custom YAML builds
- [ ] pretrained weights transfer
- [ ] COCO8-Seg training passes
- [ ] 1% COCO single-GPU passes
- [ ] 1% COCO 3-GPU passes
- [ ] A1 ready for full training

## Milestone M2 — ASPP

- [ ] ASPP implemented
- [ ] dilation strategy documented
- [ ] tests pass
- [ ] A2 YAML builds
- [ ] weight transfer checked
- [ ] smoke ladder passes
- [ ] A2 ready for full training

## Milestone M3 — CARAFE + ASPP

- [ ] A3 YAML created
- [ ] no new unrelated architecture changes
- [ ] smoke ladder passes
- [ ] VRAM measured
- [ ] DDP global batch selected
- [ ] A3 ready for full training

## Milestone M4 — DeepLabV3+ decoder

- [ ] P2 low-level source identified
- [ ] P2 projection implemented
- [ ] ASPP high-level source identified
- [ ] staged CARAFE upsampling implemented
- [ ] decoder fusion implemented
- [ ] prototype path adapted
- [ ] compatible detection weights retained
- [ ] shape tests pass
- [ ] mask loss runs
- [ ] 3-GPU smoke test passes
- [ ] A4 ready for full training

## Milestone M5 — Ablation

- [ ] A0 results
- [ ] A1 results
- [ ] A2 results
- [ ] A3 results
- [ ] A4 results
- [ ] standard COCO metrics collected
- [ ] speed/VRAM measured
- [ ] fixed qualitative examples generated
- [ ] conclusions written

---

# 36. Definition of done

The project is complete when all of the following are true:

1. The entire environment can be recreated with:

```bash
uv sync
```

2. All three RTX 3090s are used through Ultralytics DDP.

3. COCO-Seg is the only training/validation dataset used for the primary study.

4. `yolo26s-seg.pt` is the common pretrained initialization source.

5. CARAFE and ASPP can each be enabled independently.

6. The combined CARAFE + ASPP architecture trains successfully.

7. The DeepLabV3+-style low-level decoder trains successfully.

8. Compatible pretrained weights are transferred and the transfer report is saved.

9. A0–A4 are evaluated on the same COCO validation split.

10. The final report includes both accuracy and cost:

```text
mask mAP
box mAP
parameters
GFLOPs
VRAM
training throughput
inference latency/FPS
```

11. The conclusion answers:

```text
Did CARAFE improve mask quality?
Did ASPP improve context/scale handling?
Did combining them help?
Did the DeepLabV3+ low-level decoder improve masks enough to justify its cost?
Where did accuracy improve: small, medium, or large objects?
Was the gain worth the extra VRAM/FLOPs/latency?
```

---

# 37. Recommended first execution sequence

Run these in order.

```bash
# 1. Clone/fork project
git clone https://github.com/<YOUR_ACCOUNT>/ultralytics.git custom-yolo26-seg
cd custom-yolo26-seg
git checkout -b experiment/carafe-deeplabv3plus

# 2. uv environment
uv python install 3.12
uv python pin 3.12
uv sync

# 3. Check GPUs
nvidia-smi

uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

# 4. Experiment directories
mkdir -p experiments/{configs,scripts,results,notes}
mkdir -p checkpoints tests/custom

# 5. Download baseline checkpoint
uv run python - <<'PY'
from ultralytics import YOLO
YOLO("yolo26s-seg.pt")
PY

mv yolo26s-seg.pt checkpoints/yolo26s-seg.pt

# 6. Configure dataset path
uv run yolo settings datasets_dir=/home/user/yolo_custom/coco_dataset

# 7. Download/resolve COCO and reproduce baseline validation
uv run yolo segment val \
  model=checkpoints/yolo26s-seg.pt \
  data=coco.yaml \
  imgsz=640 \
  device=0 \
  nms=False

# 8. Baseline 3-GPU smoke test
uv run yolo segment train \
  model=checkpoints/yolo26s-seg.pt \
  data=coco.yaml \
  imgsz=640 \
  epochs=1 \
  fraction=0.01 \
  batch=24 \
  device=0,1,2 \
  workers=4 \
  amp=True \
  project=experiments/results \
  name=A0_ddp_smoke
```

Only after all eight steps succeed should implementation of `CARAFE` begin.

---

# 38. First coding task after preparation

The first actual custom-model task is deliberately small:

```text
Implement CARAFE only.
```

Do **not** implement ASPP or the DeepLabV3+ decoder in the same commit.

Suggested first custom commit:

```text
feat(seg): add tested CARAFE upsampling module
```

Second:

```text
feat(seg): register CARAFE in YOLO model parser
```

Third:

```text
feat(seg): add YOLO26s-Seg CARAFE experiment config
```

After A1 passes the full smoke-test ladder, continue with ASPP.

This commit structure keeps model failures attributable to one architectural change at a time.
