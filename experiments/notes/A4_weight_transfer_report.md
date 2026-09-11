# Pretrained Weight Transfer Report

- **Source Baseline**: `checkpoints/yolo26s-seg.pt`
- **Target Custom Config**: `experiments/configs/yolo26s-seg-carafe-aspp-deeplabv3plus.yaml`
- **Output Checkpoint**: `checkpoints/yolo26s-seg-carafe-aspp-deeplabv3plus_pretrained.pt`

## Transfer Statistics

| Metric | Value |
|---|---|
| **Total Parameters** | 13,565,716 |
| **Transferred Parameters** | 10,775,127 (79.43%) |
| **Newly Initialized Parameters** | 2,833,097 (20.57%) |
| **Matched Tensors** | 798 |
| **Missing Tensors (Custom Initialized)** | 86 |
| **Shape Mismatch Tensors** | 0 |
| **Unused Baseline Tensors** | 604 |

## Newly Initialized Layers (CARAFE / Custom)
```text
model.11.branch1.conv.weight
model.11.branch1.bn.weight
model.11.branch1.bn.bias
model.11.branch1.bn.running_mean
model.11.branch1.bn.running_var
model.11.branch1.bn.num_batches_tracked
model.11.branch2.conv.weight
model.11.branch2.bn.weight
model.11.branch2.bn.bias
model.11.branch2.bn.running_mean
model.11.branch2.bn.running_var
model.11.branch2.bn.num_batches_tracked
model.11.branch3.conv.weight
model.11.branch3.bn.weight
model.11.branch3.bn.bias
model.11.branch3.bn.running_mean
model.11.branch3.bn.running_var
model.11.branch3.bn.num_batches_tracked
model.11.branch4.conv.weight
model.11.branch4.bn.weight
model.11.branch4.bn.bias
model.11.branch4.bn.running_mean
model.11.branch4.bn.running_var
model.11.branch4.bn.num_batches_tracked
model.11.branch5.conv.weight
model.11.branch5.bn.weight
model.11.branch5.bn.bias
model.11.branch5.bn.running_mean
model.11.branch5.bn.running_var
model.11.branch5.bn.num_batches_tracked
model.11.project.conv.weight
model.11.project.bn.weight
model.11.project.bn.bias
model.11.project.bn.running_mean
model.11.project.bn.running_var
model.11.project.bn.num_batches_tracked
model.12.compress.conv.weight
model.12.compress.bn.weight
model.12.compress.bn.bias
model.12.compress.bn.running_mean
model.12.compress.bn.running_var
model.12.compress.bn.num_batches_tracked
model.12.kernel_predictor.weight
model.12.kernel_predictor.bias
model.15.compress.conv.weight
model.15.compress.bn.weight
model.15.compress.bn.bias
model.15.compress.bn.running_mean
model.15.compress.bn.running_var
model.15.compress.bn.num_batches_tracked
... and more
```

## Shape Mismatches
```text
None
```
