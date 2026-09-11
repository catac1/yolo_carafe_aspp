# Pretrained Weight Transfer Report

- **Source Baseline**: `checkpoints/yolo26s-seg.pt`
- **Target Custom Config**: `experiments/configs/yolo26s-seg-aspp.yaml`
- **Output Checkpoint**: `checkpoints/yolo26s-seg-aspp_pretrained.pt`

## Transfer Statistics

| Metric | Value |
|---|---|
| **Total Parameters** | 13,736,328 |
| **Transferred Parameters** | 11,546,542 (84.06%) |
| **Newly Initialized Parameters** | 2,232,838 (15.94%) |
| **Matched Tensors** | 844 |
| **Missing Tensors (Custom Initialized)** | 36 |
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

```

## Shape Mismatches
```text
None
```
