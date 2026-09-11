# Pretrained Weight Transfer Report

- **Source Baseline**: `checkpoints/yolo26s-seg.pt`
- **Target Custom Config**: `experiments/configs/yolo26s-seg-carafe.yaml`
- **Output Checkpoint**: `checkpoints/yolo26s-seg-carafe_pretrained.pt`

## Transfer Statistics

| Metric | Value |
|---|---|
| **Total Parameters** | 11,670,608 |
| **Transferred Parameters** | 11,546,542 (98.94%) |
| **Newly Initialized Parameters** | 165,066 (1.06%) |
| **Matched Tensors** | 844 |
| **Missing Tensors (Custom Initialized)** | 16 |
| **Shape Mismatch Tensors** | 0 |
| **Unused Baseline Tensors** | 0 |

## Newly Initialized Layers (CARAFE / Custom)
```text
model.11.compress.conv.weight
model.11.compress.bn.weight
model.11.compress.bn.bias
model.11.compress.bn.running_mean
model.11.compress.bn.running_var
model.11.compress.bn.num_batches_tracked
model.11.kernel_predictor.weight
model.11.kernel_predictor.bias
model.14.compress.conv.weight
model.14.compress.bn.weight
model.14.compress.bn.bias
model.14.compress.bn.running_mean
model.14.compress.bn.running_var
model.14.compress.bn.num_batches_tracked
model.14.kernel_predictor.weight
model.14.kernel_predictor.bias

```

## Shape Mismatches
```text
None
```
