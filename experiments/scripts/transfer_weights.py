# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pretrained weight transfer and audit script for YOLO26 segmentation experiments."""

import argparse
from pathlib import Path
import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import torch_safe_load


def transfer_weights(config_path: str, checkpoint_path: str, output_path: str, report_path: str = None, layer_shift_after: int = None, layer_shift: int = 1):
    """Transfers compatible weights from a baseline checkpoint to a custom architecture model."""
    print(f"Building custom model from config: {config_path}")
    custom_yolo = YOLO(config_path)
    model = custom_yolo.model

    print(f"Loading baseline checkpoint: {checkpoint_path}")
    ckpt = torch_safe_load(checkpoint_path)
    if isinstance(ckpt, tuple):
        ckpt = ckpt[0]
    source_model = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    source_state_dict = source_model.state_dict() if hasattr(source_model, "state_dict") else source_model

    target_state_dict = model.state_dict()

    matched_keys = []
    missing_keys = []
    shape_mismatched_keys = []
    unexpected_keys = []

    new_state_dict = {}

    for name, target_param in target_state_dict.items():
        src_name = name
        if src_name not in source_state_dict and layer_shift_after is not None:
            parts = name.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_idx = int(parts[1])
                if layer_idx > layer_shift_after:
                    candidate = ".".join([parts[0], str(layer_idx - layer_shift)] + parts[2:])
                    if candidate in source_state_dict:
                        src_name = candidate

        if src_name in source_state_dict:
            src_param = source_state_dict[src_name]
            if src_param.shape == target_param.shape:
                new_state_dict[name] = src_param
                matched_keys.append(name)
            else:
                shape_mismatched_keys.append((name, tuple(src_param.shape), tuple(target_param.shape)))
                new_state_dict[name] = target_param
        else:
            missing_keys.append(name)
            new_state_dict[name] = target_param

    for name in source_state_dict.keys():
        if name not in target_state_dict:
            unexpected_keys.append(name)

    model.load_state_dict(new_state_dict)

    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    transferred_params = sum(target_state_dict[k].numel() for k in matched_keys)
    new_params = sum(target_state_dict[k].numel() for k in missing_keys) + sum(target_state_dict[k[0]].numel() for k in shape_mismatched_keys)
    transfer_ratio = (transferred_params / total_params) * 100 if total_params > 0 else 0.0

    print(f"Transfer Summary:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Transferred Parameters: {transferred_params:,} ({transfer_ratio:.2f}%)")
    print(f"  New/Randomly Initialized Parameters: {new_params:,} ({100 - transfer_ratio:.2f}%)")
    print(f"  Matched Tensors: {len(matched_keys)}")
    print(f"  Missing Tensors (new custom modules): {len(missing_keys)}")
    print(f"  Shape Mismatches: {len(shape_mismatched_keys)}")
    print(f"  Unused Baseline Tensors: {len(unexpected_keys)}")

    # Save transferred model checkpoint
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    custom_yolo.save(output_path)
    print(f"Saved transferred checkpoint to: {output_path}")

    # Generate Markdown Report
    if report_path:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_content = f"""# Pretrained Weight Transfer Report

- **Source Baseline**: `{checkpoint_path}`
- **Target Custom Config**: `{config_path}`
- **Output Checkpoint**: `{output_path}`

## Transfer Statistics

| Metric | Value |
|---|---|
| **Total Parameters** | {total_params:,} |
| **Transferred Parameters** | {transferred_params:,} ({transfer_ratio:.2f}%) |
| **Newly Initialized Parameters** | {new_params:,} ({100 - transfer_ratio:.2f}%) |
| **Matched Tensors** | {len(matched_keys)} |
| **Missing Tensors (Custom Initialized)** | {len(missing_keys)} |
| **Shape Mismatch Tensors** | {len(shape_mismatched_keys)} |
| **Unused Baseline Tensors** | {len(unexpected_keys)} |

## Newly Initialized Layers (CARAFE / Custom)
```text
{chr(10).join(missing_keys[:50])}
{"... and more" if len(missing_keys) > 50 else ""}
```

## Shape Mismatches
```text
{chr(10).join([f"{k}: src {s} vs dst {d}" for k, s, d in shape_mismatched_keys]) if shape_mismatched_keys else "None"}
```
"""
        report_file.write_text(report_content, encoding="utf-8")
        print(f"Saved transfer report to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer weights from baseline to custom model")
    parser.add_argument("--config", type=str, default="experiments/configs/yolo26s-seg-carafe.yaml", help="Path to custom model config")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/yolo26s-seg.pt", help="Path to baseline checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/yolo26s-seg-carafe_pretrained.pt", help="Path for output checkpoint")
    parser.add_argument("--report", type=str, default="experiments/notes/A1_weight_transfer_report.md", help="Path for markdown transfer report")
    parser.add_argument("--layer-shift-after", type=int, default=None, help="Layer index after which target layers shift")
    parser.add_argument("--layer-shift", type=int, default=1, help="Offset amount for shifted layers")
    args = parser.parse_args()

    transfer_weights(args.config, args.checkpoint, args.output, args.report, args.layer_shift_after, args.layer_shift)
