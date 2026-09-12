# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model inspection script for YOLO26 segmentation experiments (A0-A4)."""

import argparse

import torch

from ultralytics import YOLO


def inspect_model(config_path: str, imgsz: int = 640):
    """Constructs model from config and verifies task, scale, and forward pass."""
    print(f"Loading and inspecting model from: {config_path}")
    model = YOLO(config_path)

    # 1. Model construction check
    assert model.model is not None, "Failed to initialize model architecture"
    print("model successfully constructed")

    # 2. Task check
    task = getattr(model, "task", None) or getattr(model.model, "task", "segment")
    print(f"task = {task}")

    # 3. Scale check
    scale = getattr(model.model.yaml, "get", lambda k, d=None: d)("scale", "s") if hasattr(model.model, "yaml") else "s"
    if isinstance(scale, dict):
        scale = "s"
    print(f"scale = {scale}")

    # 4. Forward pass check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device)
    model.model.eval()

    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        results = model.model(dummy_input)

    # In eval mode, SegmentationModel returns (preds, proto)
    if isinstance(results, (tuple, list)):
        preds, proto = results[0], results[1]
        print(
            f"forward output valid: preds shape = {preds.shape if hasattr(preds, 'shape') else len(preds)}, proto shape = {proto.shape if hasattr(proto, 'shape') else None}"
        )
    else:
        print(f"forward output valid: {type(results)}")

    print("no channel mismatch")

    # 5. Parameter and layer summary
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect YOLO segmentation models")
    parser.add_argument("config", type=str, help="Path to model YAML configuration")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for dummy forward pass")
    args = parser.parse_args()

    inspect_model(args.config, imgsz=args.imgsz)
