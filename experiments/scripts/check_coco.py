# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Script to verify COCO segmentation dataset integrity, image counts, and label formatting."""

import argparse
from pathlib import Path

from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER


def check_coco_integrity(data_cfg: str = "coco.yaml", num_samples: int = 20):
    """Verify image counts, polygon formatting, and class range for COCO-Seg."""
    LOGGER.info(f"Loading dataset configuration from {data_cfg}...")
    data = check_det_dataset(data_cfg)

    train_path = Path(data.get("train", ""))
    val_path = Path(data.get("val", ""))
    names = data.get("names", {})
    num_classes = len(names)

    LOGGER.info(f"Dataset root: {data.get('path')}")
    LOGGER.info(f"Number of classes: {num_classes}")

    # Check image files
    for split_name, img_path in [("train", train_path), ("val", val_path)]:
        if not img_path.exists():
            LOGGER.warning(f"Split '{split_name}' path not found: {img_path}")
            continue

        if img_path.is_file():  # .txt file with image paths
            with open(img_path, encoding="utf-8") as f:
                img_files = [line.strip() for line in f if line.strip()]
        else:
            img_files = list(img_path.glob("*.*"))

        LOGGER.info(f"Split '{split_name}' contains {len(img_files):,} images.")

        # Check random label files
        label_errors = 0
        samples_checked = 0
        for img_file in img_files[:num_samples]:
            img_p = Path(img_file)
            if not img_p.is_absolute():
                img_p = Path(data["path"]) / img_p
            # Corresponding label path: replace /images/ with /labels/ and .jpg with .txt
            label_p = Path(str(img_p.parent).replace("images", "labels")) / f"{img_p.stem}.txt"
            if not label_p.exists():
                continue

            samples_checked += 1
            with open(label_p, encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    if not (0 <= cls_id < num_classes):
                        LOGGER.error(f"{label_p}:{line_idx + 1} Invalid class {cls_id}")
                        label_errors += 1
                    if len(coords) < 6:
                        LOGGER.error(f"{label_p}:{line_idx + 1} Too few coordinates for polygon: {len(coords)}")
                        label_errors += 1
                    if any(c < 0.0 or c > 1.05 for c in coords):
                        LOGGER.error(f"{label_p}:{line_idx + 1} Coordinate out of [0, 1] range")
                        label_errors += 1

        LOGGER.info(f"Checked {samples_checked} label samples in '{split_name}'. Errors detected: {label_errors}")

    LOGGER.info("COCO-Seg dataset verification complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify COCO-Seg Dataset Integrity")
    parser.add_argument("--data", type=str, default="coco.yaml", help="Path or name of dataset yaml")
    parser.add_argument("--samples", type=int, default=50, help="Number of label samples to inspect per split")
    args = parser.parse_args()
    check_coco_integrity(args.data, args.samples)
