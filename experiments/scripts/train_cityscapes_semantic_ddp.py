# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Prepare Cityscapes semantic masks and train the A4-derived model with DDP."""

import os
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = ROOT / "experiments/configs/cityscapes_semantic.yaml"
MODEL_CONFIG = ROOT / "experiments/configs/yolo26s-sem-carafe-aspp.yaml"
CHECKPOINT = Path(
    os.getenv(
        "CITYSCAPES_CHECKPOINT",
        "/workspace/yolo_carafe_aspp/runs/segment/experiments/results/A4_local-2/weights/best.pt",
    )
)


def _has_pngs(path: Path) -> bool:
    """Return whether a directory contains at least one PNG file."""
    return path.is_dir() and any(path.glob("*.png"))


def dataset_root() -> Path:
    """Read the dataset root from the dataset YAML so one path controls preparation and training."""
    configured = Path(YAML.load(DATA_CONFIG)["path"])
    return configured if configured.is_absolute() else (DATA_CONFIG.parent / configured).resolve()


def prepare_cityscapes() -> None:
    """Flatten extracted Cityscapes images and labelIds masks into the YAML layout."""
    data_root = dataset_root()
    output_pairs = [
        (data_root / "images" / split, data_root / "masks" / split)
        for split in ("train", "val")
    ]
    if all(_has_pngs(images) and _has_pngs(masks) for images, masks in output_pairs):
        print(f"Using existing prepared Cityscapes dataset: {data_root}")
        return

    image_root, mask_root = data_root / "leftImg8bit", data_root / "gtFine"
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(
            f"Expected extracted Cityscapes directories {image_root} and {mask_root}. "
            "Extract leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip first."
        )

    copied = 0
    for split in ("train", "val"):
        source_images = image_root / split
        destination_images = data_root / "images" / split
        destination_masks = data_root / "masks" / split
        destination_images.mkdir(parents=True, exist_ok=True)
        destination_masks.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(source_images.rglob("*_leftImg8bit.png"))
        if not image_paths:
            raise FileNotFoundError(f"No Cityscapes images found under {source_images}")

        for image_path in image_paths:
            relative = image_path.relative_to(source_images)
            mask_path = mask_root / split / relative.parent / image_path.name.replace(
                "_leftImg8bit.png", "_gtFine_labelIds.png"
            )
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing semantic mask for {image_path}: {mask_path}")

            image_name = image_path.name.replace("_leftImg8bit", "")
            mask_name = mask_path.name.replace("_gtFine_labelIds", "")
            image_destination = destination_images / image_name
            mask_destination = destination_masks / mask_name
            if not image_destination.exists():
                shutil.copy2(image_path, image_destination)
            if not mask_destination.exists():
                shutil.copy2(mask_path, mask_destination)
            copied += 1

    print(f"Prepared {copied:,} Cityscapes image/mask pairs under {data_root}")


def main() -> None:
    """Prepare Cityscapes and launch three-GPU DDP fine-tuning."""
    prepare_cityscapes()
    for required in (DATA_CONFIG, MODEL_CONFIG, CHECKPOINT):
        if not required.is_file():
            raise FileNotFoundError(f"Required file not found: {required}")

    # The A4 instance-segmentation checkpoint supplies compatible backbone and neck weights.
    # SemanticSegment is newly initialized for the 19 Cityscapes classes.
    model = YOLO(str(MODEL_CONFIG))
    model.load(str(CHECKPOINT))
    model.train(
        data=str(DATA_CONFIG),
        task="semantic",
        epochs=80,
        imgsz=640,
        batch=6,  # 2 images per GPU with devices 1, 2, and 3
        device=[1, 2, 3],
        workers=4,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.0005,
        cos_lr=True,
        patience=20,
        close_mosaic=10,
        cache=False,
        amp=True,
        plots=True,
        resume=False,
        save_period=5,
        seed=0,
        project=str(ROOT / "runs/semantic/experiments/results"),
        name="A4_cityscapes_semantic_ddp",
    )


if __name__ == "__main__":
    main()
