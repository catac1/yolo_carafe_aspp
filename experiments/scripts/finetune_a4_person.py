# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Create a person-focused COCO manifest and fine-tune the A4 segmentation checkpoint."""

import random
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.utils import img2label_paths

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/workspace/coco_dataset/coco")
TRAIN_LIST = DATA_ROOT / "train2017.txt"
FOCUS_LIST = DATA_ROOT / "train_person_focus.txt"
DATA_CONFIG = ROOT / "experiments/configs/coco_remote_person_focus.yaml"
CHECKPOINT = ROOT / "runs/segment/experiments/results/A4_local-2/weights/best.pt"


def create_person_focus_list() -> None:
    """Repeat a fixed-seed half of person-positive training images once."""
    if not TRAIN_LIST.is_file():
        raise FileNotFoundError(f"COCO training image list not found: {TRAIN_LIST}")

    rng, image_paths, person_images, repeated = random.Random(0), [], 0, 0
    for line in TRAIN_LIST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image = Path(line.strip())
        image = (image if image.is_absolute() else DATA_ROOT / image).resolve()
        if not image.is_file():
            raise FileNotFoundError(
                f"Training image from {TRAIN_LIST} not found: {image}"
            )
        image_paths.append(str(image))
        label = Path(img2label_paths([str(image)])[0])
        is_person = label.is_file() and any(
            row.split()[:1] == ["0"]
            for row in label.read_text(encoding="utf-8").splitlines()
        )
        if is_person:
            person_images += 1
            if rng.random() < 0.5:
                image_paths.append(str(image))
                repeated += 1

    if not image_paths:
        raise ValueError(f"No training images found in {TRAIN_LIST}")
    FOCUS_LIST.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(image_paths):,} paths; repeated {repeated:,} of {person_images:,} person-positive images."
    )


def main() -> None:
    """Prepare the weighted manifest and start A4 fine-tuning."""
    create_person_focus_list()
    if not DATA_CONFIG.is_file() or not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Missing dataset config or starting checkpoint: {DATA_CONFIG}, {CHECKPOINT}"
        )
    YOLO(str(CHECKPOINT)).train(
        data=str(DATA_CONFIG),
        epochs=25,
        imgsz=640,
        batch=32,
        cos_lr=True,
        lr0=0.001,
        lrf=0.01,
        patience=7,
        close_mosaic=5,
        cls_pw=0.0,
        resume=False,
        save_period=5,
        seed=0,
        project=str(ROOT / "runs/segment/experiments/results"),
        name="A4_focus_person",
    )


if __name__ == "__main__":
    main()
