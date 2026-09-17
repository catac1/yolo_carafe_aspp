# A4 Class-Focused Fine-Tuning Plan

This plan prioritizes improving A4's person (human) mask performance while protecting its gains on car, motorcycle, and
truck. Bicycle and bus remain secondary targets. It uses the common COCO 2017 validation results summarized below.

## 1. Starting evidence

On the common COCO 2017 validation run, A4's overall mask AP50-95 was **0.39161**, versus **0.38945** for the
RTX6000 baseline and **0.39285** for the official checkpoint. The table summarizes the six classes of interest; AP
values are rounded to three decimals and the A4-minus-A0 deltas use the underlying full-precision values.

| COCO class | YOLO ID | A0 AP50-95 | A4 AP50-95 | Official AP50-95 |  A4 − A0 | Initial action                           |
| ---------- | ------: | ---------: | ---------: | ---------------: | -------: | ---------------------------------------- |
| Person     |       0 |      0.466 |      0.465 |            0.468 | -0.00137 | Primary target; prioritize hard examples |
| Bicycle    |       1 |      0.201 |      0.196 |            0.212 | -0.00539 | Secondary target                         |
| Car        |       2 |      0.375 |      0.379 |            0.382 | +0.00383 | Monitor; avoid initial oversampling      |
| Motorcycle |       3 |      0.368 |      0.378 |            0.388 | +0.00910 | Monitor; avoid initial oversampling      |
| Bus        |       5 |      0.650 |      0.649 |            0.654 | -0.00095 | Secondary target                         |
| Truck      |       7 |      0.348 |      0.356 |            0.356 | +0.00776 | Monitor; avoid initial oversampling      |

The stored A4 training curve peaked at epoch 92 (mask mAP50-95 **0.39683**) and was **0.39261** at epoch 100. From
epoch 92 to 100, training segmentation loss fell from **2.11067** to **2.00897**, while validation segmentation loss
rose from **2.18391** to **2.19697**. This is a reason to test a short, lower-learning-rate fine-tune rather than
simply extending the original run unchanged. The stored training metric and common-validation metric above come from
different evaluation runs and should not be compared as if they were the same measurement.

## 2. Dataset configuration and targeted sampling

The custom remote config is `experiments/configs/coco_remote_person_focus.yaml`. It points to the remote dataset,
preserves the original 80 COCO class names, and uses the generated person-focused manifest:

```yaml
path: /workspace/coco_dataset/coco
train: train_person_focus.txt
val: val2017.txt
test: test-dev2017.txt
```

The script `experiments/scripts/finetune_a4_person.py` reads the existing `train.txt` and segmentation labels,
then writes `/workspace/coco_dataset/coco/train_person_focus.txt` on the remote machine. That generated list is not
checked into the repo because it belongs beside the remote dataset.

For the first person-focused run:

1. Include every original training image once in `train_person_focus.txt`.
2. For a controlled initial boost, add one extra occurrence for a fixed-seed 50% sample of images containing person
   (ID 0). This gives person-positive images about 1.5× average image-level sampling; avoid doubling every person image
   in the first run.
3. If you have a reliable hard-example list, prioritize person images with small, occluded, crowded, truncated, or
   previously missed/poorly segmented instances when selecting that 50% sample.
4. Keep every annotation for each repeated image. Image-level sampling also repeats its co-occurring classes.
5. Leave validation unchanged and do not repeat validation paths.

Person is common in COCO (10,777 person instances in this validation split), so measure the effect of the controlled
sampling before increasing it. If person AP improves without an overall regression, a follow-up can raise the repeat
rate or add a limited extra repeat for hard person examples. Keep bicycle and bus at their normal sampling rate in this
first run so the person effect is isolated. Car, motorcycle, and truck should be monitored rather than upweighted
because A4 already improved over A0 on those classes.

Do not pass `classes=[0, 1, 2, 3, 5, 7]` or `single_cls=True` for this 80-class fine-tune. Filtering classes removes
other labels from training images and can teach the model that those objects are background.

## 3. Controlled fine-tuning runs

The script runs the person-focused experiment from A4 `best.pt`. For a sampler control, run the same settings from the
same checkpoint with a separate config that points `train:` to the original `train.txt`; keep the validation
split and all other settings identical.

| Setting                 | Initial value                                                          |
| ----------------------- | ---------------------------------------------------------------------- |
| Starting checkpoint     | `runs/segment/experiments/results/A4_local-2/weights/best.pt`          |
| Resume optimizer state  | `False` — start a new fine-tuning schedule from the checkpoint weights |
| Epochs / early stopping | `25` / `patience=7`                                                    |
| Image size / batch      | `640` / `32` if it fits the target GPU                                 |
| Initial learning rate   | `0.001`                                                                |
| LR schedule             | Cosine (`cos_lr=True`)                                                 |
| Final LR fraction       | `lrf=0.01` (final LR about `1e-5`)                                     |
| Mosaic close            | `5`                                                                    |
| Checkpoint interval     | `save_period=5`                                                        |
| `cls_pw`                | `0.0` for the first comparison, to isolate sampling                    |

Use this cosine schedule for both the control and person-focused runs. The original A4 run used linear decay
(`cos_lr=False`), so cosine is a shared fine-tuning change rather than a difference between the two experiments.

Run the script from the repository root on the remote training machine:

```bash
uv run --no-sync python experiments/scripts/finetune_a4_person.py
```

Do not use `cls_pw` as the person boost: in this repo it applies inverse-frequency weights across all classes, so it
does not let you prioritize person and may favor rarer classes instead. It also weights classification loss rather
than directly weighting mask loss. Keep `cls_pw=0.0` for this person-focused sampling experiment.

## 4. Validation and checkpoint selection

- Validate A4 `best.pt`, the control, and targeted checkpoints on the same unmodified COCO `val2017.txt` split at
  640 pixels, using the same evaluation settings.
- Record mask AP50-95 for person, bicycle, car, motorcycle, bus, and truck, plus all-class mask mAP50-95.
- Treat person mask AP50-95 as the primary success metric. Also report the other five classes and their macro-average;
  use car, motorcycle, truck, and all-class mAP as regression checks.
- Evaluate the periodic checkpoints as well as `best.pt`: Ultralytics selects `best.pt` using overall fitness, which
  may not select the checkpoint with the best target-class average.
- Keep the targeted run only if person gains are visible on the held-out validation set and overall performance does
  not materially regress. If it helps, tune the repeat rate in a later run; do not change sampling and learning rate
  together in the first comparison.
