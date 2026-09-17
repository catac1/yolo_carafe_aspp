# YOLO26-Seg COCO Comparison

## Summary

This report compares three YOLO26 segmentation checkpoints on the same COCO 2017 validation split, with emphasis on the requested classes:

`person` (human), `bicycle`, `car`, `motorcycle`, `bus`, and `truck`.

The primary metric is mask AP50-95. Higher is better.

| Model                | Checkpoint                                                    | Training/evaluation role                                                                                  |
| -------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| A0 RTX6000 baseline  | `runs/segment/experiments/results/A0_local/weights/best.pt`   | Original YOLO26s-seg architecture trained locally on RTX6000                                              |
| A4 custom            | `runs/segment/experiments/results/A4_local-2/weights/best.pt` | CARAFE + ASPP + DeepLabV3+ configuration ([config](../configs/yolo26-seg-carafe-aspp-deeplabv3plus.yaml)) |
| Official Ultralytics | `checkpoints/yolo26s-seg.pt`                                  | Official provided checkpoint, evaluated directly                                                          |

The A4 model improves the fresh full-validation mask mAP50-95 from **0.38945** to **0.39161** versus A0, a gain of **+0.00216**. The official checkpoint reaches **0.39285**.

## Evaluation Protocol

| Item                              | Value                                                        |
| --------------------------------- | ------------------------------------------------------------ |
| Dataset                           | `G:\project4_data\coco_dataset\coco`                         |
| Split                             | COCO `val2017`                                               |
| Validation images                 | 5,000                                                        |
| Segmentation annotations          | `annotations/instances_val2017.json` and YOLO polygon labels |
| Image size                        | 640                                                          |
| Device                            | GPU 0, Quadro P2200                                          |
| Batch size                        | 4                                                            |
| Precision                         | FP32 (`half=False`)                                          |
| Confidence / IoU / max detections | `0.001 / 0.7 / 300`                                          |
| NMS mode                          | External NMS (`nms=None`)                                    |

All three models were evaluated with the same settings. The fresh validation values below are authoritative; training CSV values are included only as historical context.

## Aggregate Results

Values are rounded to three decimals. Mask metrics are the main result; box metrics are included as secondary detection context.

<table>
<thead>
<tr><th>Model</th><th>Box P</th><th>Box R</th><th>Box mAP50</th><th>Box mAP50-95</th><th>Mask P</th><th>Mask R</th><th>Mask mAP50</th><th>Mask mAP50-95</th></tr>
</thead>
<tbody>
<tr><td>A0 RTX6000 baseline</td><td>0.717</td><td>0.577</td><td>0.637</td><td><span style="background-color:#f4cccc;color:#7f0000;padding:2px 6px;border-radius:4px">0.467</span></td><td>0.717</td><td>0.557</td><td>0.609</td><td><span style="background-color:#f4cccc;color:#7f0000;padding:2px 6px;border-radius:4px">0.389</span></td></tr>
<tr><td>A4 custom</td><td>0.728</td><td>0.580</td><td>0.640</td><td><span style="background-color:#fff2cc;color:#7f6000;padding:2px 6px;border-radius:4px">0.470</span></td><td>0.725</td><td>0.558</td><td>0.612</td><td><span style="background-color:#fff2cc;color:#7f6000;padding:2px 6px;border-radius:4px">0.392</span></td></tr>
<tr><td>Official yolo26s-seg.pt</td><td>0.720</td><td>0.590</td><td>0.649</td><td><span style="background-color:#d9ead3;color:#1b5e20;padding:2px 6px;border-radius:4px">0.476</span></td><td>0.716</td><td>0.564</td><td>0.618</td><td><span style="background-color:#d9ead3;color:#1b5e20;padding:2px 6px;border-radius:4px">0.393</span></td></tr>
</tbody>
</table>

### A4 minus A0 aggregate delta

| Metric         |        Delta |
| -------------- | -----------: |
| Box precision  | **+0.01076** |
| Box recall     | **+0.00356** |
| Box mAP50      | **+0.00361** |
| Box mAP50-95   | **+0.00387** |
| Mask precision | **+0.00843** |
| Mask recall    | **+0.00173** |
| Mask mAP50     | **+0.00284** |
| Mask mAP50-95  | **+0.00216** |

## Requested-Class Mask Comparison

The colored AP50-95 cells rank the three models within each class: green = highest, yellow = middle, red = lowest. The delta column is A4 minus A0: green = improvement, red = decrease, yellow = near-zero change (`|delta| < 0.001`).

<table>
<thead>
<tr><th>COCO class</th><th>GT instances</th><th>A0 baseline<br>Mask AP50-95</th><th>A4 custom<br>Mask AP50-95</th><th>Official<br>Mask AP50-95</th><th>A4 − A0</th></tr>
</thead>
<tbody>
<tr><td>person (human)</td><td>10,777</td><td><span style="background-color:#fff2cc;color:#7f6000;padding:2px 6px;border-radius:4px">0.466</span></td><td><span style="background-color:#f4cccc;color:#7f0000;padding:2px 6px;border-radius:4px">0.465</span></td><td><span style="background-color:#d9ead3;color:#1b5e20;padding:2px 6px;border-radius:4px">0.468</span></td><td><span style="background-color:#f4cccc;color:#7f0000;padding:2px 6px;border-radius:4px">🔴 −0.001</span></td></tr>
<tr><td>bicycle</td><td>314</td><td><span style="background-color:#fff2cc;color:#7f6000;padding:2px 6px;border-radius:4px">0.201</span></td><td><span style="background-color:#f4cccc;color:#7f0000;padding:2px 6px;border-radius:4px">0.196</span></td><td><span style="background-color:#d9ead3;color:#1b5e20;padding:2px 6px;border-radius:4px">0.212</span></td><td><span style="background-color:#f4cccc;color:#7f0000;padding:2px 6px;border-radius:4px">🔴 −0.005</span></td></tr>
<tr><td>car</td><td>1,918</td><td><span style="background-color:#f4cccc;color:#7f0000;padding:2px 6px;border-radius:4px">0.375</span></td><td><span style="background-color:#fff2cc;color:#7f6000;padding:2px 6px;border-radius:4px">0.379</span></td><td><span style="background-color:#d9ead3;color:#1b5e20;padding:2px 6px;border-radius:4px">0.382</span></td><td><span style="background-color:#d9ead3;color:#1b5e20;padding:2px 6px;border-radius:4px">🟢 +0.004</span></td></tr>
<tr><td>motorcycle</td><td>367</td><td><span style="background-color:#f4cccc;color:#7f0000;padding:2px 6px;border-radius:4px">0.368</span></td><td><span style="background-color:#fff2cc;color:#7f6000;padding:2px 6px;border-radius:4px">0.378</span></td><td><span style="background-color:#d9ead3;color:#1b5e20;padding:2px 6px;border-radius:4px">0.388</span></td><td><span style="background-color:#d9ead3;color:#1b5e20;padding:2px 6px;border-radius:4px">🟢 +0.009</span></td></tr>
<tr><td>bus</td><td>283</td><td><span style="background-color:#fff2cc;color:#7f6000;padding:2px 6px;border-radius:4px">0.650</span></td><td><span style="background-color:#f4cccc;color:#7f0000;padding:2px 6px;border-radius:4px">0.649</span></td><td><span style="background-color:#d9ead3;color:#1b5e20;padding:2px 6px;border-radius:4px">0.654</span></td><td><span style="background-color:#fff2cc;color:#7f6000;padding:2px 6px;border-radius:4px">🟡 −0.001</span></td></tr>
<tr><td>truck</td><td>414</td><td><span style="background-color:#f4cccc;color:#7f0000;padding:2px 6px;border-radius:4px">0.348</span></td><td><span style="background-color:#d9ead3;color:#1b5e20;padding:2px 6px;border-radius:4px">0.356</span></td><td><span style="background-color:#fff2cc;color:#7f6000;padding:2px 6px;border-radius:4px">0.356</span></td><td><span style="background-color:#d9ead3;color:#1b5e20;padding:2px 6px;border-radius:4px">🟢 +0.008</span></td></tr>
</tbody>
</table>

### Detailed mask metrics

Each cell is `precision / recall / AP50`.

| Class          |           A0 baseline |             A4 custom | Official yolo26s-seg.pt |
| -------------- | --------------------: | --------------------: | ----------------------: |
| person (human) | 0.807 / 0.716 / 0.791 | 0.808 / 0.712 / 0.790 |   0.817 / 0.715 / 0.795 |
| bicycle        | 0.670 / 0.436 / 0.471 | 0.674 / 0.420 / 0.471 |   0.721 / 0.420 / 0.494 |
| car            | 0.717 / 0.595 / 0.646 | 0.720 / 0.600 / 0.653 |   0.723 / 0.586 / 0.652 |
| motorcycle     | 0.761 / 0.632 / 0.684 | 0.770 / 0.629 / 0.694 |   0.802 / 0.629 / 0.696 |
| bus            | 0.850 / 0.742 / 0.826 | 0.837 / 0.763 / 0.832 |   0.842 / 0.777 / 0.834 |
| truck          | 0.600 / 0.442 / 0.516 | 0.626 / 0.449 / 0.530 |   0.614 / 0.429 / 0.529 |

## Focus-Class Mean

This is the arithmetic mean over the six requested classes, not the official all-80-class COCO mAP.

| Model                   | Mask precision | Mask recall | Mask AP50 | Mask AP50-95 |
| ----------------------- | -------------: | ----------: | --------: | -----------: |
| A0 RTX6000 baseline     |          0.734 |       0.594 |     0.656 |        0.402 |
| A4 custom               |          0.739 |       0.596 |     0.661 |        0.404 |
| Official yolo26s-seg.pt |          0.753 |       0.593 |     0.667 |        0.410 |

## Findings

- **Overall:** A4 improves the baseline on aggregate mask AP50-95 by **+0.00216**, but remains **0.00124** below the official checkpoint.
- **Car:** A4 improves mask AP50-95 by **+0.004** over A0.
- **Motorcycle:** A4 shows the largest requested-class gain over A0 at **+0.009**.
- **Truck:** A4 improves over A0 by **+0.008** and is marginally above the official checkpoint (`0.35610` vs `0.35596`).
- **Person and bicycle:** A4 is slightly below A0, especially bicycle (`−0.005`).
- **Bus:** A4 is effectively tied with A0 within `0.001`, while the official checkpoint is higher.
- **Official checkpoint:** It is the strongest mask model on five of the six requested classes; A4 is strongest only on truck by a negligible margin.

## Historical Training-Run Context

The stored 100-epoch training logs reported:

| Run        | Stored best mask mAP50-95 | Stored best mask mAP50 |
| ---------- | ------------------------: | ---------------------: |
| A0 local   |                   0.39474 |                0.60960 |
| A4 local-2 |                   0.39683 |                0.61404 |

Those values differ slightly from the fresh common validation above because the report intentionally re-evaluates all three checkpoints under one controlled protocol.
