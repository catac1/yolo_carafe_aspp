# YOLO26-Seg COCO Comparison

## Summary

This report now includes the full A0–A4 experiment series using its saved training-validation logs. It also retains the
previously completed common-protocol comparison for A0, A4, and the official checkpoint. No inference or validation was
rerun for this update.

The requested focus classes are:

`person` (human), `bicycle`, `car`, `motorcycle`, `bus`, and `truck`.

The primary metric is mask AP50-95. Higher is better.

| Run | Intended variant | Saved best checkpoint |
| --- | --- | --- |
| A0 | YOLO26s-seg baseline | `runs/segment/experiments/results/A0_local/weights/best.pt` |
| A1 | CARAFE ([config](../configs/yolo26-seg-carafe.yaml)) | `runs/segment/experiments/results/A1_local/weights/best.pt` |
| A2 | ASPP-only ([config](../configs/yolo26-seg-aspp.yaml)); see the run caveat below | `runs/segment/experiments/results/A2_local/weights/best.pt` |
| A3 | CARAFE + ASPP ([config](../configs/yolo26-seg-carafe-aspp.yaml)) | `runs/segment/experiments/results/A3_local/weights/best.pt` |
| A4 | CARAFE + ASPP + DeepLabV3+ ([config](../configs/yolo26-seg-carafe-aspp-deeplabv3plus.yaml)) | `runs/segment/experiments/results/A4_local-2/weights/best.pt` |
| Official | Official Ultralytics checkpoint | `checkpoints/yolo26s-seg.pt` |

### Saved A0–A4 training-validation results

These values come from the already-saved `results.csv` files. For each run, the row is the epoch with the highest mask
mAP50-95; all values in that row are reported together. The saved arguments share `coco.yaml`, 640 image size, batch
32, 100 epochs, and seed 0. The GPU IDs differ. These are per-run best validation results, not a new common-protocol
re-evaluation.

The mask mAP50-95 cells rank the five saved runs: green = highest, yellow = second, orange = tied middle, and red = lowest.

<table>
<thead>
<tr><th>Run</th><th>Best epoch</th><th>Box P</th><th>Box R</th><th>Box mAP50</th><th>Box mAP50-95</th><th>Mask P</th><th>Mask R</th><th>Mask mAP50</th><th>Mask mAP50-95</th></tr>
</thead>
<tbody>
<tr><td>A0</td><td>91</td><td>0.71405</td><td>0.57778</td><td>0.63649</td><td>0.46617</td><td>0.71449</td><td>0.55642</td><td>0.60960</td><td><span style="background-color:#f4cccc;color:#7f0000;padding:2px 6px;border-radius:4px">0.39474</span></td></tr>
<tr><td>A1</td><td>92</td><td><strong>0.72561</strong></td><td>0.57583</td><td>0.63665</td><td>0.46789</td><td>0.71518</td><td><strong>0.56126</strong></td><td>0.60992</td><td><span style="background-color:#fff2cc;color:#7f6000;padding:2px 6px;border-radius:4px">0.39619</span></td></tr>
<tr><td>A2</td><td>91</td><td>0.70833</td><td>0.57733</td><td>0.63635</td><td>0.46715</td><td>0.70948</td><td>0.55764</td><td>0.61014</td><td><span style="background-color:#fce5cd;color:#783f04;padding:2px 6px;border-radius:4px">0.39533</span></td></tr>
<tr><td>A3</td><td>91</td><td>0.70833</td><td>0.57733</td><td>0.63635</td><td>0.46715</td><td>0.70948</td><td>0.55764</td><td>0.61014</td><td><span style="background-color:#fce5cd;color:#783f04;padding:2px 6px;border-radius:4px">0.39533</span></td></tr>
<tr><td>A4</td><td>92</td><td>0.72519</td><td>0.58144</td><td><strong>0.64082</strong></td><td><strong>0.47056</strong></td><td><strong>0.72381</strong></td><td>0.56082</td><td><strong>0.61404</strong></td><td><span style="background-color:#d9ead3;color:#1b5e20;padding:2px 6px;border-radius:4px"><strong>0.39683</strong></span></td></tr>
</tbody>
</table>

| Run | Saved mask mAP50-95 | Delta vs A0 |
| --- | ------------------: | ----------: |
| A0 | 0.39474 | baseline |
| A1 | 0.39619 | +0.00145 |
| A2 | 0.39533 | +0.00059 |
| A3 | 0.39533 | +0.00059 |
| A4 | 0.39683 | +0.00209 |

**A2 run caveat:** `A2_local/args.yaml` records `checkpoints/yolo26s-seg-carafe-aspp_pretrained.pt` as its starting model. That is the
CARAFE+ASPP checkpoint used by A3; the intended ASPP-only checkpoint is `checkpoints/yolo26s-seg-aspp_pretrained.pt`
([A2 transfer report](A2_weight_transfer_report.md)). A2 and A3 therefore have identical saved best-row metrics to the
precision shown, and A2 should not be treated as a valid ASPP-only ablation.

In these stored logs, A4 has the best overall mask mAP50-95 (**0.39683**), followed by A1 (**0.39619**), A2/A3
(**0.39533**), and A0 (**0.39474**). A1 has the highest mask recall; A4 leads the other aggregate mask metrics.

## Previously completed common-validation protocol

The common-protocol values below were already measured for A0, A4, and the official checkpoint. A1–A3 were not part of
that comparison and are not added to it here; this preserves the existing measurements without rerunning validation.

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

A0, A4, and the official checkpoint were evaluated with the same settings. The fresh values below are authoritative for
that three-model comparison; the A0–A4 training-log table above is a separate historical comparison.

## Existing common-protocol aggregate results (A0, A4, and official)

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

## Existing requested-class mask comparison (A0, A4, and official)

The colored AP50-95 cells rank these three previously measured models within each class: green = highest, yellow = middle,
red = lowest. The delta column is A4 minus A0: green = improvement, red = decrease, yellow = near-zero change
(`|delta| < 0.001`). The saved training CSVs for A1–A3 contain aggregate metrics only; no new class-wise evaluation was
run, so those variants are intentionally absent from this table.

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

## Findings from the existing common-protocol class comparison

- **Overall:** A4 improves the baseline on aggregate mask AP50-95 by **+0.00216**, but remains **0.00124** below the official checkpoint.
- **Car:** A4 improves mask AP50-95 by **+0.004** over A0.
- **Motorcycle:** A4 shows the largest requested-class gain over A0 at **+0.009**.
- **Truck:** A4 improves over A0 by **+0.008** and is marginally above the official checkpoint (`0.35610` vs `0.35596`).
- **Person and bicycle:** A4 is slightly below A0, especially bicycle (`−0.005`).
- **Bus:** A4 is effectively tied with A0 within `0.001`, while the official checkpoint is higher.
- **Official checkpoint:** It is the strongest mask model on five of the six requested classes; A4 is strongest only on truck by a negligible margin.

The stored A0–A4 values above differ from the existing common-validation results because the latter were produced in a
separate controlled evaluation. Do not compare numbers across those two tables as if they were from one evaluation pass.
