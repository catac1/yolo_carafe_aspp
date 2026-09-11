# YOLO26s-Seg Layer Index Architecture & Feature Map Reference

This document maps the architectural layer indices of Ultralytics YOLO26s-Seg (`ultralytics/cfg/models/26/yolo26-seg.yaml` with scale `s`) to facilitate precise layer referencing across the A0–A4 experiment progression.

## Layer Map

| Index | Type | Arguments | Stride | Role in Baseline (A0) | Modification in A1..A4 |
|---|---|---|---|---|---|
| **0** | `Conv` | `[64, 3, 2]` | 2 | P1 / Stride 2 | Backbone downsample |
| **1** | `Conv` | `[128, 3, 2]` | 4 | **P2 / Stride 4 (low-level feature)** | **Source for DeepLabV3+ decoder (A4)** |
| **2** | `C3k2` | `[256, False, 0.25]` | 4 | P2 stage refinement | Backbone |
| **3** | `Conv` | `[256, 3, 2]` | 8 | **P3 / Stride 8** | Backbone downsample |
| **4** | `C3k2` | `[512, False, 0.25]` | 8 | P3 stage feature | Skip connection to neck (Layer 15) |
| **5** | `Conv` | `[512, 3, 2]` | 16 | **P4 / Stride 16** | Backbone downsample |
| **6** | `C3k2` | `[512, True]` | 16 | P4 stage feature | Skip connection to neck (Layer 12) |
| **7** | `Conv` | `[1024, 3, 2]` | 32 | **P5 / Stride 32** | Backbone downsample |
| **8** | `C3k2` | `[1024, True]` | 32 | P5 stage feature | Backbone |
| **9** | `SPPF` | `[1024, 5, 3, True]` | 32 | SPPF spatial pyramid pooling | Backbone multi-scale pooling |
| **10** | `C2PSA` | `[1024]` | 32 | **P5 context output** | **Insertion point for ASPP (A2, A3, A4)** |
| **11** | `nn.Upsample` | `[None, 2, 'nearest']` | 16 | **First top-down upsample (P5 -> P4)** | **Replaced with CARAFE(2, 5, 3, 64) (A1, A3, A4)** |
| **12** | `Concat` | `[[-1, 6], 1]` | 16 | Fuse upsampled P5 with backbone P4 | Top-down neck fusion |
| **13** | `C3k2` | `[512, True]` | 16 | P4 neck fusion block | Neck feature |
| **14** | `nn.Upsample` | `[None, 2, 'nearest']` | 8 | **Second top-down upsample (P4 -> P3)** | **Replaced with CARAFE(2, 5, 3, 64) (A1, A3, A4)** |
| **15** | `Concat` | `[[-1, 4], 1]` | 8 | Fuse upsampled P4 with backbone P3 | Top-down neck fusion |
| **16** | `C3k2` | `[256, True]` | 8 | P3/8 small object feature | Output to Segment head (P3) |
| **17** | `Conv` | `[256, 3, 2]` | 16 | Bottom-up downsample P3 -> P4 | Bottom-up path |
| **18** | `Concat` | `[[-1, 13], 1]` | 16 | Fuse bottom-up P3 with neck P4 | Bottom-up neck fusion |
| **19** | `C3k2` | `[512, True]` | 16 | P4/16 medium object feature | Output to Segment head (P4) |
| **20** | `Conv` | `[512, 3, 2]` | 32 | Bottom-up downsample P4 -> P5 | Bottom-up path |
| **21** | `Concat` | `[[-1, 10], 1]` | 32 | Fuse bottom-up P4 with backbone P5 | Bottom-up neck fusion |
| **22** | `C3k2` | `[1024, True, 0.5, True]` | 32 | P5/32 large object feature | Output to Segment head (P5) |
| **23** | `Segment26` | `[nc, 32, 256]` | [8, 16, 32] | Multi-scale segmentation head | Prototype mask generation & box regression |

## Architecture Evolution Plan

1. **A0 (Baseline)**: Official `yolo26-seg.yaml` at scale `s`.
2. **A1 (CARAFE)**:
   - Layer 11: `[-1, 1, CARAFE, [2, 5, 3, 64]]`
   - Layer 14: `[-1, 1, CARAFE, [2, 5, 3, 64]]`
3. **A2 (ASPP)**:
   - High-level context block placed after Layer 10 (P5 / Stride 32).
4. **A3 (CARAFE + ASPP)**:
   - Both A1 (CARAFE at layers 11, 14) and A2 (ASPP at P5) combined.
5. **A4 (DeepLabV3+ Decoder)**:
   - Staged CARAFE upsampling + Layer 1 (P2 / Stride 4) low-level projection + refinement into prototype masks.
