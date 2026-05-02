# Satellite Segmentation: Model Performance Analysis Report

This report provides a comparative analysis of five deep learning architectures (UNet, UNet++, LinkNet, DeepLabV3+, and MAnet) evaluated on 20 diverse satellite image patches.

## 1. Latency Analysis (Inference Speed)

The latency was measured on a CUDA-enabled device, averaged over 10 runs per patch.

| Model | Avg Latency (ms) | Speed Rank | Notes |
| :--- | :--- | :--- | :--- |
| **LinkNet** | **5.73 ms** | 1st | **Fastest.** Highly optimized for real-time applications. |
| **DeepLabV3+** | 5.97 ms | 2nd | Excellent balance of speed and complexity. |
| **UNet** | 7.27 ms | 3rd | Standard performance for a basic encoder-decoder. |
| **MAnet** | 8.23 ms | 4th | Slightly slower due to multi-scale attention. |
| **UNet++** | **16.73 ms** | 5th | **Slowest.** Dense skip connections increase compute cost. |

### 📍 Detailed Per-Patch Latency Breakdown (ms)

| Patch | DeepLabV3+ | LinkNet | MAnet | UNet | UNet++ | **Fastest** |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **patch_0** | 6.92 | 5.64 | 7.63 | 8.16 | 15.70 | **LinkNet** |
| **patch_10** | 5.77 | 5.61 | 7.80 | 7.10 | 17.72 | **LinkNet** |
| **patch_11** | 5.77 | 6.27 | 7.70 | 7.11 | 16.87 | **DeepLabV3+** |
| **patch_15** | 7.20 | 5.88 | 8.01 | 7.16 | 17.00 | **LinkNet** |
| **patch_170** | 5.54 | 5.43 | 8.17 | 7.39 | 16.23 | **LinkNet** |
| **patch_177** | 5.90 | 5.56 | 9.05 | 7.20 | 17.03 | **LinkNet** |
| **patch_232** | 5.85 | 6.87 | 9.69 | 8.01 | 17.21 | **DeepLabV3+** |
| **patch_234** | 5.64 | 5.42 | 8.33 | 7.00 | 16.13 | **LinkNet** |
| **patch_261** | 6.25 | 5.67 | 8.08 | 7.52 | 17.10 | **LinkNet** |
| **patch_263** | 5.72 | 5.50 | 7.77 | 7.30 | 16.87 | **LinkNet** |
| **patch_264** | 6.10 | 5.73 | 9.95 | 7.37 | 17.30 | **LinkNet** |
| **patch_290** | 5.63 | 5.52 | 8.72 | 7.12 | 17.08 | **LinkNet** |
| **patch_291** | 6.91 | 5.47 | 7.82 | 7.08 | 16.28 | **LinkNet** |
| **patch_34** | 5.86 | 5.69 | 7.59 | 7.06 | 15.67 | **LinkNet** |
| **patch_459** | 5.41 | 5.55 | 8.54 | 7.08 | 16.19 | **DeepLabV3+** |
| **patch_651** | 5.94 | 5.26 | 7.94 | 7.20 | 16.15 | **LinkNet** |
| **patch_652** | 5.60 | 5.77 | 7.76 | 7.01 | 17.60 | **DeepLabV3+** |
| **patch_679** | 6.00 | 5.81 | 8.24 | 7.22 | 17.48 | **LinkNet** |
| **patch_736** | 5.64 | 5.95 | 8.07 | 7.11 | 16.58 | **DeepLabV3+** |
| **patch_794** | 5.79 | 6.09 | 7.76 | 7.20 | 16.46 | **DeepLabV3+** |

> [!TIP]
> **LinkNet** is nearly **3x faster** than UNet++, making it the ideal choice for large-scale satellite processing pipelines.

---

## 2. Segmentation Coverage Analysis (Sensitivity)

| Model | Avg Coverage % | Detection Style |
| :--- | :--- | :--- |
| **LinkNet** | **21.13%** | **Most Sensitive.** Tends to identify more built-up pixels. |
| **MAnet** | 19.37% | Balanced detection. |
| **DeepLabV3+** | 15.24% | Robust, filters out minor noise. |
| **UNet** | 14.48% | Standard baseline detection. |
| **UNet++** | **12.76%** | **Most Conservative.** High precision, but may miss small structures. |

---
*Report generated on 2026-04-26 based on aggregated benchmark data from 20 diverse scenarios.*
