# 🚀 Satellite Building Detection: Inference Benchmark Report

This report compares the performance of five deep learning architectures on the unseen imagery of **Image 5** (934 patches total).

## 📊 Quantitative Comparison

| Model | Avg Built-Up % | Max Built-Up % | Detections Count (Patches >1%)|
| :---  | :--- | :--- | :--- |
| **LINKNET** | 2.82 | 65.37 | **332** |
| **MANET** | 2.00 | 62.91 | 128 |
| **UNET** | 1.70 | 63.80 | 163 |
| **UNETPLUSPLUS** | 1.48 | 56.22 | 105 |
| **DEEPLABV3PLUS**| 1.45 | 57.69 | 94 |

---

## 🔍 Key Findings

### 1. Most Sensitive: **LinkNet**
LinkNet identified built-up areas in 332 patches—over 3x more than standard UNet++. This suggests LinkNet is excellent at picking up smaller or low-contrast buildings that other models might miss. However, it should be visually checked for "noise" or false positives.

### 2. Most Reliable: **UNet**
UNet (ResNet34) showed the most balanced performance. It consistent with the training validation scores and shows high confidence on clear urban structures (Max 63.8%). It is recommended as the "Baseline" for this project.

### 3. Most Conservative: **DeepLabV3+**
With only 94 detections, DeepLabV3+ is very selective. It likely only labels areas as "built-up" when the evidence is unmistakable.

---

## 📂 Where to find individual results?
Inference images (Original vs Prediction) are located at:
- `D:\Research Project\Inference_Results_Image5\linknet`
- `D:\Research Project\Inference_Results_Image5\unet`
- `D:\Research Project\Inference_Results_Image5\manet`
- `D:\Research Project\Inference_Results_Image5\unetplusplus`
- `D:\Research Project\Inference_Results_Image5\deeplabv3plus`

---
*Report generated on: 2026-04-24*

**Navigation:**
- [Main README](file:///d:/Research%20Project/README.md)
- [Full Research Report](file:///d:/Research%20Project/Final_Research_Report.md)
- [Architecture Benchmark](file:///d:/Research%20Project/Final_Benchmark_Report.md)
