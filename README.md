# 🛰️ Satellite Segmentation Benchmark Project

An end-to-end AI pipeline for high-precision building and road segmentation from 16-bit satellite imagery. This project features a comparative benchmark of five state-of-the-art architectures and an optimized data preprocessing pipeline.

## 🚀 Project Overview

The project follows a streamlined 6-step workflow:

1.  **Band Stacking** (`stack_bands.py`): Combines 16-bit spectral bands (B2, B3, B4).
2.  **Mask Creation** (`mask_creation.py`): Generates binary ground truth from source data.
3.  **Tiling** (`create_patches.py`): Slices large TIFFs into 256x256 patches.
4.  **Dataset Management** (`manage_dataset.py`): Organizes patches and partitions them into Train/Val/Test splits.
5. **Training** (`train_model_v1.py` / `train_model_v2.py`): Trains any of the 5 supported architectures.
6. **Visual Evaluation** (`visualize_model_v1.py` / `visualize_model_v2.py`): Generates mapping overlays.
7. **Unseen Inference** (`inference_batch.py`): Runs the model on entirely new satellite imagery (Phase 3).

---

## 📊 Benchmark Results (Phase 2 Optimized)

| Architecture   |   IoU      | F1 Score   | Precision  | Recall     | Accuracy   |
| :-----------:  | :-------:  | :-------:  | :------:   | :------:   | :------:   |
| **UNet**       | **0.3736** | **0.5440** | 0.4664     | 0.6526     | 0.8587     |
| **UNet++**     | 0.3725     | 0.5428     | **0.4777** | 0.6285     | **0.8633** |
| **MAnet**      | 0.3616     | 0.5312     | 0.4376     | 0.6756     | 0.8460     |
| **DeepLabV3+** | 0.3505     | 0.5191     | 0.4507     | 0.6119     | 0.8536     |
| **LinkNet**    | 0.3457     | 0.5138     | 0.4051     | **0.7024** | 0.8284     |

---

## 🔍 Unseen Data Inference (Phase 3: Image 5)

| Model | Detections (Patches >1%) | Avg Built-Up % | Characteristic |
| :---  | :--- | :--- | :--- |
| **LinkNet** | **332** | 2.82% | Highest Sensitivity |
| **UNet** | 163 | 1.70% | Most Balanced |
| **MAnet** | 128 | 2.00% | Robust |
| **DeepLabV3+** | 94 | 1.45% | Conservative |

---

## 📁 Repository Structure

```
.
├── stack_bands.py         # Step 1: Combine spectral bands
├── mask_creation.py       # Step 2: Generate binary ground truth
├── create_patches.py      # Step 3: Image tiling
├── manage_dataset.py      # Step 4: Organization & Partitioning
├── train_model_v1.py      # Step 5: Unified Baseline Training
├── train_model_v2.py      # Step 5: Unified Optimized Training
├── visualize_model_v1.py  # Step 6: Single-model Visuals (V1)
├── visualize_model_v2.py  # Step 6: Single-model Visuals (V2)
├── inference_batch.py     # Step 7: Batch Inference on unseen data
├── compare_models.py      # Multi-model Comparison Grid
├── inspect_data.py        # Centralized Diagnostic Utility
├── dataset_split/         # Final partitioned dataset 
├── Inference_Results_Image5/ # Raw inference outputs
└── results_v2/            # Metrics and checkpoints
```

---

## 🛠️ Key Utilities

### 🧪 `inspect_data.py`
A centralized diagnostic tool to verify data at any stage:
- Checks band alignment and CRS metadata.
- Validates mask binary values and coverage.
- Verifies GPU availability and dataset consistency.

### 🍱 `manage_dataset.py`
A one-stop script to organize your raw patches into a structured `dataset_split` folder, handle naming conventions, and perform stratified splitting.

---

## 📜 Reports
- [Detailed Research Report](file:///d:/Research%20Project/Final_Research_Report.md)
- [Architecture Benchmark Analysis](file:///d:/Research%20Project/Final_Benchmark_Report.md)
- [Unseen Inference Benchmark](file:///d:/Research%20Project/Model_Inference_Benchmark.md)
