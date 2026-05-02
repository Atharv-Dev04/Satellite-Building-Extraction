# Semantic Segmentation of Roads and Buildings using Satellite Imagery
**Full Research Report**

## Abstract
This project presents an end-to-end AI system designed to automatically detect and segment roads and buildings from high-resolution satellite imagery. Utilizing state-of-the-art Deep Learning models (UNet, UNet++, MAnet, DeepLabV3+, and LinkNet), this project provides accurate geographical mappings to assist in urban planning, disaster recovery, and infrastructure monitoring. By optimizing the data scaling and loss functions, the project achieved significant improvements in IoU and F1-scores across all architectures.

---

## 1. Introduction
With the rapid pace of urbanization, maintaining up-to-date cartographic data is a monumental challenge. Traditional manual digitization of roads and buildings from satellite imagery is time-consuming and prone to human error. Automation using artificial intelligence provides a highly scalable and robust solution.

This research focuses on the New Delhi Area of Interest (AOI), leveraging the Bhuvanidhi portal for high-quality geospatial data. The primary objective is to implement, benchmark, and evaluate five semantic segmentation architectures to effectively distinguish road networks and building footprints from background geography.

---

## 2. Literature Review
Semantic segmentation in satellite imagery has advanced rapidly due to Convolutional Neural Networks (CNNs). 
- **UNet:** Standard symmetrical architecture with skip connections for detail recovery.
- **UNet++:** Uses nested and dense skip pathways to reduce the semantic gap between encoder and decoder feature maps.
- **MAnet:** Multi-scale Attention Net that uses special attention blocks to focus on relevant spatial and channel features.
- **DeepLabV3+:** Utilizes Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context.
- **LinkNet:** An efficient architecture designed to bypass information from encoder to decoder while maintaining low computational cost.

---

## 3. Methodology

### 3.1 Dataset Collection and Preprocessing
- **Source:** Imagery was acquired through the Bhuvan / Bhuvanidhi portal, focusing on New Delhi.
- **Patches Generation:** Large .tif files were sliced into 256x256 pixel patches, yielding a total of **3,798** valid patches.
- **Phased Optimization:** 
  - **Phase 2 (Optimized):** Implemented a fixed scaling factor of **500.0** (to utilize the full 16-bit dynamic range), **100 epochs**, and a specialized **Dice+BCE Combo loss**.
  - **Phase 3 (Deployment Inference):** Evaluated models on entirely unseen satellite imagery (**Image 5**) to test real-world generalization.

### 3.2 Dataset Splitting
- **Train:** 2,658 images (~70%)
- **Validation:** 570 images (~15%)
- **Test:** 570 images (~15%)

### 3.3 Experimental Setup
- **Hardware:** Training conducted locally on an RTX 3050 GPU.
- **Pipeline:** Developed a consolidated training architecture (`train_model_v1.py` and `train_model_v2.py`) to allow rapid benchmarking across architectures.

---

## 4. Results (V2 Optimized Phase)

The following table summarizes the performance of all five architectures using the Phase 2 settings (Test Set):

| Architecture   |   IoU      | F1 Score   | Precision  | Recall     | Accuracy   |
| :-----------:  | :-------:  | :-------:  | :------:   | :------:   | :------:   |
| **UNet**       | **0.3736** | **0.5440** | 0.4664     | 0.6526     | 0.8587     |
| **UNet++**     | 0.3725     | 0.5428     | **0.4777** | 0.6285     | **0.8633** |
| **MAnet**      | 0.3616     | 0.5312     | 0.4376     | 0.6756     | 0.8460     |
| **DeepLabV3+** | 0.3505     | 0.5191     | 0.4507     | 0.6119     | 0.8536     |
| **LinkNet**    | 0.3457     | 0.5138     | 0.4051     | **0.7024** | 0.8284     |

### 4.1 Comparative Visualization
A high-coverage comparison (59.1% building mask) was performed to visually validate model sensitivity:

![Model Comparison](file:///d:/Research%20Project/prediction_visualize.png)
*Figure 1: Comparative analysis of all 5 models on a building-dense archive tile.*

### 4.2 Phase 3: Generalization on Unseen Data (Image 5)
To validate the models' robustness, an inference pipeline was run on Image 5 (934 patches).

| Model | Detections (>1%) | Avg Built-Up % | Findings |
| :---  | :--- | :--- | :--- |
| **LinkNet** | **332** | 2.82 | Most sensitive to small building footprints. |
| **UNet** | 163 | 1.70 | Consistent and reliable performance. |
| **DeepLabV3+** | 94 | 1.45 | Most conservative; avoids false positives. |

The inference results (Visualized in `unseen_comparison_visualize.png`) confirm that while UNet is the strongest on the test set, LinkNet excels at detecting subtle built-up areas in new regions.

---

## 5. Conclusion & Future Work
The project successfully demonstrated that optimizing data scaling (utilizing the specific 0-500 range of the 16-bit tiles) and adopting a hybrid Dice+BCE loss significantly resolves underfitting. **UNet** and **UNet++** emerged as the top-performing models for building footprint detection in urban environments. Phase 3 testing further highlighted **LinkNet** as a powerful tool for high-sensitivity detection in unseen territories.

**Future Work:**
- Expanding the dataset to include varying geographies (rural, industrial).
- Implementing multi-class segmentation for simultaneous road and building detection.
- Exporting raw predictions to GeoJSON/Shapefiles for GIS integration.

---

## Appendix A: Day-wise Work Report

**Phase 1-3: Setup and Initial Prototyping**
- Finalized AOI and sourced 16-bit satellite datasets.
- Developed the geospatial preprocessing/slicing pipeline.

**Phase 4: Optimization and Deployment (Latest updates)**
- **Pipeline Consolidation:** Unified 10 architecture-specific scripts into 2 robust versions (`v1` and `v2`) with dynamic model selection.
- **Visual Analytics:** Developed `compare_models.py` to generate side-by-side grayscale comparisons on high-coverage tiles.
- **Inseen Inference:** Implemented `inference_batch.py` and benchmarked performance on Image 5, identifying LinkNet as the most sensitive architecture for new data.
- **Reporting:** Finalized the Benchmark and Research reports with validated Test and Inference metrics.
