# Comprehensive Research and Performance Analysis Report
## Satellite-Based Road and Building Segmentation

---

## 1. Introduction
Rapid urbanization poses a significant challenge for maintaining accurate cartographic data. Manual digitization is time-consuming and error-prone. This project presents an automated AI system designed to segment buildings and roads from high-resolution 16-bit satellite imagery focusing on the New Delhi area.

The primary objective was to implement, benchmark, and compare five state-of-the-art deep learning architectures (UNet, UNet++, MAnet, DeepLabV3+, and LinkNet) to determine which model offers the best balance between segmentation accuracy and real-time inference efficiency.

---

## 2. Literature Review
To address the segmentation task, five distinct architectures were selected based on their proven performance in geospatial tasks:

-   **UNet**: A symmetric encoder-decoder with skip connections, serving as the industry baseline for medical and satellite imaging.
-   **UNet++**: An evolution of UNet that utilizes nested, dense skip pathways to bridge the semantic gap between encoder and decoder features.
-   **MAnet (Multi-scale Attention Net)**: Incorporates attention mechanisms to focus on multi-scale spatial and channel-wise features, reducing noise.
-   **DeepLabV3+**: Employs Atrous Spatial Pyramid Pooling (ASPP) to capture contextual information at multiple scales, making it robust for complex urban layouts.
-   **LinkNet**: Designed specifically for efficiency, it bypasses information directly from the encoder to the decoder to maintain low computational latency.

---

## 3. Methodology

### 3.1 Data Acquisition & Preprocessing
-   **Source**: Data was acquired via the Bhuvanidhi portal (ISRO).
-   **Slicing**: Large satellite tiles were processed into 3,798 valid 256x256 pixel patches.
-   **16-bit Normalization**: A custom scaling factor of **500.0** was applied to utilize the full dynamic range of the raw 16-bit imagery before converting to 8-bit for model input.
-   **Data Split**: 70% Training, 15% Validation, and 15% Testing.

### 3.2 Training Configuration
-   **Backbone**: ResNet34 (Pre-trained on ImageNet).
-   **Loss Function**: Hybrid **Dice + BCE Combo Loss** to handle class imbalance (buildings vs. background).
-   **Optimization**: 100 Epochs with a learning rate scheduler.

### 3.3 Latency Benchmarking Pipeline
A specialized script (`benchmark_patch.py`) was developed to measure real-world performance on a single-patch basis across all models. The script measures latency over 10 consecutive runs per patch on a CUDA-enabled device and generates side-by-side visual comparisons.

---

## 4. Results

### 4.1 Global Performance Metrics (Phase 2 Testing)
| Architecture | IoU | F1 Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **UNet** | **0.3736** | **0.5440** | 0.4664 | 0.6526 |
| **UNet++** | 0.3725 | 0.5428 | **0.4777** | 0.6285 |
| **MAnet** | 0.3616 | 0.5312 | 0.4376 | 0.6756 |
| **DeepLabV3+** | 0.3505 | 0.5191 | 0.4507 | 0.6119 |
| **LinkNet** | 0.3457 | 0.5138 | 0.4051 | **0.7024** |

### 4.2 Detailed Latency & Coverage Analysis
Based on the 20-patch benchmarking suite across diverse urban and rural scenarios:

| Model | Avg Latency (ms) | Avg Coverage % | Characterization |
| :--- | :---: | :---: | :--- |
| **LinkNet** | **5.73** | **21.13%** | High Sensitivity / Real-time King |
| **DeepLabV3+** | 5.97 | 15.24% | Robust Context / Balanced |
| **UNet** | 7.27 | 14.48% | Reliable Baseline |
| **MAnet** | 8.23 | 19.37% | Attention-Focused |
| **UNet++** | **16.73** | 12.76% | High Precision / Computationally Heavy |

---

## 5. Conclusions
The project successfully implemented an end-to-end pipeline for satellite segmentation. 

1.  **Segmentation Performance**: **UNet** and **UNet++** provide the most balanced IoU and F1 scores, making them suitable for static mapping tasks.
2.  **Operational Efficiency**: **LinkNet** is the undisputed leader for high-throughput applications, offering nearly **3x the speed** of UNet++ while maintaining the highest recall for small structures.
3.  **Sensitivity**: LinkNet tends to be more inclusive (high sensitivity), whereas UNet++ is more conservative (high precision).

---

## 6. Future Work
-   **GIS Integration**: Developing a module to convert binary masks directly into GeoJSON and Shapefiles for use in ArcGIS/QGIS.
-   **Multi-Class Expansion**: Simultaneous segmentation of Roads, Buildings, and Vegetation.
-   **Dynamic Tiling**: Implementing a sliding-window inference pipeline for large, seamless satellite tiles to avoid edge artifacts.

---
*Generated: 2026-04-26*
