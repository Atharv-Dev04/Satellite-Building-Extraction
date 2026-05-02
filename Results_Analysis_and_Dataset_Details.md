# Project Results and Dataset Analysis

This document provides a detailed breakdown of the dataset used in this research and a comparative analysis of the model evolution from V1 (Baseline) to V2 (Optimized).

---

## 1. Dataset Details

The dataset focuses on the **New Delhi Area of Interest (AOI)**, specifically targeting urban and semi-urban landscapes for road and building segmentation.

### 1.1 Data Source and Acquisition
- **Provider:** ISRO Bhuvan / Bhuvanidhi Portal.
- **Imagery Type:** Multi-spectral satellite imagery.
- **Bit Depth:** 16-bit (allowing for high dynamic range geospatial data).
- **Format:** GeoTIFF (.tif).

### 1.2 Ground Truth Generation
Unlike many datasets that rely on manual labeling, this project utilized automated geospatial cross-referencing:
- **Vector Source:** OpenStreetMap (OSM) Building Footprints.
- **Rasterization:** Vector geometries were downloaded using `osmnx` and rasterized to match the satellite coordinate reference system (CRS) using `rasterio`.
- **Refinement:** Applied a small spatial buffer (0.5m) to ensure buildings were fully captured within the raster masks.

### 1.3 Data Preprocessing & Pipeline
To make the high-resolution satellite tiles compatible with Deep Learning models, the following pipeline was implemented:
- **Patching:** Large tiles were sliced into fixed **256x256 pixel patches**.
- **Filtering:** To reduce noise and focus on "information-rich" tiles, any patch with a building mask area of less than **500 pixels** was excluded from the training set.
- **Scaling:** 
  - **V1:** Normalized using a generic factor of 1000.
  - **V2:** Optimized to a factor of **500.0**, which better captures the spectral intensity of the target AOI in 16-bit.

### 1.4 Dataset Statistics
| Attribute | Value |
| :--- | :--- |
| **Total Valid Patches** | 3,798 |
| **Training Set (70%)** | 2,658 patches |
| **Validation Set (15%)** | 570 patches |
| **Test Set (15%)** | 570 patches |
| **Input Channels** | 3 (RGB) |
| **Output Classes** | 1 (Binary Segmentation) |

---

## 2. Evolution: V1 (Baseline) vs. V2 (Optimized)

The transition from Version 1 to Version 2 involved significant hyperparameter tuning and loss function optimization to resolve initial underfitting.

### 2.1 Comparative Training Configuration
| Feature | V1 (Baseline) | V2 (Optimized) | Impact |
| :--- | :--- | :--- | :--- |
| **Epochs** | 50 | 100 | Allowed deeper convergence. |
| **Scaling Factor** | 1000.0 | **500.0** | Significant boost in feature visibility. |
| **Loss Function** | Dice + Focal + BCE | **Dice + BCE (50/50)** | More stable gradient descent for building edges. |
| **Early Stopping** | 7 Epochs | 15 Epochs | Prevented premature termination. |
| **Learning Rate** | 1e-4 | 1e-4 | Maintained for stability. |

### 2.2 Performance Improvement (UNet Example)
| Metric | V1 Results | V2 Results | Improvement |
| :--- | :--- | :--- | :--- |
| **Test IoU** | 0.3515 | **0.3736** | **+6.3%** |
| **Test F1 Score** | 0.5202 | **0.5440** | **+4.6%** |

---

## 3. Final Results Summary (V2 Optimized)

The V2 models show consistent performance across all evaluated architectures. UNet and UNet++ lead in raw accuracy, while LinkNet shows superior sensitivity in detection tasks.

### 3.1 Model Benchmarking (Test Set)
| Architecture | IoU | F1 Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **UNet** | **0.3736** | **0.5440** | 0.4664 | 0.6526 |
| **UNet++** | 0.3725 | 0.5428 | **0.4777** | 0.6285 |
| **MAnet** | 0.3616 | 0.5312 | 0.4376 | 0.6756 |
| **DeepLabV3+** | 0.3505 | 0.5191 | 0.4507 | 0.6119 |
| **LinkNet** | 0.3457 | 0.5138 | 0.4051 | **0.7024** |

### 3.2 Inference Efficiency (Latency)
Benchmarked on an RTX 3050 GPU (256x256 patch size):
| Architecture | Avg Latency (ms) | Speed Rank |
| :--- | :---: | :---: |
| **LinkNet** | **5.74** | 1 |
| **DeepLabV3+** | 5.92 | 2 |
| **UNet** | 7.28 | 3 |
| **MAnet** | 8.24 | 4 |
| **UNet++** | 16.65 | 5 |

### 3.3 Generalization Performance (Unseen Data: Image 5)
When tested on entirely new imagery, the models demonstrated robust generalization:
- **LinkNet** emerged as the most sensitive, detecting **332** building clusters.
- **UNet** provided the most balanced masks with an average built-up area of **1.70%**.

---

## 4. Conclusion of Results
The optimization phase (V2) successfully addressed the dynamic range issues of 16-bit satellite imagery. By adjusting the scaling factor and refining the loss function composition, we achieved a more precise segmentation of urban structures, providing a reliable foundation for automated cartography.
