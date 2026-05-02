import os
import glob
import rasterio
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import random

# ======================
# 1. RAW BAND & TIFF INSPECTION
# ======================
def get_band_path(folder, band_name):
    all_files = glob.glob(os.path.join(folder, "*.tif"))
    for file in all_files:
        if band_name.lower() in os.path.basename(file).lower(): return file
    return None

def inspect_raw_bands(base_path, folders):
    print("\n--- 🔍 RAW BAND INSPECTION ---")
    for f in folders:
        folder_path = os.path.join(base_path, f)
        if not os.path.exists(folder_path): continue
        print(f"\n📂 Folder: {f}")
        try:
            b2 = get_band_path(folder_path, "band2")
            b3 = get_band_path(folder_path, "band3")
            b4 = get_band_path(folder_path, "band4")
            
            with rasterio.open(b2) as s2, rasterio.open(b3) as s3, rasterio.open(b4) as s4:
                shapes = [s2.shape, s3.shape, s4.shape]
                crs = [s2.crs, s3.crs, s4.crs]
                if shapes[0] == shapes[1] == shapes[2]: print(f"✅ Shape Match: {shapes[0]}")
                else: print(f"❌ Shape Mismatch: {shapes}")
                if crs[0] == crs[1] == crs[2]: print(f"✅ CRS Match: {crs[0]}")
                else: print(f"❌ CRS Mismatch: {crs}")
        except Exception as e: print(f"🚨 Error: {e}")

def print_image_stats(tif_path):
    print(f"\n--- 📄 STATS: {os.path.basename(tif_path)} ---")
    with rasterio.open(tif_path) as src:
        print(f"Dimensions: {src.width}x{src.height} | Bands: {src.count}")
        print(f"CRS: {src.crs}")
        data = src.read()
        print(f"Min/Max: {data.min()}/{data.max()} | NaN Count: {np.isnan(data).sum()}")

def visual_patch_check(image_path, mask_path):
    print("\n🖼️ Checking random patches...")
    with rasterio.open(mask_path) as m_src: mask = m_src.read(1)
    with rasterio.open(image_path) as i_src: img = i_src.read()
    
    h, w = mask.shape; size = 512
    row = random.randint(0, h - size); col = random.randint(0, w - size)
    window = rasterio.windows.Window(col, row, size, size)
    
    with rasterio.open(image_path) as i_src: i_patch = i_src.read([1,2,3], window=window)
    with rasterio.open(mask_path) as m_src: m_patch = m_src.read(1, window=window)
    
    print(f"Sample Patch unique values: {np.unique(m_patch)}")

# ======================
# 2. MASK VALIDATION
# ======================
def validate_mask(mask_path):
    print("\n--- 🎭 MASK VALIDATION ---")
    with rasterio.open(mask_path) as src:
        mask = src.read(1); unique = np.unique(mask)
        print(f"Unique values: {unique}")
        coverage = np.sum(mask > 0) / (src.width * src.height)
        print(f"Feature Coverage: {coverage:.2%}")

# ======================
# 3. PATCH & DATASET INTEGRITY
# ======================
def check_final_dataset(split_dir):
    print("\n--- 🚀 FINAL DATASET & SYSTEM CHECK ---")
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(split_dir, split, "images")
        mask_dir = os.path.join(split_dir, split, "masks")
        if os.path.exists(img_dir):
            imgs = sorted(os.listdir(img_dir))
            masks = sorted(os.listdir(mask_dir))
            print(f"Split [{split}]: {len(imgs)} imgs / {len(masks)} masks")
            if len(imgs) != len(masks): print("❌ COUNT MISMATCH!")
            
    print("\nGPU Check:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available(): print(f"Device: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    # Diagnostic Hub Initialization
    if os.path.exists("Data"):
        inspect_raw_bands("Data", ["Image 1", "Image 2", "Image 3", "Image 4", "Image 5"])
    
    if os.path.exists("dataset_split"):
        check_final_dataset("dataset_split")
    else:
        print("\n⚠️ dataset_split folder not found. Run manage_dataset.py first.")
