import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ======================
# CONFIG
# ======================
dataset_root = "Dataset"
output_dir = "dataset_split"

# Split locations
split_paths = [
    os.path.join(output_dir, "train/images"), os.path.join(output_dir, "train/masks"),
    os.path.join(output_dir, "val/images"), os.path.join(output_dir, "val/masks"),
    os.path.join(output_dir, "test/images"), os.path.join(output_dir, "test/masks")
]

for path in split_paths: os.makedirs(path, exist_ok=True)

# ======================
# DATA DISCOVERY & SPLITTING
# ======================
def manage_dataset():
    all_pairs = []
    image_folders = [f for f in os.listdir(dataset_root) if f.startswith("Image")]

    for img_folder in image_folders:
        try:
            idx = img_folder.split(' ')[1]
            mask_folder = f"Mask {idx}"
        except: continue

        img_src = os.path.join(dataset_root, img_folder, "images")
        mask_src = os.path.join(dataset_root, mask_folder, "masks")

        if not os.path.exists(img_src): continue

        print(f"Scanning source: {img_folder}...")
        for patch in os.listdir(img_src):
            if patch.endswith(".tif") and os.path.exists(os.path.join(mask_src, patch)):
                all_pairs.append((os.path.join(img_src, patch), os.path.join(mask_src, patch), f"img{idx}_{patch}"))

    print(f"Total pairs found: {len(all_pairs)}")

    # Filter out empty masks (optional - but recommended for urban building detection)
    non_empty = []; empty = []
    for s_img, s_mask, name in all_pairs:
        m = cv2.imread(s_mask, 0)
        if m is not None and np.sum(m) > 0: non_empty.append((s_img, s_mask, name))
        else: empty.append((s_img, s_mask, name))

    print(f"Non-empty: {len(non_empty)} | Empty: {len(empty)}")

    # Splitting logic
    train_ne, temp_ne = train_test_split(non_empty, test_size=0.30, random_state=42)
    val_ne, test_ne = train_test_split(temp_ne, test_size=0.50, random_state=42)
    
    # Add some empty patches to test set for robustness check
    test_empty = empty[:min(len(empty), len(test_ne))]
    
    splits = {
        "train": train_ne,
        "val": val_ne,
        "test": test_ne + test_empty
    }

    # Copy files
    for split_name, data in splits.items():
        print(f"Populating {split_name} split...")
        img_dst = os.path.join(output_dir, split_name, "images")
        mask_dst = os.path.join(output_dir, split_name, "masks")
        for s_img, s_mask, name in data:
            shutil.copy(s_img, os.path.join(img_dst, name))
            shutil.copy(s_mask, os.path.join(mask_dst, name))

    print("\n✅ Dataset management & partitioning completed!")

if __name__ == "__main__":
    manage_dataset()
