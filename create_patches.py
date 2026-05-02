import os
import rasterio
import numpy as np
from tqdm import tqdm

# ==========================================
# INPUT
# ==========================================

image_path = r"D:\Research Project\Output\Image 5\image.tif"
# mask_path = r"D:\Research Project\Output\Mask 4\final_mask.tif"

output_img_dir = r"D:\Research Project\Dataset\Image 5\images"
# output_mask_dir = r"D:\Research Project\Dataset\Mask 4\masks"

os.makedirs(output_img_dir, exist_ok=True)
# os.makedirs(output_mask_dir, exist_ok=True)

# ==========================================
# PARAMETERS
# ==========================================

tile_size = 512
stride = 512

building_threshold = 0.01   # 1%
black_threshold = 0.90      # 90% nodata

# ==========================================
# LOAD DATA
# ==========================================

with rasterio.open(image_path) as src_img:
    image = src_img.read()
    nodata = src_img.nodata
    # Set H, W from image bands (Rasterio is C, H, W)
    _, H, W = image.shape

# with rasterio.open(mask_path) as src_mask:
#     mask = src_mask.read(1)

# H, W = mask.shape

print("📐 Image shape:", image.shape)
print("Nodata value:", nodata)

# ==========================================
# PATCH EXTRACTION
# ==========================================

count = 0

for y in tqdm(range(0, H - tile_size, stride)):
    for x in range(0, W - tile_size, stride):

        img_patch = image[:, y:y+tile_size, x:x+tile_size]
        # mask_patch = mask[y:y+tile_size, x:x+tile_size]

        # ==========================================
        # ✅ FIXED BLACK DETECTION
        # ==========================================

        if nodata is not None:
            black_pixels = np.sum(img_patch == nodata)
        else:
            # fallback: check all bands are zero
            black_pixels = np.sum(np.all(img_patch == 0, axis=0))

        total_pixels = tile_size * tile_size

        if black_pixels / total_pixels > black_threshold:
            continue

        # ==========================================
        # BUILDING FILTER (DISABLED FOR UNSEEN IMAGES)
        # ==========================================

        # building_pixels = np.count_nonzero(mask_patch)

        # if building_pixels / total_pixels < building_threshold:
        #     continue

        # ==========================================
        # SAVE PATCH
        # ==========================================

        img_name = f"patch_{count}.tif"
        mask_name = f"patch_{count}.tif"

        # Save image
        with rasterio.open(
            os.path.join(output_img_dir, img_name),
            'w',
            driver='GTiff',
            height=tile_size,
            width=tile_size,
            count=image.shape[0],
            dtype=image.dtype
        ) as dst:
            dst.write(img_patch)

        # Save mask
        # with rasterio.open(
        #     os.path.join(output_mask_dir, mask_name),
        #     'w',
        #     driver='GTiff',
        #     height=tile_size,
        #     width=tile_size,
        #     count=1,
        #     dtype=mask.dtype
        # ) as dst:
        #     dst.write(mask_patch, 1)

        count += 1

print("\n🎉 Total VALID patches saved:", count)