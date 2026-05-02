import rasterio
import os
import glob
import numpy as np

# ==========================================
# PATH (UPDATE THIS)
# ==========================================

folder = r"D:\Research Project\Data"
output_path = r"D:\Research Project\Output\Image 5\image.tif"

# ==========================================
# FIND BAND FILES
# ==========================================

def get_band(folder, name):
    files = glob.glob(os.path.join(folder, "*.tif"))
    for f in files:
        if name.lower() in os.path.basename(f).lower():
            return f
    raise FileNotFoundError(f"{name} not found")

b2_path = get_band(folder, "band2")
b3_path = get_band(folder, "band3")
b4_path = get_band(folder, "band4")


print("\n📂 Files Found:")
print("BAND2:", b2_path)
print("BAND3:", b3_path)
print("BAND4:", b4_path)


# ==========================================
# LOAD BANDS
# ==========================================

b2 = rasterio.open(b2_path)
b3 = rasterio.open(b3_path)
b4 = rasterio.open(b4_path)


band2 = b2.read(1)
band3 = b3.read(1)
band4 = b4.read(1)


# ==========================================
# CHECK PROPERTIES
# ==========================================

print("\n🔍 CHECKING PROPERTIES")

# Shape
shapes = [band2.shape, band3.shape, band4.shape]
print("Shapes:", shapes)

if shapes[0] == shapes[1] == shapes[2] :
    print("✅ Shape MATCH")
else:
    print("❌ Shape MISMATCH")

# CRS
crs_list = [b2.crs, b3.crs, b4.crs ]
print("\nCRS:", crs_list)

if crs_list[0] == crs_list[1] == crs_list[2] :
    print("✅ CRS MATCH")
else:
    print("❌ CRS MISMATCH")

# Resolution
res_list = [b2.res, b3.res, b4.res ]
print("\nResolution:", res_list)

if res_list[0] == res_list[1] == res_list[2] :
    print("✅ Resolution MATCH")
else:
    print("❌ Resolution MISMATCH")

# ==========================================
# STACK BANDS (IMPORTANT ORDER)
# ==========================================

print("\n📦 Stacking bands...")

# Order: [NIR, Red, Green]
stacked = np.stack([band4, band3, band2 ])

# Metadata
meta = b2.meta.copy()
meta.update({"count": 3})

# Save
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with rasterio.open(output_path, "w", **meta) as dst:
    dst.write(stacked)

print("\n🎉 Stacked image saved at:", output_path)