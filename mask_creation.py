import osmnx as ox
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import pyproj
from shapely.geometry import box
from shapely.ops import transform as shp_transform

# ==========================================
# INPUT
# ==========================================

raster_path = r"D:\Research Project\Output\Image 4\image.tif"
output_mask = r"D:\Research Project\Output\Mask 4\final_mask.tif"

# ==========================================
# LOAD RASTER
# ==========================================

with rasterio.open(raster_path) as src:
    bounds = src.bounds
    transform = src.transform
    shape = (src.height, src.width)
    crs = src.crs

print("📐 Raster loaded")
print("Shape:", shape)
print("CRS:", crs)

# ==========================================
# CONVERT TO LAT-LON (FOR OSM)
# ==========================================

project = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform

bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
bbox_latlon = shp_transform(project, bbox)

print("🌍 Converted to lat/lon")

# ==========================================
# DOWNLOAD OSM BUILDINGS
# ==========================================

print("\n⬇️ Downloading OSM buildings...")
buildings = ox.features_from_polygon(bbox_latlon, tags={"building": True})

# ==========================================
# HANDLE EMPTY CASE
# ==========================================

if buildings.empty:
    print("⚠️ No building data found!")
    mask = np.zeros(shape, dtype=np.uint8)

else:
    print("✅ Buildings found:", len(buildings))

    # ==========================================
    # CLEAN GEOMETRIES
    # ==========================================
    buildings = buildings[buildings.geometry.notnull()]
    buildings = buildings[buildings.is_valid]

    # Fix invalid polygons
    buildings["geometry"] = buildings.buffer(0)

    # ==========================================
    # REPROJECT TO IMAGE CRS
    # ==========================================
    buildings = buildings.to_crs(crs)

    # ==========================================
    # CLIP TO IMAGE BOUNDS (CRITICAL)
    # ==========================================
    raster_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    buildings = buildings.clip(raster_bbox)

    print("📍 Buildings after clipping:", len(buildings))

    # ==========================================
    # OPTIONAL: SMALL BUFFER (IMPROVES COVERAGE)
    # ==========================================
    buildings["geometry"] = buildings.buffer(0.5)  # adjust 0.3–1 if needed

    # ==========================================
    # CREATE SHAPES
    # ==========================================
    shapes = [(geom, 1) for geom in buildings.geometry if geom is not None]

    # ==========================================
    # RASTERIZE (KEY STEP)
    # ==========================================
    print("\n🧮 Rasterizing buildings...")

    mask = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,   # 🔥 important for full coverage
        dtype=np.uint8
    )

# ==========================================
# SAVE MASK
# ==========================================

meta = {
    "driver": "GTiff",
    "height": shape[0],
    "width": shape[1],
    "count": 1,
    "dtype": "uint8",
    "crs": crs,
    "transform": transform
}

with rasterio.open(output_mask, "w", **meta) as dst:
    dst.write(mask, 1)

print("\n🎉 BUILDING MASK GENERATED:", output_mask)

# ==========================================
# QUICK STATS
# ==========================================

total = mask.size
non_zero = np.count_nonzero(mask)

print("\n📊 MASK STATS")
print("Total pixels:", total)
print("Building pixels:", non_zero)
print("Percentage:", (non_zero / total) * 100)