"""
Quick inspection of Romania terrain files.
Run this BEFORE the main risk projection script.
Place this file in your ~/Medes extension/ folder and run:
    python inspect_romania_data.py
"""

import json
import os

# ── Check which files exist ────────────────────────────────────────────────
FILES = [
    "rivers-romania.geojson",
    "contours-romania.geojson",
    "elevation-30m-romania.tif",
]

print("=" * 55)
print("FILE CHECK")
print("=" * 55)
for f in FILES:
    exists = os.path.exists(f)
    size   = f"{os.path.getsize(f) / 1024 / 1024:.1f} MB" if exists else "—"
    print(f"  {'OK' if exists else 'MISSING'}  {f}  ({size})")

print()

# ── Inspect rivers GeoJSON ─────────────────────────────────────────────────
rivers_file   = "rivers-romania.geojson"
if os.path.exists(rivers_file):
    print("=" * 55)
    print("RIVERS FILE")
    print("=" * 55)
    with open(rivers_file, encoding="utf-8") as f:
        rivers = json.load(f)

    features = rivers.get("features", [])
    print(f"Total features : {len(features)}")

    geom_types = set(f["geometry"]["type"] for f in features if f.get("geometry"))
    print(f"Geometry types : {geom_types}")

    # Show first 3 features
    for i, feat in enumerate(features[:3]):
        print(f"\n  Feature {i}:")
        print(f"    Geometry : {feat['geometry']['type']}")
        print(f"    Properties : {feat.get('properties', {})}")
        coords = feat["geometry"]["coordinates"]
        if feat["geometry"]["type"] == "LineString":
            print(f"    Points     : {len(coords)}")
            print(f"    First coord: {coords[0]}")
            print(f"    Last coord : {coords[-1]}")
        elif feat["geometry"]["type"] == "MultiLineString":
            print(f"    Lines      : {len(coords)}")
            print(f"    First point: {coords[0][0]}")

    # Bounding box
    all_coords = []
    for feat in features:
        if feat["geometry"]["type"] == "LineString":
            all_coords.extend(feat["geometry"]["coordinates"])
        elif feat["geometry"]["type"] == "MultiLineString":
            for line in feat["geometry"]["coordinates"]:
                all_coords.extend(line)
    if all_coords:
        lons = [c[0] for c in all_coords]
        lats = [c[1] for c in all_coords]
        print(f"\n  Bounding box:")
        print(f"    Lon: {min(lons):.3f} to {max(lons):.3f}")
        print(f"    Lat: {min(lats):.3f} to {max(lats):.3f}")

# ── Inspect contours GeoJSON ───────────────────────────────────────────────
contours_file = "contours-romania.geojson"
if os.path.exists(contours_file):
    print()
    print("=" * 55)
    print("CONTOURS FILE")
    print("=" * 55)
    with open(contours_file, encoding="utf-8") as f:
        contours = json.load(f)

    features = contours.get("features", [])
    print(f"Total features : {len(features)}")

    geom_types = set(f["geometry"]["type"] for f in features if f.get("geometry"))
    print(f"Geometry types : {geom_types}")

    # Show property keys and sample values
    if features:
        props = features[0].get("properties", {})
        print(f"Property keys  : {list(props.keys())}")
        print(f"\n  First 3 features:")
        for i, feat in enumerate(features[:3]):
            print(f"    Feature {i}: {feat.get('properties', {})}")

    # Check elevation field name
    all_props = [f.get("properties", {}) for f in features[:20]]
    elev_keys = set()
    for p in all_props:
        for k in p.keys():
            if any(x in k.lower() for x in ["elev", "height", "alt", "z", "level", "ele"]):
                elev_keys.add(k)
    print(f"\n  Likely elevation field(s): {elev_keys if elev_keys else 'not detected — check properties above'}")

# ── Inspect DEM raster ─────────────────────────────────────────────────────
dem_file = "elevation-30m-romania.tif"
if os.path.exists(dem_file):
    print()
    print("=" * 55)
    print("DEM RASTER FILE")
    print("=" * 55)
    try:
        import rasterio
        with rasterio.open(dem_file) as src:
            print(f"CRS         : {src.crs}")
            print(f"Dimensions  : {src.width} x {src.height} pixels")
            print(f"Bands       : {src.count}")
            print(f"Resolution  : {src.res}")
            print(f"Bounds      : {src.bounds}")
            print(f"NoData val  : {src.nodata}")
            import numpy as np
            data = src.read(1)
            valid = data[data != src.nodata] if src.nodata else data.flatten()
            print(f"Elev range  : {valid.min():.1f} m to {valid.max():.1f} m")
            print(f"Mean elev   : {valid.mean():.1f} m")
    except ImportError:
        print("rasterio not installed yet — install with:")
        print("  pip install rasterio")
        print("Then run this script again to see DEM details.")

print()
print("=" * 55)
print("Inspection complete. Share this output so the")
print("risk projection script can be built correctly.")
print("=" * 55)