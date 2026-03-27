#!/usr/bin/env python3
"""
scripts/fetch_terrain.py — Download terrain data to data/terrain/.

Sources:
  srtm        — NASA SRTM 30m via USGS/ESA mirror
  copernicus  — Copernicus DEM 25m (GLO-30)
  osm-buildings — OpenStreetMap building footprints via Overpass API
  corine      — Corine Land Cover 2018

Usage:
  python scripts/fetch_terrain.py --source srtm --bbox 4.0 43.0 6.5 44.5
  python scripts/fetch_terrain.py --source osm-buildings --bbox 4.5 43.4 5.2 43.8
  python scripts/fetch_terrain.py --source all --bbox 4.0 43.0 6.5 44.5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

TERRAIN_DIR = ROOT / "data" / "terrain"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download terrain data for lunchfork")
    p.add_argument(
        "--source",
        choices=["srtm", "copernicus", "osm-buildings", "corine", "all"],
        required=True,
    )
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        required=True,
        help="Bounding box: lon_min lat_min lon_max lat_max",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=TERRAIN_DIR,
        help="Output cache directory",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# SRTM
# ---------------------------------------------------------------------------


def _srtm_tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}.hgt"


def fetch_srtm(lon_min: float, lat_min: float, lon_max: float, lat_max: float, cache_dir: Path) -> None:
    """Download SRTM 1-arc-second tiles covering the bbox."""
    out_dir = cache_dir / "srtm"
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_min_i = int(math.floor(lat_min))
    lat_max_i = int(math.floor(lat_max))
    lon_min_i = int(math.floor(lon_min))
    lon_max_i = int(math.floor(lon_max))

    index = {}
    index_path = out_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())

    downloaded = 0
    skipped = 0

    for lat in range(lat_min_i, lat_max_i + 1):
        for lon in range(lon_min_i, lon_max_i + 1):
            tile_name = _srtm_tile_name(lat, lon)
            tile_path = out_dir / tile_name
            if tile_path.exists():
                print(f"  [cache] {tile_name}")
                skipped += 1
                continue

            # ESA STEP SRTMGL1 mirror
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            zip_name = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}.SRTMGL1.hgt.zip"
            url = f"https://step.esa.int/auxdata/dem/SRTMGL1/{zip_name}"

            print(f"  [dl] {tile_name} from {url}")
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    data = resp.read()
                with zipfile.ZipFile(BytesIO(data)) as zf:
                    hgt_names = [n for n in zf.namelist() if n.endswith(".hgt")]
                    if hgt_names:
                        tile_path.write_bytes(zf.read(hgt_names[0]))
                        print(f"  [ok] {tile_name} ({len(data) // 1024} KB)")
                        downloaded += 1
                        index[tile_name] = {"lat": lat, "lon": lon, "source": "srtm-gl1"}
            except Exception as exc:
                print(f"  [warn] {tile_name}: {exc}")
                # Try CGIAR mirror (3-arc-second tiles, different naming)
                # Tile number: col = (lon + 180) / 5 + 1, row = (64 - lat) / 5
                col = int((lon + 180) / 5) + 1
                row = int((64 - lat) / 5)
                cgiar_url = (
                    f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/"
                    f"srtm_{col:02d}_{row:02d}.zip"
                )
                print(f"  [try] CGIAR fallback {cgiar_url}")
                try:
                    with urllib.request.urlopen(cgiar_url, timeout=60) as resp:
                        data = resp.read()
                    with zipfile.ZipFile(BytesIO(data)) as zf:
                        tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
                        if tif_names:
                            tif_data = zf.read(tif_names[0])
                            tif_path = out_dir / tif_names[0]
                            tif_path.write_bytes(tif_data)
                            print(f"  [ok] CGIAR {tif_names[0]}")
                            downloaded += 1
                except Exception as exc2:
                    print(f"  [error] CGIAR also failed: {exc2}")

    index_path.write_text(json.dumps(index, indent=2))
    print(f"\nSRTM: {downloaded} downloaded, {skipped} cached. Index: {index_path}")


# ---------------------------------------------------------------------------
# Copernicus DEM
# ---------------------------------------------------------------------------


def fetch_copernicus(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float, cache_dir: Path
) -> None:
    """
    Download Copernicus GLO-30 DEM tiles (25m, 1-arc-second).

    Tiles available via AWS Open Data (no registration required):
    s3://copernicus-dem-30m/Copernicus_DSM_COG_10_{lat}_{lon}_DEM/
    Accessed via HTTPS.
    """
    out_dir = cache_dir / "copernicus"
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_min_i = int(math.floor(lat_min))
    lat_max_i = int(math.floor(lat_max))
    lon_min_i = int(math.floor(lon_min))
    lon_max_i = int(math.floor(lon_max))

    index = {}
    index_path = out_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())

    downloaded = 0
    skipped = 0

    for lat in range(lat_min_i, lat_max_i + 1):
        for lon in range(lon_min_i, lon_max_i + 1):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "E"  # Copernicus uses E for all
            lat_str = f"{ns}{abs(lat):02d}"
            lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            tile_name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM.tif"
            tile_path = out_dir / tile_name

            if tile_path.exists():
                print(f"  [cache] {tile_name}")
                skipped += 1
                continue

            # AWS open data
            base = "https://copernicus-dem-30m.s3.amazonaws.com"
            dir_name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
            url = f"{base}/{dir_name}/{tile_name}"

            print(f"  [dl] Copernicus {tile_name}")
            try:
                with urllib.request.urlopen(url, timeout=120) as resp:
                    data = resp.read()
                tile_path.write_bytes(data)
                print(f"  [ok] {tile_name} ({len(data) // 1024} KB)")
                downloaded += 1
                index[tile_name] = {"lat": lat, "lon": lon, "source": "copernicus-glo30"}
            except Exception as exc:
                print(f"  [warn] {tile_name}: {exc}")

    index_path.write_text(json.dumps(index, indent=2))
    print(f"\nCopernicus: {downloaded} downloaded, {skipped} cached.")


# ---------------------------------------------------------------------------
# OSM buildings via Overpass API
# ---------------------------------------------------------------------------


def fetch_osm_buildings(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float, cache_dir: Path
) -> None:
    """Fetch building footprints from OSM Overpass API."""
    out_dir = cache_dir / "osm"
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox_key = f"{lat_min:.4f}_{lon_min:.4f}_{lat_max:.4f}_{lon_max:.4f}"
    out_path = out_dir / f"buildings_{bbox_key}.geojson"

    if out_path.exists():
        print(f"  [cache] OSM buildings {out_path.name}")
        return

    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
[out:json][timeout:60];
(
  way["building"]({lat_min},{lon_min},{lat_max},{lon_max});
  relation["building"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out body;
>;
out skel qt;
"""

    print(f"  [dl] OSM buildings bbox={lat_min},{lon_min},{lat_max},{lon_max}")
    try:
        data = query.encode("utf-8")
        req = urllib.request.Request(
            overpass_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            result = resp.read()
        out_path.write_bytes(result)
        size_kb = len(result) // 1024
        print(f"  [ok] OSM buildings {out_path.name} ({size_kb} KB)")
    except Exception as exc:
        print(f"  [error] OSM buildings: {exc}")


# ---------------------------------------------------------------------------
# Corine Land Cover
# ---------------------------------------------------------------------------


def fetch_corine(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float, cache_dir: Path
) -> None:
    """
    Corine Land Cover 2018 raster data.

    Full dataset available from EEA (European Environment Agency).
    This downloads the pan-European 100m raster via WCS endpoint.
    """
    out_dir = cache_dir / "corine"
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox_key = f"{lat_min:.3f}_{lon_min:.3f}_{lat_max:.3f}_{lon_max:.3f}"
    out_path = out_dir / f"clc2018_{bbox_key}.tif"

    if out_path.exists():
        print(f"  [cache] Corine {out_path.name}")
        return

    # EEA WCS endpoint for CLC 2018 (100m)
    # EPSG:3035 (ETRS89 / LAEA Europe) bounding box needed
    # Approximate conversion for Western Europe
    x_min = int((lon_min + 180) * 111320 * 0.75)  # rough
    y_min = int((lat_min + 90) * 111320)
    x_max = int((lon_max + 180) * 111320 * 0.75)
    y_max = int((lat_max + 90) * 111320)

    wcs_url = (
        "https://image.discomap.eea.europa.eu/arcgis/services/Corine/CLC2018_WM/MapServer/WCSServer"
        f"?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
        f"&COVERAGE=1&CRS=EPSG:4326&BBOX={lon_min},{lat_min},{lon_max},{lat_max}"
        f"&WIDTH=1000&HEIGHT=1000&FORMAT=GeoTIFF"
    )

    print(f"  [dl] Corine 2018 {out_path.name}")
    try:
        with urllib.request.urlopen(wcs_url, timeout=120) as resp:
            data = resp.read()
        out_path.write_bytes(data)
        print(f"  [ok] Corine {out_path.name} ({len(data) // 1024} KB)")
    except Exception as exc:
        print(f"  [warn] Corine WCS: {exc}")
        print("  Tip: Download manually from https://land.copernicus.eu/pan-european/corine-land-cover")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    lon_min, lat_min, lon_max, lat_max = args.bbox
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Terrain fetch: source={args.source} bbox=[{lon_min},{lat_min},{lon_max},{lat_max}]")
    print(f"Cache dir: {cache_dir}\n")

    sources = [args.source] if args.source != "all" else ["srtm", "copernicus", "osm-buildings", "corine"]

    for source in sources:
        print(f"=== {source.upper()} ===")
        if source == "srtm":
            fetch_srtm(lon_min, lat_min, lon_max, lat_max, cache_dir)
        elif source == "copernicus":
            fetch_copernicus(lon_min, lat_min, lon_max, lat_max, cache_dir)
        elif source == "osm-buildings":
            fetch_osm_buildings(lon_min, lat_min, lon_max, lat_max, cache_dir)
        elif source == "corine":
            fetch_corine(lon_min, lat_min, lon_max, lat_max, cache_dir)
        print()


if __name__ == "__main__":
    main()
