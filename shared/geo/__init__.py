"""
shared.geo — Geospatial preprocessing for lunchfork.

Provides:
  - BBox: bounding box dataclass
  - GeoPreprocessor: conditioning tensor builder
  - fetch_srtm(): download SRTM terrain tiles from public sources

Phase 1: returns zero tensors (no terrain data required).
Phase 2: implements SRTM fetch + rasterisation.
"""

from __future__ import annotations

import io
import logging
import os
import struct
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BBox:
    """
    Geographic bounding box in WGS84 decimal degrees.

    Convention: (lat_min, lon_min, lat_max, lon_max)
    """

    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

    @property
    def center_lat(self) -> float:
        return (self.lat_min + self.lat_max) / 2.0

    @property
    def center_lon(self) -> float:
        return (self.lon_min + self.lon_max) / 2.0

    @property
    def width_deg(self) -> float:
        return self.lon_max - self.lon_min

    @property
    def height_deg(self) -> float:
        return self.lat_max - self.lat_min

    def as_dict(self) -> dict:
        return {
            "lat_min": self.lat_min,
            "lon_min": self.lon_min,
            "lat_max": self.lat_max,
            "lon_max": self.lon_max,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BBox":
        return cls(
            lat_min=d["lat_min"],
            lon_min=d["lon_min"],
            lat_max=d["lat_max"],
            lon_max=d["lon_max"],
        )

    @classmethod
    def from_center(cls, lat: float, lon: float, radius_m: float) -> "BBox":
        """Create a BBox centred at (lat, lon) with given radius in metres."""
        # ~111320 m per degree latitude; longitude varies with cos(lat)
        import math
        lat_delta = radius_m / 111320.0
        lon_delta = radius_m / (111320.0 * math.cos(math.radians(lat)))
        return cls(
            lat_min=lat - lat_delta,
            lon_min=lon - lon_delta,
            lat_max=lat + lat_delta,
            lon_max=lon + lon_delta,
        )


def _srtm_tile_name(lat: int, lon: int) -> str:
    """Return the SRTM tile filename for integer (lat, lon) of SW corner."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}.hgt"


def _srtm_zip_name(lat: int, lon: int) -> str:
    return _srtm_tile_name(lat, lon).replace(".hgt", ".SRTMGL1.hgt.zip")


def fetch_srtm(bbox: BBox, cache_dir: str | Path, source: str = "usgs") -> Path | None:
    """
    Download SRTM 1-arc-second (30m) tiles covering bbox to cache_dir.

    Returns the path to the merged .hgt file, or None on failure.

    source: 'usgs' — NASA SRTM via opentopography-style URL (public)
    """
    cache_dir = Path(cache_dir) / "srtm"
    cache_dir.mkdir(parents=True, exist_ok=True)

    import math

    lat_min_i = int(math.floor(bbox.lat_min))
    lat_max_i = int(math.floor(bbox.lat_max))
    lon_min_i = int(math.floor(bbox.lon_min))
    lon_max_i = int(math.floor(bbox.lon_max))

    downloaded: list[Path] = []

    for lat in range(lat_min_i, lat_max_i + 1):
        for lon in range(lon_min_i, lon_max_i + 1):
            tile_name = _srtm_tile_name(lat, lon)
            tile_path = cache_dir / tile_name
            if tile_path.exists():
                logger.info("srtm.cache_hit", tile=tile_name)
                downloaded.append(tile_path)
                continue

            # Try public SRTM mirror (NASA Earthdata requires login; use OpenTopography mirror)
            url = (
                f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/ASCII/"
                f"{tile_name}.zip"
            )
            # Fallback: USGS EarthExplorer-compatible endpoint
            url_alt = (
                f"https://step.esa.int/auxdata/dem/SRTMGL1/{_srtm_zip_name(lat, lon)}"
            )

            for dl_url in [url_alt]:
                try:
                    logger.info("srtm.downloading", tile=tile_name, url=dl_url)
                    with urllib.request.urlopen(dl_url, timeout=30) as resp:
                        data = resp.read()
                    # Unzip if needed
                    if dl_url.endswith(".zip"):
                        with zipfile.ZipFile(io.BytesIO(data)) as zf:
                            hgt_name = next(
                                n for n in zf.namelist() if n.endswith(".hgt")
                            )
                            tile_path.write_bytes(zf.read(hgt_name))
                    else:
                        tile_path.write_bytes(data)
                    downloaded.append(tile_path)
                    logger.info("srtm.downloaded", tile=tile_name)
                    break
                except Exception as exc:
                    logger.warning("srtm.download_failed", tile=tile_name, url=dl_url, error=str(exc))

    return cache_dir if downloaded else None


import functools

@functools.lru_cache(maxsize=64)
def _read_srtm_hgt(path: Path) -> np.ndarray:
    """
    Read a SRTM .hgt file into a numpy array (LRU-cached per tile).
    SRTM1 = 3601×3601 int16 big-endian, SRTM3 = 1201×1201.
    """
    data = path.read_bytes()
    n_samples = len(data) // 2
    side = int(round(n_samples**0.5))
    arr = np.frombuffer(data, dtype=">i2").reshape(side, side).astype(np.float32)
    # Replace void values
    arr[arr == -32768] = np.nan
    return arr


def _crop_srtm_to_bbox(
    arr: np.ndarray, tile_lat: int, tile_lon: int, bbox: BBox
) -> np.ndarray:
    """Crop a SRTM tile array to the given bbox."""
    side = arr.shape[0]  # 3601 or 1201
    # SRTM tiles: row 0 = north edge, col 0 = west edge
    # Tile covers [tile_lat, tile_lat+1] × [tile_lon, tile_lon+1]
    lat_res = 1.0 / (side - 1)
    lon_res = 1.0 / (side - 1)

    row_start = int(max(0, (tile_lat + 1 - bbox.lat_max) / lat_res))
    row_end = int(min(side, (tile_lat + 1 - bbox.lat_min) / lat_res)) + 1
    col_start = int(max(0, (bbox.lon_min - tile_lon) / lon_res))
    col_end = int(min(side, (bbox.lon_max - tile_lon) / lon_res)) + 1

    return arr[row_start:row_end, col_start:col_end]


class GeoPreprocessor:
    """
    Build conditioning tensors for the diffusion inference pipeline.

    Conditioning tensor shape: [3, H, W] float32
      channel 0: normalised DEM
      channel 1: normalised building height
      channel 2: vegetation attenuation

    Phase 1: returns zero tensors.
    Phase 2: loads SRTM data from cache, bilinear-interpolates to grid.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._cache_dir = Path(
            cache_dir or os.environ.get("TERRAIN_CACHE_DIR", "data/terrain")
        )

    def get_conditioning_tensor(
        self, bbox: BBox, resolution_px: int
    ) -> np.ndarray:
        """
        Return [3, H, W] float32 conditioning tensor for the given bbox.

        H = W = resolution_px.
        Phase 2: channels 0 filled from SRTM; channels 1,2 remain zero
        until OSM/Corine integration.
        """
        dem = self._get_dem(bbox, resolution_px)
        buildings = np.zeros((resolution_px, resolution_px), dtype=np.float32)
        vegetation = np.zeros((resolution_px, resolution_px), dtype=np.float32)
        return np.stack([dem, buildings, vegetation], axis=0)

    def get_mnt_hires(self, bbox: BBox, resolution_px: int) -> np.ndarray:
        """Return [1, H, W] float32 high-resolution DEM for super-resolution."""
        dem = self._get_dem(bbox, resolution_px)
        return dem[np.newaxis, :, :]

    def _get_dem(self, bbox: BBox, resolution_px: int) -> np.ndarray:
        """
        Load and interpolate DEM for bbox at resolution_px × resolution_px.
        Falls back to zeros if no terrain data available.
        """
        srtm_dir = self._cache_dir / "srtm"
        if not srtm_dir.exists():
            return np.zeros((resolution_px, resolution_px), dtype=np.float32)

        import math

        lat_min_i = int(math.floor(bbox.lat_min))
        lat_max_i = int(math.floor(bbox.lat_max))
        lon_min_i = int(math.floor(bbox.lon_min))
        lon_max_i = int(math.floor(bbox.lon_max))

        patches: list[np.ndarray] = []

        for lat in range(lat_min_i, lat_max_i + 1):
            for lon in range(lon_min_i, lon_max_i + 1):
                tile_path = srtm_dir / _srtm_tile_name(lat, lon)
                if not tile_path.exists():
                    continue
                try:
                    tile = _read_srtm_hgt(tile_path)
                    patch = _crop_srtm_to_bbox(tile, lat, lon, bbox)
                    patches.append(patch)
                except Exception as exc:
                    logger.warning("geo.srtm_read_failed", tile=str(tile_path), error=str(exc))

        if not patches:
            return np.zeros((resolution_px, resolution_px), dtype=np.float32)

        # Simple approach: use first patch if only one tile, else stack largest
        # Full mosaicing would require rasterio — kept minimal for Phase 2
        combined = patches[0]
        for p in patches[1:]:
            # Naive concatenation (correct only for adjacent tiles in same column/row)
            if p.shape[1] == combined.shape[1]:
                combined = np.vstack([combined, p])

        # Fill NaN with interpolated values
        combined = np.where(np.isnan(combined), 0.0, combined)

        # Bilinear resize to resolution_px × resolution_px
        resized = _resize_bilinear(combined, resolution_px, resolution_px)

        # Normalise to [0, 1]
        vmin, vmax = resized.min(), resized.max()
        if vmax > vmin:
            resized = (resized - vmin) / (vmax - vmin)

        return resized.astype(np.float32)


def _resize_bilinear(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Simple bilinear resize without scipy/cv2 dependency."""
    in_h, in_w = arr.shape
    row_idx = np.linspace(0, in_h - 1, out_h)
    col_idx = np.linspace(0, in_w - 1, out_w)

    r0 = np.floor(row_idx).astype(int).clip(0, in_h - 2)
    c0 = np.floor(col_idx).astype(int).clip(0, in_w - 2)
    r1 = r0 + 1
    c1 = c0 + 1

    dr = (row_idx - r0)[:, np.newaxis]
    dc = (col_idx - c0)[np.newaxis, :]

    out = (
        arr[r0][:, c0] * (1 - dr) * (1 - dc)
        + arr[r0][:, c1] * (1 - dr) * dc
        + arr[r1][:, c0] * dr * (1 - dc)
        + arr[r1][:, c1] * dr * dc
    )
    return out.astype(np.float32)


__all__ = ["BBox", "GeoPreprocessor", "fetch_srtm"]
