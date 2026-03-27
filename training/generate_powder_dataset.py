#!/usr/bin/env python3
"""
training/generate_powder_dataset.py — Convert POWDER RSS samples into UNet training scenes.

Each scene uses a Gaussian heatmap at the ground-truth TX position as the training
target, allowing direct fine-tuning of the UNet for localization (not just radio map
completion).

Input format:
  data/datasets/powder/separated_data/all_data/single_tx.json   — RSS samples
  data/datasets/powder/corrected_dsm.tif                       — DSM terrain (optional)
  data/datasets/powder/corrected_buildings.tif                  — Building heights (optional)

Output: .npz files matching the format expected by train_unet.py:
  conditioning  [3, H, W]  — DEM-norm, buildings-norm, zeros (vegetation)
  rssi_map      [H, W]     — sparse normalized RSSI at receiver positions
  rssi_mask     [H, W]     — 1 where receiver present
  radiomap_gt   [H, W]     — Gaussian heatmap at TX position (training label)
  bbox          [4]        — lat_min, lon_min, lat_max, lon_max
  emitter_pos   [3]        — lat, lon, alt_m (ground truth)
  emitter_px    [2]        — row, col in grid

Usage:
  python training/generate_powder_dataset.py \\
    --data    data/datasets/powder/separated_data/all_data/single_tx.json \\
    --dsm     data/datasets/powder/corrected_dsm.tif \\
    --buildings data/datasets/powder/corrected_buildings.tif \\
    --output  data/datasets/powder_unet_v1/ \\
    --resolution 64 \\
    --sigma-m 80
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants (must match eval_powder_dataset.py)
# ---------------------------------------------------------------------------

FREQ_HZ = 462_700_000.0
TX_POWER_DBM = 30.0
SEARCH_RADIUS_M = 3000.0
RSSI_FLOOR_DBM = -120.0
RSSI_RANGE_DB = 80.0   # normalise [-120, -40] dBm → [0, 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def rssi_to_norm(rssi_dbm: float) -> float:
    return float(np.clip((rssi_dbm - RSSI_FLOOR_DBM) / RSSI_RANGE_DB, 0.0, 1.0))


def bbox_from_center(lat: float, lon: float, radius_m: float) -> dict:
    deg = radius_m / 111_000.0
    return {"lat_min": lat-deg, "lat_max": lat+deg,
            "lon_min": lon-deg, "lon_max": lon+deg}


def make_grids(bbox: dict, res: int) -> tuple[np.ndarray, np.ndarray]:
    lats = np.linspace(bbox["lat_max"], bbox["lat_min"], res)
    lons = np.linspace(bbox["lon_min"], bbox["lon_max"], res)
    lon_g, lat_g = np.meshgrid(lons, lats)
    return lat_g, lon_g


def gaussian_heatmap(tx_row: float, tx_col: float, H: int, W: int,
                     sigma_px: float) -> np.ndarray:
    rows = np.arange(H, dtype=np.float32)[:, None]
    cols = np.arange(W, dtype=np.float32)[None, :]
    g = np.exp(-((rows - tx_row)**2 + (cols - tx_col)**2) / (2*sigma_px**2))
    g = g / (g.max() + 1e-8)
    return g.astype(np.float32)


# ---------------------------------------------------------------------------
# Terrain sampler (rasterio + pyproj)
# ---------------------------------------------------------------------------


class TifSampler:
    def __init__(self, tif_path: Path) -> None:
        import rasterio
        self._path = tif_path
        with rasterio.open(tif_path) as ds:
            self._crs_epsg = ds.crs.to_epsg()
            self._nodata = ds.nodata

    def sample_grid(self, lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
        import rasterio
        from pyproj import Transformer
        H, W = lat_grid.shape
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{self._crs_epsg}", always_xy=True)
        xs, ys = tr.transform(lon_grid.ravel(), lat_grid.ravel())
        with rasterio.open(self._path) as ds:
            raw = np.array([r[0] for r in ds.sample(zip(xs.tolist(), ys.tolist()))],
                           dtype=np.float32).reshape(H, W)
        if self._nodata is not None:
            raw[raw == self._nodata] = 0.0
        raw[~np.isfinite(raw)] = 0.0
        return raw


# ---------------------------------------------------------------------------
# Parse one POWDER sample
# ---------------------------------------------------------------------------


def parse_sample(sample: dict[str, Any]) -> dict | None:
    rx_data: list = sample.get("rx_data", [])
    tx_coords: list = sample.get("tx_coords", [])
    if not tx_coords or not rx_data:
        return None

    tx = tx_coords[0]
    if isinstance(tx, (list, tuple)):
        tx_lat, tx_lon = float(tx[0]), float(tx[1])
    elif isinstance(tx, dict):
        tx_lat = float(tx.get("latitude") or tx.get("lat") or 0.0)
        tx_lon = float(tx.get("longitude") or tx.get("lon") or 0.0)
    else:
        return None
    if tx_lat == 0.0 or tx_lon == 0.0:
        return None

    raw_rss, rx_positions = [], []
    for rx in rx_data:
        try:
            if isinstance(rx, (list, tuple)):
                rss, lat, lon = float(rx[0]), float(rx[1]), float(rx[2])
            elif isinstance(rx, dict):
                rss = float(rx.get("rss") or rx.get("rssi") or 0)
                lat = float(rx.get("latitude") or rx.get("lat") or 0)
                lon = float(rx.get("longitude") or rx.get("lon") or 0)
            else:
                continue
        except (TypeError, ValueError, IndexError):
            continue
        raw_rss.append(rss)
        rx_positions.append((lat, lon))

    if len(rx_positions) < 2:
        return None

    return {"tx_lat": tx_lat, "tx_lon": tx_lon,
            "raw_rss": raw_rss, "rx_positions": rx_positions}


def calibrate_rssi(raw_rss: list[float], distances_m: list[float]) -> list[float]:
    """Friis-based per-sample calibration (matches eval_powder_dataset.py)."""
    def fspl(d: float) -> float:
        d = max(d, 1.0)
        return 20*math.log10(d) + 20*math.log10(FREQ_HZ) - 147.55
    expected = [TX_POWER_DBM + 2.15 - fspl(d) for d in distances_m]
    raw = np.array(raw_rss, dtype=np.float64)
    exp = np.array(expected, dtype=np.float64)
    offset = float(np.mean(exp - raw))
    return [v + offset for v in raw_rss]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to single_tx.json")
    parser.add_argument("--dsm", default=None)
    parser.add_argument("--buildings", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--sigma-m", type=float, default=80.0,
                        help="Gaussian target σ in metres (default: 80m)")
    parser.add_argument("--min-obs", type=int, default=20)
    parser.add_argument("--max-scenes", type=int, default=0,
                        help="Max scenes to generate (0 = all)")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Fraction of samples used for training vs val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "val").mkdir(exist_ok=True)

    # Load data
    with open(args.data) as f:
        raw = json.load(f)
    samples = list(raw.values()) if isinstance(raw, dict) else raw
    samples = [s for s in samples if len(s.get("rx_data", [])) >= args.min_obs]
    print(f"Loaded {len(samples)} samples with ≥{args.min_obs} receivers")

    # Shuffle and split
    indices = np.random.permutation(len(samples))
    n_train = int(len(indices) * args.train_split)
    train_idx = set(indices[:n_train].tolist())

    # Terrain samplers
    dsm_sampler = TifSampler(Path(args.dsm)) if args.dsm else None
    bld_sampler = TifSampler(Path(args.buildings)) if args.buildings else None
    if dsm_sampler:
        print(f"Terrain: DSM={args.dsm}" + (f", buildings={args.buildings}" if bld_sampler else ""))

    sigma_deg = args.sigma_m / 111_000.0
    sigma_px = sigma_deg / (2 * SEARCH_RADIUS_M / 111_000.0) * args.resolution

    res = args.resolution
    success, skipped = 0, 0

    for scene_idx, sample in enumerate(samples):
        if args.max_scenes and success >= args.max_scenes:
            break

        parsed = parse_sample(sample)
        if parsed is None:
            skipped += 1
            continue

        tx_lat, tx_lon = parsed["tx_lat"], parsed["tx_lon"]
        rx_positions = parsed["rx_positions"]
        raw_rss = parsed["raw_rss"]

        # Center bbox on receiver centroid (same as eval)
        rx_lats = [p[0] for p in rx_positions]
        rx_lons = [p[1] for p in rx_positions]
        center_lat = (sum(rx_lats) + tx_lat) / (len(rx_lats) + 1)
        center_lon = (sum(rx_lons) + tx_lon) / (len(rx_lons) + 1)
        bbox = bbox_from_center(center_lat, center_lon, SEARCH_RADIUS_M)

        lat_g, lon_g = make_grids(bbox, res)

        # Calibrated RSSI → normalized
        distances = [haversine_m(tx_lat, tx_lon, rlat, rlon) for rlat, rlon in rx_positions]
        cal_rss = calibrate_rssi(raw_rss, distances)

        # Build sparse RSSI map + mask
        rssi_map = np.zeros((res, res), dtype=np.float32)
        rssi_mask = np.zeros((res, res), dtype=np.float32)
        lat_range = bbox["lat_max"] - bbox["lat_min"]
        lon_range = bbox["lon_max"] - bbox["lon_min"]
        for (rlat, rlon), rssi in zip(rx_positions, cal_rss):
            row = int((bbox["lat_max"] - rlat) / lat_range * res)
            col = int((rlon - bbox["lon_min"]) / lon_range * res)
            row = np.clip(row, 0, res-1)
            col = np.clip(col, 0, res-1)
            rssi_map[row, col] = rssi_to_norm(rssi)
            rssi_mask[row, col] = 1.0

        # Terrain conditioning [3, H, W]
        cond = np.zeros((3, res, res), dtype=np.float32)
        if dsm_sampler is not None:
            dsm = dsm_sampler.sample_grid(lat_g, lon_g)
            # Normalise DSM to [0,1] using approximate elevation range
            dsm_min, dsm_max = dsm[dsm > 0].min() if (dsm > 0).any() else 1300, dsm.max()
            if dsm_max > dsm_min:
                cond[0] = np.clip((dsm - dsm_min) / (dsm_max - dsm_min), 0, 1)
        if bld_sampler is not None:
            bld = bld_sampler.sample_grid(lat_g, lon_g)
            bld_max = max(bld.max(), 1.0)
            cond[1] = np.clip(bld / bld_max, 0, 1)

        # Emitter pixel position + Gaussian heatmap target
        tx_row = (bbox["lat_max"] - tx_lat) / lat_range * res
        tx_col = (tx_lon - bbox["lon_min"]) / lon_range * res
        target = gaussian_heatmap(tx_row, tx_col, res, res, sigma_px)

        # Check TX is within grid
        if not (0 <= tx_row < res and 0 <= tx_col < res):
            skipped += 1
            continue

        split = "train" if scene_idx in train_idx else "val"
        out_dir = args.output if split == "train" else args.output / "val"
        path = out_dir / f"scene_{success:06d}.npz"

        np.savez_compressed(
            path,
            conditioning=cond,
            rssi_map=rssi_map,
            rssi_mask=rssi_mask,
            radiomap_gt=target,
            bbox=np.array([bbox["lat_min"], bbox["lon_min"],
                           bbox["lat_max"], bbox["lon_max"]]),
            emitter_pos=np.array([tx_lat, tx_lon, 2.0]),
            emitter_px=np.array([tx_row, tx_col]),
            freq_hz=np.float32(FREQ_HZ),
        )
        success += 1

        if success % 200 == 0:
            print(f"  {success} scenes  ({skipped} skipped)  split={split}")

    # Save metadata
    meta = {
        "source": "POWDER outdoor RSS 462.7 MHz",
        "n_scenes": success,
        "n_skipped": skipped,
        "resolution_px": res,
        "sigma_m": args.sigma_m,
        "sigma_px": float(sigma_px),
        "search_radius_m": SEARCH_RADIUS_M,
        "train_split": args.train_split,
    }
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone: {success} scenes saved to {args.output}  ({skipped} skipped)")
    print(f"  sigma_px = {sigma_px:.2f}  ({args.sigma_m:.0f}m at 64px over {SEARCH_RADIUS_M*2/1000:.0f}km)")


if __name__ == "__main__":
    main()
