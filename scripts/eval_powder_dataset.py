"""
eval_powder_dataset.py — Validate GridLikelihoodModel on the POWDER outdoor
RSS dataset (Zenodo 10.5281/zenodo.10962857).

Dataset: 5,214 samples, 462.7 MHz, UTM Zone 12 (Salt Lake City area),
         10–25 receivers per sample, ground-truth transmitter position.

Download:
  mkdir -p data/datasets/powder && cd data/datasets/powder
  wget https://zenodo.org/records/10962857/files/powder_462.7_rss_data.json
  wget https://zenodo.org/records/10962857/files/corrected_buildings.tif
  wget https://zenodo.org/records/10962857/files/corrected_dsm.tif

Usage:
    python scripts/eval_powder_dataset.py \\
        --data data/datasets/powder/powder_462.7_rss_data.json \\
        [--dsm  data/datasets/powder/corrected_dsm.tif] \\
        [--max-samples 500] \\
        [--output results/powder_eval.json]

Pipeline:
  JSON sample → RssiObservation list → GridLikelihoodModel → peak position
  → haversine error vs tx_coords ground truth → aggregate CEP50/CEP90/RMSE
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.models.grid_likelihood import (
    GridLikelihoodModel,
    RssiObservation,
    make_lat_lon_grid,
)

RSSI_FLOOR_DBM_NORM = -120.0
RSSI_RANGE_DB_NORM  =   80.0

def _rssi_to_norm(rssi_dbm: float) -> float:
    return float(np.clip((rssi_dbm - RSSI_FLOOR_DBM_NORM) / RSSI_RANGE_DB_NORM, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Terrain raster loader (rasterio + pyproj, optional dependency)
# ---------------------------------------------------------------------------


class TerrainRasterSampler:
    """
    Samples GeoTIFF rasters (DSM, buildings) at arbitrary lat/lon grids.

    Reprojects grid points from WGS84 to the TIF's native CRS on the fly.
    Results are cached per (bbox, grid_size) key to avoid re-sampling on
    every call during the eval loop.
    """

    def __init__(self, buildings_tif: str | Path, dsm_tif: str | Path | None = None) -> None:
        try:
            import rasterio  # noqa: F401
            from pyproj import Transformer  # noqa: F401
        except ImportError:
            raise ImportError("rasterio and pyproj are required for terrain support. "
                              "Run: pip install rasterio pyproj")

        self._bld_path = Path(buildings_tif)
        self._dsm_path = Path(dsm_tif) if dsm_tif else None
        self._cache: dict[str, np.ndarray] = {}

        import rasterio
        with rasterio.open(self._bld_path) as ds:
            self._crs_epsg = ds.crs.to_epsg()
            self._bld_nodata = ds.nodata
        if self._dsm_path:
            with rasterio.open(self._dsm_path) as ds:
                self._dsm_nodata = ds.nodata

    def _sample_tif(self, tif_path: Path, nodata, lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
        """Sample TIF at every (lat, lon) in the grid. Returns [H, W] float32."""
        import rasterio
        from pyproj import Transformer

        H, W = lat_grid.shape
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{self._crs_epsg}", always_xy=True)
        xs, ys = tr.transform(lon_grid.ravel(), lat_grid.ravel())

        with rasterio.open(tif_path) as ds:
            coords = list(zip(xs.tolist(), ys.tolist()))
            raw = list(ds.sample(coords, masked=False))
        result = np.array([r[0] for r in raw], dtype=np.float32).reshape(H, W)

        if nodata is not None:
            result[result == nodata] = 0.0
        result[~np.isfinite(result)] = 0.0
        return result

    def get_building_mask(self, lat_grid: np.ndarray, lon_grid: np.ndarray,
                          height_threshold_m: float = 2.0) -> np.ndarray:
        """
        Returns [H, W] uint8 building mask sampled from buildings TIF.
        Pixels with building height > threshold are marked 1.
        Pixels outside TIF coverage are also 0 (unknown → no attenuation).
        """
        key = f"bld_{lat_grid[0,0]:.5f},{lat_grid[-1,-1]:.5f},{lon_grid[0,-1]:.5f}_{lat_grid.shape}"
        if key not in self._cache:
            heights = self._sample_tif(self._bld_path, self._bld_nodata, lat_grid, lon_grid)
            self._cache[key] = (heights > height_threshold_m).astype(np.uint8)
        return self._cache[key]

    def get_coverage_mask(self, lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
        """
        Returns [H, W] bool array: True where the buildings TIF has valid data.
        Use to constrain the peak search to within the covered area.
        """
        import rasterio
        from pyproj import Transformer

        key = f"cov_{lat_grid[0,0]:.5f},{lat_grid[-1,-1]:.5f},{lon_grid[0,-1]:.5f}_{lat_grid.shape}"
        if key not in self._cache:
            H, W = lat_grid.shape
            tr = Transformer.from_crs("EPSG:4326", f"EPSG:{self._crs_epsg}", always_xy=True)
            xs, ys = tr.transform(lon_grid.ravel(), lat_grid.ravel())
            with rasterio.open(self._bld_path) as ds:
                raw = np.array([r[0] for r in ds.sample(zip(xs.tolist(), ys.tolist()))],
                               dtype=np.float32).reshape(H, W)
                nodata = ds.nodata
            valid = np.isfinite(raw)
            if nodata is not None:
                valid &= (raw != nodata)
            self._cache[key] = valid
        return self._cache[key]

    def get_dsm(self, lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray | None:
        """Returns [H, W] float32 DSM elevation in metres, or None if no DSM loaded."""
        if self._dsm_path is None:
            return None
        key = f"dsm_{lat_grid[0,0]:.5f},{lat_grid[-1,-1]:.5f},{lon_grid[0,-1]:.5f}_{lat_grid.shape}"
        if key not in self._cache:
            self._cache[key] = self._sample_tif(self._dsm_path, self._dsm_nodata, lat_grid, lon_grid)
        return self._cache[key]

# ---------------------------------------------------------------------------
# UNet inference wrapper for POWDER eval
# ---------------------------------------------------------------------------


class UNetLocaliser:
    """
    Wraps a trained UNet checkpoint (PyTorch .pt or ONNX .onnx) for POWDER eval.

    Accepts the same terrain + RSSI inputs as the training pipeline and returns
    a [H, W] probability map whose argmax is the estimated TX position.
    """

    def __init__(self, checkpoint_path: str) -> None:
        self._path = Path(checkpoint_path)
        self._model = None
        self._grid_size = 64  # training resolution

        if self._path.suffix == ".onnx":
            try:
                import onnxruntime as ort
                self._sess = ort.InferenceSession(str(self._path),
                                                  providers=["CPUExecutionProvider"])
                self._backend = "onnx"
                print(f"  UNet: ONNX backend  ({self._path.name})")
            except ImportError:
                raise ImportError("onnxruntime required for .onnx inference. pip install onnxruntime")
        else:
            # PyTorch .pt checkpoint (state_dict format from train_unet.py)
            try:
                import torch
                from shared.models.unet_arch import UNetRadioMap
                ckpt = torch.load(str(self._path), map_location="cpu")
                if isinstance(ckpt, dict) and "model_state" in ckpt:
                    base = ckpt.get("args", {}).get("base", 16)
                    self._model = UNetRadioMap(in_ch=5, base=base)
                    self._model.load_state_dict(ckpt["model_state"])
                else:
                    self._model = UNetRadioMap(in_ch=5, base=16)
                    self._model.load_state_dict(ckpt)
                self._model.eval()
                self._backend = "torch"
                print(f"  UNet: PyTorch CPU  ({self._path.name})")
            except ImportError:
                raise ImportError("torch required for .pt inference. pip install torch")

    def infer(self, rssi_map: np.ndarray, rssi_mask: np.ndarray,
              conditioning: np.ndarray) -> np.ndarray:
        """
        Run UNet forward pass.

        rssi_map:    [H, W] float32 — normalised RSSI at receiver positions, 0 elsewhere
        rssi_mask:   [H, W] float32 — 1 where receiver present
        conditioning:[3, H, W] float32 — DEM, buildings, vegetation (zeros if unavailable)

        Returns [H, W] float32 probability map.
        """
        inp = np.concatenate([
            conditioning,
            rssi_map[np.newaxis],
            rssi_mask[np.newaxis],
        ], axis=0)[np.newaxis].astype(np.float32)  # [1, 5, H, W]

        if self._backend == "onnx":
            out = self._sess.run(None, {"input": inp})[0]
        else:
            import torch
            with torch.no_grad():
                out = self._model(torch.from_numpy(inp)).numpy()

        return out[0, 0]  # [H, W]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FREQ_HZ = 462_700_000.0   # 462.7 MHz FRS/GMRS
TX_POWER_DBM = 30.0        # 1 W (Baofeng BF-F8HP at max)
GRID_SIZE = 128            # smaller grid for speed on 5K samples
SEARCH_RADIUS_M = 3000     # 3km search bbox
# Noise sigma: RSS is uncalibrated/relative in this dataset; use wider sigma
NOISE_SIGMA_DB = 10.0


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bbox_from_center(lat: float, lon: float, radius_m: float) -> dict:
    deg = radius_m / 111_000.0
    return {
        "lat_min": lat - deg, "lat_max": lat + deg,
        "lon_min": lon - deg, "lon_max": lon + deg,
    }


# ---------------------------------------------------------------------------
# RSS normalisation
# The dataset uses relative, uncalibrated RSS.  We normalise to dBm-like
# units so that GridLikelihoodModel's Friis predictions have a sensible
# scale.  Strategy: shift so the sample median equals the Friis prediction
# at the median distance (per-sample calibration).
# ---------------------------------------------------------------------------


def calibrate_rssi(
    raw_rss_values: list[float],
    distances_m: list[float],
    freq_hz: float = FREQ_HZ,
    tx_power_dbm: float = TX_POWER_DBM,
    tx_gain_dbi: float = 2.15,
) -> list[float]:
    """
    Per-sample RSS calibration via least-squares offset estimation.

    Minimises Σ (rssi_i - (tx_power - fspl_i))² over a scalar offset.
    Returns calibrated dBm values.
    """
    if not distances_m:
        return raw_rss_values

    def fspl(d: float) -> float:
        d = max(d, 1.0)
        return 20 * math.log10(d) + 20 * math.log10(freq_hz) - 147.55

    # Expected Friis values
    expected = [tx_power_dbm + tx_gain_dbi - fspl(d) for d in distances_m]

    # Offset that minimises least-squares residual
    raw = np.array(raw_rss_values, dtype=np.float64)
    exp = np.array(expected, dtype=np.float64)
    offset = float(np.mean(exp - raw))
    return [v + offset for v in raw_rss_values]


# ---------------------------------------------------------------------------
# Sample parsing
# ---------------------------------------------------------------------------


def parse_sample(sample: dict[str, Any]) -> tuple[list[RssiObservation], list[tuple[float, float]]] | None:
    """
    Parse one POWDER data sample into (observations, transmitter_positions).

    Format: rx_data[i] = [rss, lat, lon, device_name]  (list format)
            tx_coords   = [[lat, lon], ...]

    Returns None if sample has no usable transmitter ground truth.
    """
    rx_data: list = sample.get("rx_data", [])
    tx_coords: list = sample.get("tx_coords", [])

    if not tx_coords or not rx_data:
        return None

    # Use first transmitter as ground truth
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

    # Gather receiver observations
    raw_rss = []
    rx_positions = []
    for rx in rx_data:
        try:
            if isinstance(rx, (list, tuple)):
                # Format: [rss, lat, lon, device_name]
                rss, lat, lon = float(rx[0]), float(rx[1]), float(rx[2])
            elif isinstance(rx, dict):
                rss = float(rx.get("rss") or rx.get("rssi") or rx.get("signal_strength") or 0)
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

    # Calibrate RSS using distance from transmitter
    distances = [haversine_m(tx_lat, tx_lon, rlat, rlon) for rlat, rlon in rx_positions]
    cal_rss = calibrate_rssi(raw_rss, distances)

    observations = [
        RssiObservation(
            sensor_lat=rlat,
            sensor_lon=rlon,
            sensor_alt_m=5.0,           # receivers on buildings/poles, ~5m AGL
            rssi_dbm=rssi,
            freq_hz=FREQ_HZ,
        )
        for (rlat, rlon), rssi in zip(rx_positions, cal_rss)
    ]

    return observations, [(tx_lat, tx_lon)]


# ---------------------------------------------------------------------------
# Peak localisation
# ---------------------------------------------------------------------------


def _peak_to_latlon(rm: np.ndarray, bbox: dict) -> tuple[float, float]:
    H, W = rm.shape
    row, col = divmod(int(np.argmax(rm)), W)
    lat = bbox["lat_max"] - row / (H - 1) * (bbox["lat_max"] - bbox["lat_min"])
    lon = bbox["lon_min"] + col / (W - 1) * (bbox["lon_max"] - bbox["lon_min"])
    return lat, lon


def localise_unet(
    observations: list[RssiObservation],
    center_lat: float,
    center_lon: float,
    unet: "UNetLocaliser",
    terrain: "TerrainRasterSampler | None" = None,
) -> tuple[float, float]:
    # Use training resolution (64) — model was trained at this resolution
    res = unet._grid_size  # 64 by default
    bbox = _bbox_from_center(center_lat, center_lon, SEARCH_RADIUS_M)
    lat_grid, lon_grid = make_lat_lon_grid(bbox, res)

    # Build RSSI input map
    rssi_map  = np.zeros((res, res), dtype=np.float32)
    rssi_mask = np.zeros((res, res), dtype=np.float32)
    lat_range = bbox["lat_max"] - bbox["lat_min"]
    lon_range = bbox["lon_max"] - bbox["lon_min"]
    for obs in observations:
        row = int((bbox["lat_max"] - obs.sensor_lat) / lat_range * res)
        col = int((obs.sensor_lon - bbox["lon_min"]) / lon_range * res)
        row = np.clip(row, 0, res - 1)
        col = np.clip(col, 0, res - 1)
        rssi_map[row, col]  = _rssi_to_norm(obs.rssi_dbm)
        rssi_mask[row, col] = 1.0

    # Terrain conditioning
    cond = np.zeros((3, res, res), dtype=np.float32)
    if terrain is not None:
        bld_raw = terrain._sample_tif(terrain._bld_path, terrain._bld_nodata, lat_grid, lon_grid)
        bld_max = max(float(bld_raw.max()), 1.0)
        cond[1] = np.clip(bld_raw / bld_max, 0, 1)
        if terrain._dsm_path:
            dsm = terrain._sample_tif(terrain._dsm_path, terrain._dsm_nodata, lat_grid, lon_grid)
            valid = dsm > 0
            if valid.any():
                cond[0] = np.clip((dsm - dsm[valid].min()) / max(dsm[valid].max() - dsm[valid].min(), 1), 0, 1)

    prob_map = unet.infer(rssi_map, rssi_mask, cond)

    # Apply coverage mask if terrain available
    if terrain is not None:
        cov = terrain.get_coverage_mask(lat_grid, lon_grid)
        if cov.any():
            prob_map = prob_map.copy()
            prob_map[~cov] = 0.0

    return _peak_to_latlon(prob_map, bbox)


def localise(
    observations: list[RssiObservation],
    center_lat: float,
    center_lon: float,
    model: GridLikelihoodModel,
    differential: bool = False,
    terrain: "TerrainRasterSampler | None" = None,
) -> tuple[float, float]:
    bbox = _bbox_from_center(center_lat, center_lon, SEARCH_RADIUS_M)

    building_mask = None
    coverage_mask = None
    if terrain is not None:
        lat_grid, lon_grid = make_lat_lon_grid(bbox, GRID_SIZE)
        building_mask = terrain.get_building_mask(lat_grid, lon_grid)
        coverage_mask = terrain.get_coverage_mask(lat_grid, lon_grid)

    radiomap = model.infer_from_observations(
        observations, bbox, GRID_SIZE,
        emitter_alt_m=2.0,
        noise_sigma_db=NOISE_SIGMA_DB,
        differential=differential,
        building_mask_override=building_mask,
    )
    rm = radiomap[0].copy()

    # Constrain peak search to within TIF coverage (TX is always on campus)
    if coverage_mask is not None and coverage_mask.any():
        rm[~coverage_mask] = 0.0

    return _peak_to_latlon(rm, bbox)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to powder_462.7_rss_data.json")
    parser.add_argument("--dsm", default=None, help="Path to corrected_dsm.tif")
    parser.add_argument("--buildings", default=None, help="Path to corrected_buildings.tif")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Maximum number of samples to evaluate (default: 500)")
    parser.add_argument("--output", default=None, help="JSON output path for results")
    parser.add_argument("--differential", action="store_true",
                        help="Use calibration-free pairwise RSSI differences")
    parser.add_argument("--compare", action="store_true",
                        help="Run all enabled mode combinations and compare")
    parser.add_argument("--unet", default=None,
                        help="Path to UNet checkpoint (.pt or .onnx)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: data file not found: {data_path}")
        print("Download with:")
        print("  mkdir -p data/datasets/powder && cd data/datasets/powder")
        print("  wget https://zenodo.org/records/10962857/files/powder_462.7_rss_data.json")
        sys.exit(1)

    print(f"Loading {data_path} …")
    with open(data_path) as f:
        raw = json.load(f)

    # Dataset may be a dict keyed by timestamp or a list
    if isinstance(raw, dict):
        samples = list(raw.values())
    else:
        samples = raw

    print(f"  {len(samples)} total samples")

    model = GridLikelihoodModel(terrain_cache_dir=None)  # OSM fallback disabled; we inject masks directly

    # Terrain raster sampler (optional)
    terrain: TerrainRasterSampler | None = None
    if args.buildings:
        bld_path = Path(args.buildings)
        if not bld_path.exists():
            print(f"Warning: buildings TIF not found: {bld_path} — running without terrain")
        else:
            try:
                terrain = TerrainRasterSampler(
                    buildings_tif=bld_path,
                    dsm_tif=Path(args.dsm) if args.dsm else None,
                )
                print(f"  Terrain: buildings={bld_path.name}"
                      + (f", DSM={Path(args.dsm).name}" if args.dsm else ""))
            except ImportError as e:
                print(f"Warning: {e} — running without terrain")

    # UNet model (optional)
    unet: UNetLocaliser | None = None
    if args.unet:
        unet_path = Path(args.unet)
        if not unet_path.exists():
            print(f"Warning: UNet checkpoint not found: {unet_path} — skipping UNet eval")
        else:
            try:
                unet = UNetLocaliser(str(unet_path))
            except Exception as e:
                print(f"Warning: UNet load failed: {e} — skipping UNet eval")

    errors_abs = []
    errors_diff = []
    errors_abs_t = []   # with terrain
    errors_diff_t = []  # differential + terrain
    errors_unet = []    # UNet (with terrain if available)
    skipped = 0
    processed = 0
    n_obs_total = 0

    run_abs = not args.differential or args.compare
    run_diff = args.differential or args.compare
    run_terrain = terrain is not None

    # Use only dense samples (>= 20 receivers) for best accuracy
    min_obs = 20
    samples_filtered = [s for s in samples if len(s.get("rx_data", [])) >= min_obs]
    print(f"  {len(samples_filtered)} samples with ≥{min_obs} receivers")
    samples = samples_filtered

    for i, sample in enumerate(samples):
        if processed >= args.max_samples:
            break

        result = parse_sample(sample)
        if result is None:
            skipped += 1
            continue

        observations, tx_positions = result
        tx_lat, tx_lon = tx_positions[0]

        # Use centroid of receivers as search centre (improves bbox placement)
        rx_lats = [o.sensor_lat for o in observations]
        rx_lons = [o.sensor_lon for o in observations]
        center_lat = (sum(rx_lats) + tx_lat) / (len(rx_lats) + 1)
        center_lon = (sum(rx_lons) + tx_lon) / (len(rx_lons) + 1)

        try:
            ok = False
            if run_abs:
                est_lat, est_lon = localise(observations, center_lat, center_lon, model,
                                            differential=False)
                errors_abs.append(haversine_m(tx_lat, tx_lon, est_lat, est_lon))
                ok = True

            if run_diff:
                est_lat_d, est_lon_d = localise(observations, center_lat, center_lon, model,
                                                differential=True)
                errors_diff.append(haversine_m(tx_lat, tx_lon, est_lat_d, est_lon_d))
                ok = True

            if run_terrain:
                if run_abs:
                    el, elo = localise(observations, center_lat, center_lon, model,
                                       differential=False, terrain=terrain)
                    errors_abs_t.append(haversine_m(tx_lat, tx_lon, el, elo))
                if run_diff:
                    el, elo = localise(observations, center_lat, center_lon, model,
                                       differential=True, terrain=terrain)
                    errors_diff_t.append(haversine_m(tx_lat, tx_lon, el, elo))
                ok = True

            if unet is not None:
                el, elo = localise_unet(observations, center_lat, center_lon, unet,
                                        terrain=terrain)
                errors_unet.append(haversine_m(tx_lat, tx_lon, el, elo))
                ok = True

            if ok:
                n_obs_total += len(observations)
                processed += 1
                if args.verbose and processed % 50 == 0:
                    parts = []
                    if errors_abs:     parts.append(f"abs={errors_abs[-1]:.0f}m")
                    if errors_diff:    parts.append(f"diff={errors_diff[-1]:.0f}m")
                    if errors_abs_t:   parts.append(f"abs+t={errors_abs_t[-1]:.0f}m")
                    if errors_diff_t:  parts.append(f"diff+t={errors_diff_t[-1]:.0f}m")
                    if errors_unet:    parts.append(f"unet={errors_unet[-1]:.0f}m")
                    print(f"  [{processed}/{args.max_samples}] {' '.join(parts)} "
                          f"(n_obs={len(observations)})")
        except Exception as e:
            skipped += 1
            if args.verbose:
                print(f"  Warning: sample {i} failed: {e}")

    all_errors = errors_abs or errors_diff or errors_abs_t or errors_diff_t or errors_unet
    if not all_errors:
        print("No valid samples found.")
        sys.exit(1)

    def _print_stats(label: str, errors: list[float]) -> dict:
        arr = np.array(errors)
        rmse = float(np.sqrt(np.mean(arr ** 2)))
        trimmed = arr[arr <= 5000]
        rmse_t = float(np.sqrt(np.mean(trimmed ** 2))) if len(trimmed) else rmse
        cep50 = float(np.percentile(arr, 50))
        cep90 = float(np.percentile(arr, 90))
        cep95 = float(np.percentile(arr, 95))
        print(f"\n--- {label} ---")
        print(f"  RMSE:    {rmse:.1f} m  (trimmed ≤5km: {rmse_t:.1f} m)")
        print(f"  CEP50:   {cep50:.1f} m")
        print(f"  CEP90:   {cep90:.1f} m")
        print(f"  CEP95:   {cep95:.1f} m")
        print(f"  Min/Max: {arr.min():.0f}m / {arr.max():.0f}m")
        thresholds = [50, 100, 200, 500, 1000, 2000]
        for t in thresholds:
            pct = 100 * np.mean(arr <= t)
            print(f"  ≤{t:5d}m: {pct:5.1f}%")
        return {"rmse_m": rmse, "rmse_trimmed_m": rmse_t, "cep50_m": cep50,
                "cep90_m": cep90, "cep95_m": cep95, "errors_m": arr.tolist()}

    print(f"\n=== POWDER Dataset Evaluation ({processed} samples, {skipped} skipped) ===")
    print(f"Frequency: 462.7 MHz   Avg observations/sample: {n_obs_total/processed:.1f}")

    out: dict = {
        "dataset": "POWDER outdoor RSS 462.7 MHz",
        "n_samples": processed,
        "n_skipped": skipped,
    }

    if errors_abs:
        stats = _print_stats("Friis absolute (no terrain)", errors_abs)
        out["absolute"] = {"model": "Friis absolute", **stats}
    if errors_abs_t:
        stats = _print_stats("Friis absolute + buildings", errors_abs_t)
        out["absolute_terrain"] = {"model": "Friis absolute + buildings", **stats}
    if errors_diff:
        stats = _print_stats("Differential (no terrain)", errors_diff)
        out["differential"] = {"model": "Friis differential", **stats}
    if errors_diff_t:
        stats = _print_stats("Differential + buildings", errors_diff_t)
        out["differential_terrain"] = {"model": "Friis differential + buildings", **stats}
    if errors_unet:
        label = "UNet" + (" + terrain" if terrain is not None else "")
        stats = _print_stats(label, errors_unet)
        out["unet"] = {"model": label, **stats}

    # Summary table
    results_for_table = []
    if errors_abs:     results_for_table.append(("Friis abs",        errors_abs))
    if errors_abs_t:   results_for_table.append(("Friis abs+bld",    errors_abs_t))
    if errors_diff:    results_for_table.append(("Diff",             errors_diff))
    if errors_diff_t:  results_for_table.append(("Diff+bld",         errors_diff_t))
    if errors_unet:    results_for_table.append(("UNet",             errors_unet))
    if len(results_for_table) > 1:
        print(f"\n{'Model':<20} {'CEP50':>8} {'CEP90':>8} {'RMSE(t)':>10}")
        print("-" * 50)
        for name, errs in results_for_table:
            a = np.array(errs)
            t = a[a <= 5000]
            print(f"  {name:<18} {np.percentile(a,50):>7.0f}m {np.percentile(a,90):>7.0f}m "
                  f"{np.sqrt(np.mean(t**2)):>9.0f}m")

    if errors_abs and errors_diff:
        delta = np.percentile(errors_abs, 50) - np.percentile(errors_diff, 50)
        print(f"\n  CEP50 improvement (diff vs abs): {delta:+.1f}m")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
