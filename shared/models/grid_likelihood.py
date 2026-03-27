"""
shared.models.grid_likelihood — Grid-search likelihood surface for RF localisation.

Replaces the diffusion model when no trained checkpoint is available.

For each grid point (lat, lon), computes:
  log P(observations | emitter at (lat,lon))
  = Σ_i  −(rssi_observed_i − rssi_predicted_i)² / (2σ²)

where rssi_predicted_i = Friis_RSSI − building_attenuation(emitter→sensor path)

Building attenuation (optional):
  Load OSM building centroids from terrain cache.
  Rasterize them onto the likelihood grid.
  For each sensor, count building cells along the ray from each candidate
  emitter position to the sensor. Add N × ATT_PER_BUILDING_DB to path loss.

Vectorised with numpy — runs in <100ms for 256×256 grid, 20 observations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

SPEED_OF_LIGHT = 2.998e8  # m/s
ATT_PER_BUILDING_DB = 10.0   # dB attenuation per building cell along ray
ATT_MAX_BUILDING_DB = 30.0   # physical cap (signal diffracts in deep NLOS)
BUILDING_RAY_SAMPLES = 30    # samples along emitter→sensor ray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fspl_db(distance_m: np.ndarray, freq_hz: float) -> np.ndarray:
    """Vectorised free-space path loss in dB. Clips distance at 1m."""
    d = np.maximum(distance_m, 1.0)
    return (
        20 * np.log10(d)
        + 20 * np.log10(freq_hz)
        - 20 * math.log10(SPEED_OF_LIGHT / (4 * math.pi))
    )


def _haversine_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    """
    Vectorised haversine distance (metres) from grid to a single point.

    lat1, lon1: [H, W] grid arrays in decimal degrees
    lat2, lon2: scalar point
    """
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * math.cos(phi2) * np.sin(dl / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _distance_3d_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    alt1_m: float,
    lat2: float,
    lon2: float,
    alt2_m: float,
) -> np.ndarray:
    """3D distance from each grid point to a sensor, in metres."""
    d_horiz = _haversine_m(lat1, lon1, lat2, lon2)
    dz = alt2_m - alt1_m  # emitter assumed at alt1_m (0 m default)
    return np.sqrt(d_horiz**2 + dz**2)


# ---------------------------------------------------------------------------
# OSM building loading and rasterisation
# ---------------------------------------------------------------------------


def load_building_centroids(terrain_cache_dir: str | Path, bbox_dict: dict) -> np.ndarray | None:
    """
    Load OSM building centroids from the terrain cache that best overlaps bbox_dict.

    Returns an [N, 2] float32 array of (lat, lon) centroids, or None if no data.
    The cache stores files as buildings_{lat_min}_{lon_min}_{lat_max}_{lon_max}.geojson
    (Overpass JSON format with 'elements' containing nodes and ways).
    """
    osm_dir = Path(terrain_cache_dir) / "osm"
    if not osm_dir.exists():
        return None

    # Find the geojson file whose bbox most covers the requested area
    lat_min = bbox_dict["lat_min"]
    lon_min = bbox_dict["lon_min"]
    lat_max = bbox_dict["lat_max"]
    lon_max = bbox_dict["lon_max"]

    best_file = None
    for p in osm_dir.glob("buildings_*.geojson"):
        parts = p.stem.split("_")
        if len(parts) == 5:
            try:
                flm, flnm, flx, flnx = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                if flm <= lat_min and flnm <= lon_min and flx >= lat_max and flnx >= lon_max:
                    best_file = p
                    break
            except ValueError:
                pass

    if best_file is None:
        return None

    try:
        data = json.loads(best_file.read_text())
    except Exception:
        return None

    elements = data.get("elements", [])
    nodes_by_id: dict[int, tuple[float, float]] = {
        e["id"]: (e["lat"], e["lon"])
        for e in elements if e.get("type") == "node" and "lat" in e
    }
    ways = [e for e in elements if e.get("type") == "way"]

    centroids = []
    for way in ways:
        node_ids = way.get("nodes", [])
        pts = [nodes_by_id[n] for n in node_ids if n in nodes_by_id]
        if len(pts) >= 3:
            lats = [p[0] for p in pts]
            lons = [p[1] for p in pts]
            clat = sum(lats) / len(lats)
            clon = sum(lons) / len(lons)
            # Only include buildings within the bbox
            if lat_min <= clat <= lat_max and lon_min <= clon <= lon_max:
                centroids.append((clat, clon))

    if not centroids:
        return None

    return np.array(centroids, dtype=np.float32)


def rasterize_building_centroids(
    centroids: np.ndarray,  # [N, 2] lat/lon
    lat_grid: np.ndarray,   # [H, W]
    lon_grid: np.ndarray,   # [H, W]
) -> np.ndarray:
    """
    Rasterize building centroids onto the grid by marking the nearest pixel.
    Returns [H, W] uint8 building mask (1 = building present, 0 = free).
    """
    H, W = lat_grid.shape
    mask = np.zeros((H, W), dtype=np.uint8)

    if len(centroids) == 0:
        return mask

    # Grid extents
    lat_max = lat_grid[0, 0]
    lat_min = lat_grid[-1, 0]
    lon_min = lon_grid[0, 0]
    lon_max = lon_grid[0, -1]

    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    if lat_range == 0 or lon_range == 0:
        return mask

    # Convert centroids to pixel coords
    rows = ((lat_max - centroids[:, 0]) / lat_range * (H - 1)).clip(0, H - 1).astype(int)
    cols = ((centroids[:, 1] - lon_min) / lon_range * (W - 1)).clip(0, W - 1).astype(int)

    mask[rows, cols] = 1
    return mask


def building_ray_attenuation_grid(
    sensor_lat: float,
    sensor_lon: float,
    lat_grid: np.ndarray,   # [H, W]
    lon_grid: np.ndarray,   # [H, W]
    building_mask: np.ndarray,  # [H, W] uint8
    n_samples: int = BUILDING_RAY_SAMPLES,
    att_per_building_db: float = ATT_PER_BUILDING_DB,
) -> np.ndarray:
    """
    Compute building attenuation (dB) from each candidate emitter position to a sensor.

    For each grid point (emitter candidate), samples N points along the straight
    path to the sensor and counts building mask hits. Returns [H, W] float32 dB.
    """
    H, W = lat_grid.shape

    lat_max = lat_grid[0, 0]
    lat_min = lat_grid[-1, 0]
    lon_min = lon_grid[0, 0]
    lon_max = lon_grid[0, -1]
    lat_range = lat_max - lat_min or 1e-9
    lon_range = lon_max - lon_min or 1e-9

    # Sensor pixel coords
    s_row = (lat_max - sensor_lat) / lat_range * (H - 1)
    s_col = (sensor_lon - lon_min) / lon_range * (W - 1)

    # Grid of row/col indices [H, W]
    rows = np.arange(H, dtype=np.float32)[:, None] * np.ones(W, dtype=np.float32)
    cols = np.arange(W, dtype=np.float32)[None, :] * np.ones(H, dtype=np.float32)

    # Sample fractions along ray: sensor → emitter (avoid endpoints)
    fracs = np.linspace(0.05, 0.95, n_samples, dtype=np.float32)  # [n_samples]

    # Sample positions: [n_samples, H, W]
    sample_rows = (s_row + fracs[:, None, None] * (rows[None] - s_row)).clip(0, H - 1)
    sample_cols = (s_col + fracs[:, None, None] * (cols[None] - s_col)).clip(0, W - 1)

    # Integer indices for lookup
    sr = sample_rows.astype(np.int32)
    sc = sample_cols.astype(np.int32)

    # Building hits along ray [n_samples, H, W]
    hits = building_mask[sr, sc]  # [n_samples, H, W]

    # Count transitions free→building (entering a building)
    transitions = np.diff(hits.astype(np.int8), axis=0)  # [n_samples-1, H, W]
    n_crossings = np.sum(transitions > 0, axis=0).astype(np.float32)  # [H, W]

    return np.minimum(n_crossings * att_per_building_db, ATT_MAX_BUILDING_DB)


# ---------------------------------------------------------------------------
# Observation record
# ---------------------------------------------------------------------------


class RssiObservation(NamedTuple):
    """A single RSSI observation from a sensor node."""
    sensor_lat: float
    sensor_lon: float
    sensor_alt_m: float
    rssi_dbm: float
    freq_hz: float


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def compute_likelihood_grid(
    observations: list[RssiObservation],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    emitter_alt_m: float = 5.0,
    noise_sigma_db: float = 6.0,
    tx_power_dbm: float = 10.0,
    tx_gain_dbi: float = 2.15,
    rx_gain_dbi: float = 0.0,
    building_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute a normalised likelihood surface [H, W] in [0, 1].

    For each candidate emitter position (lat, lon) in the grid:
      log_likelihood += Σ_i −(rssi_obs_i − rssi_pred_i)² / (2σ²)

    where rssi_pred_i = tx_power + tx_gain + rx_gain
                        − FSPL(distance_i, freq_hz_i)
                        − building_attenuation(emitter→sensor ray)

    Parameters
    ----------
    observations : list of RssiObservation
    lat_grid, lon_grid : [H, W] arrays of candidate emitter positions
    emitter_alt_m : assumed emitter altitude (metres)
    noise_sigma_db : assumed RSSI measurement noise (dB std)
    tx_power_dbm, tx_gain_dbi, rx_gain_dbi : assumed TX parameters
    building_mask : optional [H, W] uint8 rasterised building footprints;
                   when provided, building wall crossings are counted along
                   each emitter→sensor ray and added to path loss.

    Returns
    -------
    likelihood : [H, W] float32, normalised to [0, 1]
    """
    log_ll = np.zeros(lat_grid.shape, dtype=np.float64)

    for obs in observations:
        # Distance from each candidate position to this sensor
        dist_m = _distance_3d_m(
            lat_grid, lon_grid, emitter_alt_m,
            obs.sensor_lat, obs.sensor_lon, obs.sensor_alt_m,
        )

        # Predicted RSSI at sensor if emitter is at each grid point
        rssi_pred = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - _fspl_db(dist_m, obs.freq_hz)

        # Optional building attenuation along the ray.
        # Skip for elevated sensors (≥30m MSL) — they're on high ground or airborne
        # and their paths clear buildings via line-of-sight over terrain.
        # Scale by cos(elevation_angle) for remaining near-ground paths.
        if building_mask is not None and obs.sensor_alt_m < 30.0:
            extra_db = building_ray_attenuation_grid(
                obs.sensor_lat, obs.sensor_lon,
                lat_grid, lon_grid, building_mask,
            )
            # Approximate horizontal distance from sensor to each grid cell
            d_horiz_m = _haversine_m(lat_grid, lon_grid, obs.sensor_lat, obs.sensor_lon)
            d_vert_m = abs(obs.sensor_alt_m - emitter_alt_m)
            elev_factor = np.cos(np.arctan2(d_vert_m, np.maximum(d_horiz_m, 1.0)))
            rssi_pred -= extra_db * elev_factor

        # Gaussian log-likelihood
        log_ll += -((obs.rssi_dbm - rssi_pred) ** 2) / (2.0 * noise_sigma_db ** 2)

    # Normalise: shift so max = 0, then exponentiate
    log_ll -= log_ll.max()
    likelihood = np.exp(log_ll).astype(np.float32)

    return likelihood


def compute_differential_likelihood_grid(
    observations: list[RssiObservation],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    emitter_alt_m: float = 5.0,
    noise_sigma_db: float = 6.0,
    building_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calibration-free likelihood surface using pairwise RSSI differences.

    Eliminates TX power and per-device gain offsets entirely:
      delta_ij_obs  = rssi_i - rssi_j
      delta_ij_pred = FSPL(d_j) - FSPL(d_i) = 20*log10(d_j / d_i)
                    + building_att_j - building_att_i

    log_likelihood = Σ_{i<j}  -(delta_ij_obs - delta_ij_pred)² / (2σ²)

    The noise sigma applies to the *difference* — in practice use
    sqrt(2) × single-measurement sigma (default handled by caller).

    Returns [H, W] float32 normalised to [0, 1].
    """
    n = len(observations)
    if n < 2:
        return np.ones(lat_grid.shape, dtype=np.float32)

    freq_hz = observations[0].freq_hz  # assume same channel for all obs

    # Pre-compute distances and building attenuation for each sensor [n, H, W]
    fspl_grids = []
    for obs in observations:
        dist_m = _distance_3d_m(
            lat_grid, lon_grid, emitter_alt_m,
            obs.sensor_lat, obs.sensor_lon, obs.sensor_alt_m,
        )
        fspl = _fspl_db(dist_m, freq_hz)  # [H, W]

        if building_mask is not None and obs.sensor_alt_m < 30.0:
            extra_db = building_ray_attenuation_grid(
                obs.sensor_lat, obs.sensor_lon,
                lat_grid, lon_grid, building_mask,
            )
            d_horiz_m = _haversine_m(lat_grid, lon_grid, obs.sensor_lat, obs.sensor_lon)
            d_vert_m = abs(obs.sensor_alt_m - emitter_alt_m)
            elev_factor = np.cos(np.arctan2(d_vert_m, np.maximum(d_horiz_m, 1.0)))
            fspl = fspl + extra_db * elev_factor

        fspl_grids.append(fspl)

    log_ll = np.zeros(lat_grid.shape, dtype=np.float64)
    sigma2 = 2.0 * noise_sigma_db ** 2

    for i in range(n):
        for j in range(i + 1, n):
            delta_obs = observations[i].rssi_dbm - observations[j].rssi_dbm
            # predicted difference = path_loss_j - path_loss_i
            delta_pred = fspl_grids[j] - fspl_grids[i]
            log_ll += -((delta_obs - delta_pred) ** 2) / sigma2

    log_ll -= log_ll.max()
    return np.exp(log_ll).astype(np.float32)


# ---------------------------------------------------------------------------
# High-level API matching DiffusionModel.infer() signature
# ---------------------------------------------------------------------------


def make_lat_lon_grid(
    bbox_dict: dict,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build [H, W] lat/lon grids from a bbox dict.
    Row 0 = north (lat_max), Row H-1 = south (lat_min).
    Col 0 = west (lon_min), Col W-1 = east (lon_max).
    """
    lats = np.linspace(bbox_dict["lat_max"], bbox_dict["lat_min"], grid_size)
    lons = np.linspace(bbox_dict["lon_min"], bbox_dict["lon_max"], grid_size)
    lon_grid, lat_grid = np.meshgrid(lons, lats)  # both [H, W]
    return lat_grid, lon_grid


class GridLikelihoodModel:
    """
    Drop-in replacement for DiffusionModel when no checkpoint is available.

    Computes a Friis+building-attenuation likelihood surface from sparse RSSI
    observations.  If terrain_cache_dir is set and OSM buildings cover the bbox,
    building wall crossings along each emitter→sensor ray add extra path loss,
    sharpening the likelihood peak in built-up areas.
    """

    def __init__(self, terrain_cache_dir: str | Path | None = None) -> None:
        self._terrain_cache_dir = Path(terrain_cache_dir) if terrain_cache_dir else None
        self._buildings_cache: dict[str, np.ndarray | None] = {}  # bbox_key → centroids

    def _get_building_mask(
        self,
        bbox_dict: dict,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
    ) -> np.ndarray | None:
        """Load (and cache) building mask for the given bbox."""
        if self._terrain_cache_dir is None:
            return None
        key = f"{bbox_dict['lat_min']:.4f},{bbox_dict['lon_min']:.4f}," \
              f"{bbox_dict['lat_max']:.4f},{bbox_dict['lon_max']:.4f}"
        if key not in self._buildings_cache:
            centroids = load_building_centroids(self._terrain_cache_dir, bbox_dict)
            if centroids is not None and len(centroids) > 0:
                self._buildings_cache[key] = rasterize_building_centroids(
                    centroids, lat_grid, lon_grid
                )
            else:
                self._buildings_cache[key] = None
        return self._buildings_cache.get(key)

    def infer_from_observations(
        self,
        observations: list[RssiObservation],
        bbox_dict: dict,
        grid_size: int,
        emitter_alt_m: float = 5.0,
        noise_sigma_db: float = 6.0,
        differential: bool = False,
        building_mask_override: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Returns [1, H, W] float32 likelihood grid.

        Parameters match what the master pipeline needs to call.
        Building attenuation is applied automatically when terrain_cache_dir
        was supplied at construction and covers the requested bbox.

        differential : if True, use pairwise RSS differences (calibration-free).
                       Requires ≥2 observations; sigma applies to the difference.
        building_mask_override : [H, W] uint8 array — when provided, used instead
                                 of the terrain-cache-derived mask.  Allows callers
                                 to inject raster-derived building footprints.
        """
        if not observations:
            return np.zeros((1, grid_size, grid_size), dtype=np.float32)

        lat_grid, lon_grid = make_lat_lon_grid(bbox_dict, grid_size)
        if building_mask_override is not None:
            building_mask = building_mask_override
        else:
            building_mask = self._get_building_mask(bbox_dict, lat_grid, lon_grid)

        if differential:
            likelihood = compute_differential_likelihood_grid(
                observations,
                lat_grid,
                lon_grid,
                emitter_alt_m=emitter_alt_m,
                noise_sigma_db=noise_sigma_db,
                building_mask=building_mask,
            )
        else:
            likelihood = compute_likelihood_grid(
                observations,
                lat_grid,
                lon_grid,
                emitter_alt_m=emitter_alt_m,
                noise_sigma_db=noise_sigma_db,
                building_mask=building_mask,
            )
        return likelihood[np.newaxis, :, :]  # [1, H, W]


__all__ = [
    "RssiObservation",
    "GridLikelihoodModel",
    "compute_likelihood_grid",
    "compute_differential_likelihood_grid",
    "make_lat_lon_grid",
]
