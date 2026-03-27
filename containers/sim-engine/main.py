"""
sim-engine — RF propagation simulation engine.

Provides REST API for RSSI computation and radio map generation.
Supports propagation models: friis, itm (simplified), sionna (stub).
"""

from __future__ import annotations

import functools
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure shared is importable when running inside Docker
sys.path.insert(0, "/app")

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger(__name__)

def _seed_emitters_from_env() -> None:
    """
    Auto-register emitters defined via environment variables at startup.

    Single emitter:
        EMITTER_LAT, EMITTER_LON, EMITTER_ALT_M, EMITTER_FREQ_HZ, EMITTER_POWER_DBM

    Indexed emitters (N >= 0):
        EMITTER_0_LAT, EMITTER_0_LON, EMITTER_0_ALT_M, EMITTER_0_FREQ_HZ, EMITTER_0_POWER_DBM
        EMITTER_1_LAT, ...
    """
    # Single emitter shorthand
    if os.environ.get("EMITTER_LAT") and os.environ.get("EMITTER_FREQ_HZ"):
        record = EmitterRecord(
            id=os.environ.get("EMITTER_ID", "emitter-default"),
            lat=float(os.environ["EMITTER_LAT"]),
            lon=float(os.environ.get("EMITTER_LON", "0")),
            alt_m=float(os.environ.get("EMITTER_ALT_M", "5")),
            freq_hz=int(os.environ["EMITTER_FREQ_HZ"]),
            power_dbm=float(os.environ.get("EMITTER_POWER_DBM", "10")),
        )
        _emitters[record.id] = record
        logger.info(
            "emitter.seeded_from_env",
            id=record.id,
            lat=record.lat,
            lon=record.lon,
            freq_hz=record.freq_hz,
        )

    # Indexed emitters
    idx = 0
    while True:
        lat_key = f"EMITTER_{idx}_LAT"
        freq_key = f"EMITTER_{idx}_FREQ_HZ"
        if not (os.environ.get(lat_key) and os.environ.get(freq_key)):
            break
        record = EmitterRecord(
            id=os.environ.get(f"EMITTER_{idx}_ID", f"emitter-{idx}"),
            lat=float(os.environ[lat_key]),
            lon=float(os.environ.get(f"EMITTER_{idx}_LON", "0")),
            alt_m=float(os.environ.get(f"EMITTER_{idx}_ALT_M", "5")),
            freq_hz=int(os.environ[freq_key]),
            power_dbm=float(os.environ.get(f"EMITTER_{idx}_POWER_DBM", "10")),
        )
        _emitters[record.id] = record
        logger.info(
            "emitter.seeded_from_env",
            id=record.id,
            lat=record.lat,
            lon=record.lon,
            freq_hz=record.freq_hz,
        )
        idx += 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    _seed_emitters_from_env()
    _load_buildings_from_terrain()
    yield


app = FastAPI(title="lunchfork sim-engine", version="0.1.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Propagation constants
# ---------------------------------------------------------------------------

SPEED_OF_LIGHT = 2.998e8  # m/s
DEFAULT_TX_GAIN_DBI = 2.15
DEFAULT_RX_GAIN_DBI = 0.0
ATT_PER_BUILDING_DB = 10.0      # dB per building wall crossing
BUILDING_PROXIMITY_M = 25.0     # centroid within this distance of path → wall crossed
ATT_MAX_BUILDING_DB = 30.0      # physical cap: signal diffracts even in deep NLOS

# Building centroids loaded at startup: [N, 2] float32 (lat, lon)
_building_centroids: np.ndarray | None = None


def _load_buildings_from_terrain() -> None:
    """Load OSM building centroids from terrain cache into module-level array."""
    global _building_centroids
    terrain_dir = os.environ.get("TERRAIN_CACHE_DIR", "")
    if not terrain_dir:
        return
    osm_dir = Path(terrain_dir) / "osm"
    if not osm_dir.exists():
        return
    # Pick the first buildings geojson available
    geojson_files = sorted(osm_dir.glob("buildings_*.geojson"))
    if not geojson_files:
        return
    try:
        data = json.loads(geojson_files[0].read_text())
        elements = data.get("elements", [])
        nodes_by_id: dict[int, tuple[float, float]] = {
            e["id"]: (e["lat"], e["lon"])
            for e in elements if e.get("type") == "node" and "lat" in e
        }
        centroids = []
        for way in elements:
            if way.get("type") != "way":
                continue
            pts = [nodes_by_id[n] for n in way.get("nodes", []) if n in nodes_by_id]
            if len(pts) >= 3:
                lats = [p[0] for p in pts]
                lons = [p[1] for p in pts]
                centroids.append((sum(lats) / len(lats), sum(lons) / len(lons)))
        if centroids:
            _building_centroids = np.array(centroids, dtype=np.float32)
            logger.info("buildings.loaded", n=len(centroids), file=str(geojson_files[0]))
    except Exception as exc:
        logger.warning("buildings.load_failed", error=str(exc))


def building_attenuation_scalar(
    emitter_lat: float, emitter_lon: float,
    sensor_lat: float, sensor_lon: float,
    emitter_alt_m: float = 5.0,
    sensor_alt_m: float = 0.0,
) -> float:
    """
    Compute building wall attenuation (dB) on the emitter→sensor path.

    Counts OSM building centroids within BUILDING_PROXIMITY_M of the horizontal
    projection of the path, weighted by cos(elevation_angle) to reduce attenuation
    for near-vertical paths (e.g., ground emitter → high-altitude UAV).
    """
    if _building_centroids is None or len(_building_centroids) == 0:
        return 0.0

    # If either endpoint is clearly above rooftop height (≥30m MSL used as a proxy
    # for elevated terrain / airborne), the path clears buildings → no attenuation.
    # This avoids spurious attenuation for hill-mounted sensors or UAV paths.
    if emitter_alt_m >= 30.0 or sensor_alt_m >= 30.0:
        return 0.0

    # Approximate equal-area projection: scale lon by cos(mean_lat)
    cos_lat = math.cos(math.radians((emitter_lat + sensor_lat) / 2.0))
    p1 = np.array([emitter_lat, emitter_lon * cos_lat], dtype=np.float64)
    p2 = np.array([sensor_lat, sensor_lon * cos_lat], dtype=np.float64)
    v = p2 - p1
    v_len2 = float(np.dot(v, v))
    if v_len2 < 1e-30:
        return 0.0

    cents = _building_centroids.astype(np.float64)
    cents[:, 1] *= cos_lat  # scale lon

    # Projection of each centroid onto segment, clamped to [0, 1]
    w = cents - p1[np.newaxis, :]       # [N, 2]
    t = w @ v / v_len2                  # [N]
    t = np.clip(t, 0.0, 1.0)

    # Nearest point on segment for each centroid
    nearest = p1[np.newaxis, :] + t[:, np.newaxis] * v[np.newaxis, :]  # [N, 2]

    # Distance from centroid to nearest point (deg → m: 1° ≈ 111 km)
    diff = cents - nearest
    dist_m = np.sqrt(np.sum(diff ** 2, axis=1)) * 111_000.0

    n_buildings = int(np.sum(dist_m < BUILDING_PROXIMITY_M))
    raw_att = min(n_buildings * ATT_PER_BUILDING_DB, ATT_MAX_BUILDING_DB)

    # Scale by cos(elevation_angle): 1.0 for horizontal paths, → 0 for vertical.
    # This reduces attenuation for UAV paths that are mostly aerial.
    d_horiz_m = math.sqrt(
        (emitter_lat - sensor_lat) ** 2 + ((emitter_lon - sensor_lon) * cos_lat) ** 2
    ) * 111_000.0
    d_vert_m = abs(sensor_alt_m - emitter_alt_m)
    elev_factor = math.cos(math.atan2(d_vert_m, max(d_horiz_m, 1.0)))

    return raw_att * elev_factor


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class Position(BaseModel):
    lat: float
    lon: float
    alt_m: float = 0.0


class RssiRequest(BaseModel):
    freq_hz: float = Field(..., gt=0)
    emitter: Position
    sensor: Position
    power_dbm: float = 10.0
    tx_gain_dbi: float = DEFAULT_TX_GAIN_DBI
    rx_gain_dbi: float = DEFAULT_RX_GAIN_DBI


class RssiResponse(BaseModel):
    rssi_dbm: float
    is_nlos: bool
    path_loss_db: float
    distance_m: float


class RadioMapRequest(BaseModel):
    freq_hz: float = Field(..., gt=0)
    emitter: Position
    bbox: dict = Field(
        ..., description="{lat_min, lon_min, lat_max, lon_max}"
    )
    resolution_px: int = Field(64, ge=8, le=1024)
    power_dbm: float = 10.0


class RadioMapResponse(BaseModel):
    radiomap: list[list[float]]
    bbox: dict
    resolution_m_per_px: float
    crs: str = "WGS84"


class EmitterCreate(BaseModel):
    id: str | None = None
    lat: float
    lon: float
    alt_m: float = 0.0
    freq_hz: int
    power_dbm: float = 10.0


class EmitterRecord(BaseModel):
    id: str
    lat: float
    lon: float
    alt_m: float
    freq_hz: int
    power_dbm: float


# ---------------------------------------------------------------------------
# In-memory emitter registry
# ---------------------------------------------------------------------------

_emitters: dict[str, EmitterRecord] = {}

# Cached GeoPreprocessor instance — avoids re-reading SRTM .hgt files on every NLOS query
_geo_preprocessor: "GeoPreprocessor | None" = None

def _get_geo() -> "GeoPreprocessor | None":
    """Return a shared GeoPreprocessor instance, or None if terrain dir is unset."""
    global _geo_preprocessor
    terrain_dir = os.environ.get("TERRAIN_CACHE_DIR")
    if terrain_dir is None:
        return None
    if _geo_preprocessor is None:
        try:
            from shared.geo import GeoPreprocessor as _GP
            _geo_preprocessor = _GP(terrain_dir)
        except Exception:
            return None
    return _geo_preprocessor


# ---------------------------------------------------------------------------
# Propagation models
# ---------------------------------------------------------------------------


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in metres."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fspl_db(distance_m: float, freq_hz: float) -> float:
    """Free-Space Path Loss in dB. Returns 0 dB for distance < 1m."""
    if distance_m < 1.0:
        distance_m = 1.0
    return 20 * math.log10(distance_m) + 20 * math.log10(freq_hz) - 20 * math.log10(
        SPEED_OF_LIGHT / (4 * math.pi)
    )


def friis_rssi(
    emitter: Position,
    sensor: Position,
    freq_hz: float,
    power_dbm: float,
    tx_gain_dbi: float = DEFAULT_TX_GAIN_DBI,
    rx_gain_dbi: float = DEFAULT_RX_GAIN_DBI,
) -> tuple[float, float, bool]:
    """
    Compute RSSI using Friis free-space equation.

    Returns (rssi_dbm, path_loss_db, is_nlos=False).
    """
    # Horizontal distance
    d_horiz = haversine_m(emitter.lat, emitter.lon, sensor.lat, sensor.lon)
    # 3D distance including altitude difference
    dz = sensor.alt_m - emitter.alt_m
    distance_m = math.sqrt(d_horiz**2 + dz**2)

    pl = fspl_db(distance_m, freq_hz)
    rssi = power_dbm + tx_gain_dbi + rx_gain_dbi - pl
    return rssi, pl, False


# ---------------------------------------------------------------------------
# Simplified ITM (Irregular Terrain Model) — Longley-Rice approximation
# ---------------------------------------------------------------------------


def _effective_earth_radius() -> float:
    """Effective Earth radius for standard atmosphere refractivity."""
    return 6371000.0 * (4.0 / 3.0)


def itm_rssi(
    emitter: Position,
    sensor: Position,
    freq_hz: float,
    power_dbm: float,
    tx_gain_dbi: float = DEFAULT_TX_GAIN_DBI,
    rx_gain_dbi: float = DEFAULT_RX_GAIN_DBI,
    terrain_cache_dir: str | None = None,
) -> tuple[float, float, bool]:
    """
    Simplified ITM propagation model.

    Uses Friis free-space loss plus empirical corrections:
    - Distance-dependent excess loss beyond free-space (ground-wave / diffraction)
    - Knife-edge diffraction penalty if terrain data available
    - Additional NLOS penalty based on height geometry

    Returns (rssi_dbm, path_loss_db, is_nlos).
    """
    d_horiz = haversine_m(emitter.lat, emitter.lon, sensor.lat, sensor.lon)
    dz = sensor.alt_m - emitter.alt_m
    distance_m = math.sqrt(d_horiz**2 + dz**2)
    freq_mhz = freq_hz / 1e6

    # Free-space path loss
    pl_fs = fspl_db(distance_m, freq_hz)

    # ITM excess loss beyond free-space (empirical approximation for VHF/UHF outdoor)
    # Based on Longley-Rice median attenuation ratio curves (continental temperate)
    climate = int(os.environ.get("ITM_CLIMATE", "5"))  # 5 = continental temperate
    pl_excess = _itm_excess_loss(distance_m, freq_mhz, climate)

    # Basic NLOS detection: compare effective antenna heights
    # LOS criterion: tx_ht + rx_ht > 4/3 * R_earth * d^2 / (2*R_eff)  (horizon)
    is_nlos = _is_nlos_heuristic(emitter, sensor, distance_m)

    # Two-ray ground reflection excess (d⁴ regime for near-ground links)
    pl_tworay = _two_ray_excess_db(
        max(emitter.alt_m, 1.0), max(sensor.alt_m, 1.0), distance_m, freq_hz
    )

    # Add diffraction loss for NLOS
    pl_diffraction = 0.0
    if is_nlos:
        pl_diffraction = _knife_edge_diffraction_db(
            emitter, sensor, distance_m, freq_hz, terrain_cache_dir
        )

    pl_total = pl_fs + pl_excess + pl_tworay + pl_diffraction
    rssi = power_dbm + tx_gain_dbi + rx_gain_dbi - pl_total

    return rssi, pl_total, is_nlos


def _itm_excess_loss(distance_m: float, freq_mhz: float, climate: int) -> float:
    """
    Empirical excess attenuation beyond free-space for ITM/Longley-Rice.

    Approximates median basic transmission loss curves for continental temperate (climate=5).
    Valid for VHF/UHF (100–1000 MHz), distances 1–1000 km.
    """
    if distance_m < 1000:
        return 0.0

    d_km = distance_m / 1000.0

    # Empirical fit to L-R curves (continental temperate, median conditions)
    # Loss increases roughly as d^n beyond line-of-sight horizon
    if freq_mhz < 300:  # VHF
        n = 3.5
        k = 0.8
    else:  # UHF
        n = 3.0
        k = 0.6

    # Reference distance where excess begins (typically ~10–50 km for ground paths)
    d0 = 30.0
    if d_km > d0:
        excess = k * n * 10 * math.log10(d_km / d0)
    else:
        excess = 0.0

    return max(0.0, excess)


def _two_ray_excess_db(
    h_tx: float, h_rx: float, distance_m: float, freq_hz: float
) -> float:
    """
    Two-ray ground reflection excess path loss over free-space.

    Beyond crossover distance d_cross = 4π·h_tx·h_rx/λ, the ground
    reflection causes path loss to scale as d⁴ (20 dB/decade over Friis).

    Only applied when both antennas are near-ground (< 50 m); airborne links
    are dominated by free-space propagation and have a d_cross that exceeds
    typical scene sizes anyway.

    Capped at 25 dB to account for imperfect ground reflectivity.
    """
    if h_tx > 50.0 or h_rx > 50.0:
        return 0.0
    if distance_m < 100.0:
        return 0.0
    wavelength = 3e8 / freq_hz
    h_tx_eff = max(h_tx, 1.0)
    h_rx_eff = max(h_rx, 1.0)
    d_cross = (4.0 * math.pi * h_tx_eff * h_rx_eff) / wavelength
    if distance_m <= d_cross:
        return 0.0
    return min(25.0, 20.0 * math.log10(distance_m / d_cross))


def _is_nlos_heuristic(
    emitter: Position, sensor: Position, distance_m: float
) -> bool:
    """
    Basic LOS check based on antenna heights and Earth curvature.
    """
    # Height of both antennas above ground (using alt_m directly)
    h_tx = max(emitter.alt_m, 1.0)
    h_rx = max(sensor.alt_m, 1.0)
    r_eff = _effective_earth_radius()

    # Maximum LOS distance (simplified, ignoring terrain)
    d_los_m = math.sqrt(2 * r_eff * h_tx) + math.sqrt(2 * r_eff * h_rx)

    return distance_m > d_los_m


def _knife_edge_diffraction_db(
    emitter: Position,
    sensor: Position,
    distance_m: float,
    freq_hz: float,
    terrain_cache_dir: str | None,
) -> float:
    """
    Simplified knife-edge diffraction loss for NLOS scenarios.

    Uses a Fresnel-zone obstruction model. Without terrain data,
    applies a distance-dependent heuristic based on over-horizon geometry.
    """
    if terrain_cache_dir is None:
        # Heuristic: additional loss proportional to over-horizon distance
        h_tx = max(emitter.alt_m, 1.0)
        h_rx = max(sensor.alt_m, 1.0)
        r_eff = _effective_earth_radius()
        d_los_m = math.sqrt(2 * r_eff * h_tx) + math.sqrt(2 * r_eff * h_rx)
        over_horizon_m = max(0.0, distance_m - d_los_m)
        # Rough approximation: ~0.2 dB per km beyond horizon
        return 0.2 * over_horizon_m / 1000.0

    # With terrain: attempt to read DEM and compute knife-edge parameter
    try:
        from shared.geo import BBox

        center_lat = (emitter.lat + sensor.lat) / 2
        center_lon = (emitter.lon + sensor.lon) / 2
        bbox = BBox.from_center(center_lat, center_lon, radius_m=distance_m * 0.6)
        geo = _get_geo()
        if geo is None:
            return 0.0
        dem = geo._get_dem(bbox, resolution_px=64)
        if dem.max() == 0.0:
            return 0.0

        # Find the maximum terrain obstruction along the path
        n_samples = 64
        lats = np.linspace(emitter.lat, sensor.lat, n_samples)
        lons = np.linspace(emitter.lon, sensor.lon, n_samples)
        # Map lat/lon to DEM pixel coordinates
        h_px = dem.shape[0]
        w_px = dem.shape[1]
        row_idx = ((bbox.lat_max - lats) / bbox.height_deg * h_px).clip(0, h_px - 1).astype(int)
        col_idx = ((lons - bbox.lon_min) / bbox.width_deg * w_px).clip(0, w_px - 1).astype(int)
        terrain_heights_norm = dem[row_idx, col_idx]  # normalised 0–1

        # Convert normalised height back to approximate metres (very rough)
        # Without actual elevation range, use heuristic max 500m
        terrain_heights_m = terrain_heights_norm * 500.0

        # LOS heights along path
        los_heights = np.linspace(emitter.alt_m, sensor.alt_m, n_samples)
        clearance = los_heights - terrain_heights_m

        min_clearance_m = clearance.min()
        if min_clearance_m > 0:
            return 0.0  # LOS

        # Fresnel zone radius at obstruction
        wavelength_m = SPEED_OF_LIGHT / freq_hz
        d_obs = distance_m / 2.0
        r1 = math.sqrt(wavelength_m * d_obs * (distance_m - d_obs) / distance_m)

        # Fresnel-Kirchhoff diffraction parameter v
        obstruction_m = abs(min_clearance_m)
        v = obstruction_m / (r1 + 1e-9)

        # Knife-edge diffraction loss (ITU-R P.526)
        if v < -0.7:
            return 0.0
        elif v < 0:
            return 0.0
        else:
            loss = 6.9 + 20 * math.log10(math.sqrt((v - 0.1) ** 2 + 1) + v - 0.1)
        return max(0.0, loss)

    except Exception as exc:
        logger.warning("itm.diffraction_failed", error=str(exc))
        return 0.0


def compute_rssi(
    emitter: Position,
    sensor: Position,
    freq_hz: float,
    power_dbm: float = 10.0,
    tx_gain_dbi: float = DEFAULT_TX_GAIN_DBI,
    rx_gain_dbi: float = DEFAULT_RX_GAIN_DBI,
) -> tuple[float, float, bool]:
    """Dispatch to the configured propagation model, then apply building attenuation."""
    model = os.environ.get("PROPAGATION_MODEL", "friis").lower()
    terrain_dir = os.environ.get("TERRAIN_CACHE_DIR", None)

    if model == "friis":
        rssi, pl, nlos = friis_rssi(emitter, sensor, freq_hz, power_dbm, tx_gain_dbi, rx_gain_dbi)
    elif model == "itm":
        rssi, pl, nlos = itm_rssi(emitter, sensor, freq_hz, power_dbm, tx_gain_dbi, rx_gain_dbi, terrain_dir)
    else:
        logger.warning("propagation.unknown_model", model=model, fallback="friis")
        rssi, pl, nlos = friis_rssi(emitter, sensor, freq_hz, power_dbm, tx_gain_dbi, rx_gain_dbi)

    # Apply building wall attenuation when OSM data is loaded
    batt = building_attenuation_scalar(
        emitter.lat, emitter.lon, sensor.lat, sensor.lon,
        emitter_alt_m=emitter.alt_m, sensor_alt_m=sensor.alt_m,
    )
    if batt > 0.0:
        rssi -= batt
        pl += batt
        nlos = True

    return rssi, pl, nlos


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    model = os.environ.get("PROPAGATION_MODEL", "friis")
    return {"status": "ok", "propagation_model": model, "n_emitters": len(_emitters)}


@app.post("/rssi", response_model=RssiResponse)
async def post_rssi(req: RssiRequest) -> RssiResponse:
    rssi_dbm, path_loss_db, is_nlos = compute_rssi(
        req.emitter,
        req.sensor,
        req.freq_hz,
        req.power_dbm,
        req.tx_gain_dbi,
        req.rx_gain_dbi,
    )
    d = haversine_m(req.emitter.lat, req.emitter.lon, req.sensor.lat, req.sensor.lon)
    logger.info(
        "rssi.computed",
        rssi_dbm=round(rssi_dbm, 1),
        path_loss_db=round(path_loss_db, 1),
        is_nlos=is_nlos,
        distance_m=round(d, 0),
    )
    return RssiResponse(
        rssi_dbm=rssi_dbm,
        is_nlos=is_nlos,
        path_loss_db=path_loss_db,
        distance_m=d,
    )


@app.post("/radiomap", response_model=RadioMapResponse)
async def post_radiomap(req: RadioMapRequest) -> RadioMapResponse:
    bbox = req.bbox
    lat_min = bbox["lat_min"]
    lon_min = bbox["lon_min"]
    lat_max = bbox["lat_max"]
    lon_max = bbox["lon_max"]
    n = req.resolution_px

    # Compute grid resolution in metres
    d_lat_m = haversine_m(lat_min, lon_min, lat_max, lon_min)
    d_lon_m = haversine_m(lat_min, lon_min, lat_min, lon_max)
    resolution_m = max(d_lat_m, d_lon_m) / n

    lats = np.linspace(lat_max, lat_min, n)  # north to south
    lons = np.linspace(lon_min, lon_max, n)  # west to east

    radiomap = np.zeros((n, n), dtype=np.float32)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            sensor = Position(lat=float(lat), lon=float(lon), alt_m=1.5)
            rssi_dbm, _, _ = compute_rssi(
                req.emitter, sensor, req.freq_hz, req.power_dbm
            )
            radiomap[i, j] = rssi_dbm

    logger.info(
        "radiomap.generated",
        resolution_px=n,
        resolution_m=round(resolution_m, 1),
        rssi_min=round(float(radiomap.min()), 1),
        rssi_max=round(float(radiomap.max()), 1),
    )

    return RadioMapResponse(
        radiomap=radiomap.tolist(),
        bbox=bbox,
        resolution_m_per_px=resolution_m,
        crs="WGS84",
    )


@app.post("/emitter", response_model=EmitterRecord)
async def create_emitter(req: EmitterCreate) -> EmitterRecord:
    emitter_id = req.id or f"emitter-{uuid.uuid4().hex[:8]}"
    record = EmitterRecord(
        id=emitter_id,
        lat=req.lat,
        lon=req.lon,
        alt_m=req.alt_m,
        freq_hz=req.freq_hz,
        power_dbm=req.power_dbm,
    )
    _emitters[emitter_id] = record
    logger.info("emitter.created", id=emitter_id, lat=req.lat, lon=req.lon, freq_hz=req.freq_hz)
    return record


@app.delete("/emitter/{emitter_id}")
async def delete_emitter(emitter_id: str) -> dict:
    if emitter_id not in _emitters:
        raise HTTPException(status_code=404, detail=f"Emitter {emitter_id!r} not found")
    del _emitters[emitter_id]
    logger.info("emitter.deleted", id=emitter_id)
    return {"deleted": emitter_id}


@app.get("/emitters", response_model=list[EmitterRecord])
async def list_emitters() -> list[EmitterRecord]:
    return list(_emitters.values())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log_level = os.environ.get("LOG_LEVEL", "INFO").lower()
    logger.info(
        "sim_engine.starting",
        propagation_model=os.environ.get("PROPAGATION_MODEL", "friis"),
    )
    n_workers = int(os.environ.get("UVICORN_WORKERS", "1"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9000,
        log_level=log_level,
        workers=n_workers,
    )
