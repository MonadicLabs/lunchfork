"""
demo_buildings.py — Demonstrate building attenuation effect on localisation.

Simulates an urban scenario with 4 ground-level sensors (~3m MSL) and a
ground emitter (~3m MSL).  Compares GridLikelihoodModel accuracy:
  - WITHOUT building model (pure Friis)
  - WITH building model (Friis + OSM building attenuation)

Usage:
    python scripts/demo_buildings.py [--terrain-cache data/terrain]

No running services required.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.models.grid_likelihood import (
    GridLikelihoodModel,
    RssiObservation,
    load_building_centroids,
    rasterize_building_centroids,
    make_lat_lon_grid,
)

# ---------------------------------------------------------------------------
# Scenario definition — urban street-level sensors around an emitter
# ---------------------------------------------------------------------------

EMITTER_LAT = 43.530
EMITTER_LON = 5.450
EMITTER_ALT_M = 3.0
EMITTER_FREQ_HZ = 433_920_000
EMITTER_POWER_DBM = 27.0   # 500 mW — legal in France on 433 MHz amateur/ISM

# 4 sensors within ~700m of the emitter, at street level (3m MSL)
SENSORS = [
    ("north",  43.536, 5.450, 3.0),
    ("south",  43.524, 5.450, 3.0),
    ("east",   43.530, 5.459, 3.0),
    ("west",   43.530, 5.441, 3.0),
]

SPEED_OF_LIGHT = 2.998e8
ATT_PER_BUILDING_DB = 10.0
ATT_MAX_BUILDING_DB = 30.0
BUILDING_PROXIMITY_M = 25.0

GRID_SIZE = 256
SEARCH_RADIUS_M = 3000  # 3km search area


# ---------------------------------------------------------------------------
# Forward model: compute RSSI with building attenuation (scalar)
# ---------------------------------------------------------------------------


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fspl_db(d_m: float, freq_hz: float) -> float:
    if d_m < 1.0:
        d_m = 1.0
    return 20 * math.log10(d_m) + 20 * math.log10(freq_hz) - 147.55


def building_attenuation_scalar(
    emitter_lat: float, emitter_lon: float, emitter_alt_m: float,
    sensor_lat: float, sensor_lon: float, sensor_alt_m: float,
    building_centroids: np.ndarray | None,
) -> float:
    if building_centroids is None or len(building_centroids) == 0:
        return 0.0
    if emitter_alt_m >= 30.0 or sensor_alt_m >= 30.0:
        return 0.0

    cos_lat = math.cos(math.radians((emitter_lat + sensor_lat) / 2.0))
    p1 = np.array([emitter_lat, emitter_lon * cos_lat], dtype=np.float64)
    p2 = np.array([sensor_lat, sensor_lon * cos_lat], dtype=np.float64)
    v = p2 - p1
    v_len2 = float(np.dot(v, v))
    if v_len2 < 1e-30:
        return 0.0

    cents = building_centroids.astype(np.float64)
    cents[:, 1] *= cos_lat
    w = cents - p1[np.newaxis, :]
    t = np.clip(w @ v / v_len2, 0.0, 1.0)
    nearest = p1[np.newaxis, :] + t[:, np.newaxis] * v[np.newaxis, :]
    dist_m = np.sqrt(np.sum((cents - nearest) ** 2, axis=1)) * 111_000.0

    n_buildings = int(np.sum(dist_m < BUILDING_PROXIMITY_M))
    raw_att = min(n_buildings * ATT_PER_BUILDING_DB, ATT_MAX_BUILDING_DB)

    d_horiz_m = math.sqrt(
        (emitter_lat - sensor_lat) ** 2 + ((emitter_lon - sensor_lon) * cos_lat) ** 2
    ) * 111_000.0
    d_vert_m = abs(sensor_alt_m - emitter_alt_m)
    elev_factor = math.cos(math.atan2(d_vert_m, max(d_horiz_m, 1.0)))
    return raw_att * elev_factor


def simulate_rssi(
    sensor_lat, sensor_lon, sensor_alt_m,
    building_centroids: np.ndarray | None,
) -> float:
    d_horiz = haversine_m(EMITTER_LAT, EMITTER_LON, sensor_lat, sensor_lon)
    d_vert = sensor_alt_m - EMITTER_ALT_M
    d_3d = math.sqrt(d_horiz ** 2 + d_vert ** 2)
    rssi = EMITTER_POWER_DBM + 2.15 + 0.0 - fspl_db(d_3d, EMITTER_FREQ_HZ)
    batt = building_attenuation_scalar(
        EMITTER_LAT, EMITTER_LON, EMITTER_ALT_M,
        sensor_lat, sensor_lon, sensor_alt_m,
        building_centroids,
    )
    return rssi - batt


# ---------------------------------------------------------------------------
# Peak localisation from radiomap
# ---------------------------------------------------------------------------


def radiomap_peak_position(
    radiomap: np.ndarray,  # [1, H, W]
    bbox_dict: dict,
) -> tuple[float, float]:
    rm = radiomap[0]
    idx = np.unravel_index(np.argmax(rm), rm.shape)
    row, col = idx
    H, W = rm.shape
    lat = bbox_dict["lat_max"] - row / (H - 1) * (bbox_dict["lat_max"] - bbox_dict["lat_min"])
    lon = bbox_dict["lon_min"] + col / (W - 1) * (bbox_dict["lon_max"] - bbox_dict["lon_min"])
    return lat, lon


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain-cache", default="data/terrain")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Monte Carlo trials with random RSSI noise")
    parser.add_argument("--noise-db", type=float, default=4.0,
                        help="Gaussian measurement noise std (dB)")
    args = parser.parse_args()

    terrain_cache = Path(args.terrain_cache)

    # Build search bbox around emitter
    deg_per_m = 1.0 / 111_000.0
    bbox_dict = {
        "lat_min": EMITTER_LAT - SEARCH_RADIUS_M * deg_per_m,
        "lat_max": EMITTER_LAT + SEARCH_RADIUS_M * deg_per_m,
        "lon_min": EMITTER_LON - SEARCH_RADIUS_M * deg_per_m,
        "lon_max": EMITTER_LON + SEARCH_RADIUS_M * deg_per_m,
    }

    # Load buildings
    print(f"Loading buildings from {terrain_cache} …")
    building_centroids = load_building_centroids(terrain_cache, bbox_dict)
    if building_centroids is None:
        print("  No building data found — run: python scripts/fetch_terrain.py "
              "--source osm-buildings --bbox 5.40 43.46 5.52 43.60")
        print("  Running without buildings for comparison only.")
    else:
        print(f"  Loaded {len(building_centroids)} building centroids.")

    lat_grid, lon_grid = make_lat_lon_grid(bbox_dict, GRID_SIZE)
    if building_centroids is not None:
        bm = rasterize_building_centroids(building_centroids, lat_grid, lon_grid)
        n_building_pixels = int(bm.sum())
        print(f"  Building mask: {n_building_pixels} occupied pixels / {GRID_SIZE}² total")

    # Models
    model_friis = GridLikelihoodModel(terrain_cache_dir=None)
    model_buildings = GridLikelihoodModel(terrain_cache_dir=terrain_cache if building_centroids is not None else None)

    rng = np.random.default_rng(42)

    # Print ideal (noiseless) RSSI per sensor
    print(f"\n{'Sensor':12} {'Dist (m)':>9} {'Friis RSSI':>12} {'w/ Buildings':>14}")
    for name, slat, slon, salt in SENSORS:
        d = haversine_m(EMITTER_LAT, EMITTER_LON, slat, slon)
        rssi_friis = simulate_rssi(slat, slon, salt, None)
        rssi_batt  = simulate_rssi(slat, slon, salt, building_centroids)
        print(f"  {name:10} {d:9.0f}m {rssi_friis:10.1f} dBm {rssi_batt:12.1f} dBm")

    errors_friis = []
    errors_buildings = []

    for trial in range(args.n_trials):
        obs_friis = []
        obs_buildings = []
        for _name, slat, slon, salt in SENSORS:
            noise = float(rng.normal(0, args.noise_db))
            rssi_f = simulate_rssi(slat, slon, salt, None) + noise
            rssi_b = simulate_rssi(slat, slon, salt, building_centroids) + noise

            obs_friis.append(RssiObservation(
                sensor_lat=slat, sensor_lon=slon, sensor_alt_m=salt,
                rssi_dbm=rssi_f, freq_hz=float(EMITTER_FREQ_HZ),
            ))
            obs_buildings.append(RssiObservation(
                sensor_lat=slat, sensor_lon=slon, sensor_alt_m=salt,
                rssi_dbm=rssi_b, freq_hz=float(EMITTER_FREQ_HZ),
            ))

        # Localise with Friis-only model (Friis observations, Friis inverse)
        rm_f = model_friis.infer_from_observations(obs_friis, bbox_dict, GRID_SIZE)
        lat_f, lon_f = radiomap_peak_position(rm_f, bbox_dict)
        errors_friis.append(haversine_m(EMITTER_LAT, EMITTER_LON, lat_f, lon_f))

        if building_centroids is not None:
            # Localise with buildings model (buildings observations, buildings inverse)
            rm_b = model_buildings.infer_from_observations(obs_buildings, bbox_dict, GRID_SIZE)
            lat_b, lon_b = radiomap_peak_position(rm_b, bbox_dict)
            errors_buildings.append(haversine_m(EMITTER_LAT, EMITTER_LON, lat_b, lon_b))

    def cep(errors, p):
        return float(np.percentile(errors, p * 100))

    print(f"\n=== Results ({args.n_trials} trials, noise σ={args.noise_db} dB) ===")
    print(f"{'Model':30} {'RMSE':>8} {'CEP50':>8} {'CEP90':>8}")
    e_f = np.array(errors_friis)
    print(f"  {'Friis (no buildings)':28} {np.sqrt(np.mean(e_f**2)):>7.0f}m "
          f"{cep(e_f, 0.5):>7.0f}m {cep(e_f, 0.9):>7.0f}m")
    if errors_buildings:
        e_b = np.array(errors_buildings)
        print(f"  {'Friis + buildings':28} {np.sqrt(np.mean(e_b**2)):>7.0f}m "
              f"{cep(e_b, 0.5):>7.0f}m {cep(e_b, 0.9):>7.0f}m")
        delta = cep(e_f, 0.5) - cep(e_b, 0.5)
        print(f"\n  CEP50 improvement with buildings: {delta:+.0f}m")

    # Also show: what if WRONG model (buildings in sim but Friis in inverse)?
    if building_centroids is not None:
        errors_mismatch = []
        for trial in range(args.n_trials):
            obs = []
            for _name, slat, slon, salt in SENSORS:
                noise = float(rng.normal(0, args.noise_db))
                rssi_b = simulate_rssi(slat, slon, salt, building_centroids) + noise
                obs.append(RssiObservation(
                    sensor_lat=slat, sensor_lon=slon, sensor_alt_m=salt,
                    rssi_dbm=rssi_b, freq_hz=float(EMITTER_FREQ_HZ),
                ))
            rm = model_friis.infer_from_observations(obs, bbox_dict, GRID_SIZE)
            lat_e, lon_e = radiomap_peak_position(rm, bbox_dict)
            errors_mismatch.append(haversine_m(EMITTER_LAT, EMITTER_LON, lat_e, lon_e))

        e_m = np.array(errors_mismatch)
        print(f"  {'Friis inverse / buildings fwd (mismatch)':28} "
              f"{np.sqrt(np.mean(e_m**2)):>7.0f}m "
              f"{cep(e_m, 0.5):>7.0f}m {cep(e_m, 0.9):>7.0f}m")
        print(f"\n  Model mismatch CEP50 penalty: "
              f"{cep(e_m, 0.5) - cep(e_b, 0.5):+.0f}m")


if __name__ == "__main__":
    main()
