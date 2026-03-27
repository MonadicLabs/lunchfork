#!/usr/bin/env python3
"""
training/generate_dataset.py — Generate synthetic radio map dataset.

For each scene:
1. Sample a random terrain zone and bbox
2. Load terrain conditioning tensor via GeoPreprocessor
3. Place a random emitter within bbox
4. Query sim-engine for ground truth radio map (used as training label)
5. Sample N_sensors random sensor positions within bbox
6. Query sim-engine for RSSI at each sensor → build sparse observation map (model input)
7. Save as .npz

The sim-engine (ITM or Friis) is the propagation simulator — equivalent of the
"raycasting sim" needed for training. Start it with docker compose before running.

Usage:
  # With docker compose running:
  python training/generate_dataset.py \\
    --n-scenes 5000 --freq-range 400e6 500e6 \\
    --terrain-zones france \\
    --propagation-model itm \\
    --output data/datasets/uhf_433_v1/

  # Quick smoke test (Friis, no terrain needed):
  python training/generate_dataset.py \\
    --n-scenes 100 --propagation-model friis \\
    --output data/datasets/smoke_test/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.geo import BBox, GeoPreprocessor


# ---------------------------------------------------------------------------
# Terrain zones
# ---------------------------------------------------------------------------

TERRAIN_ZONES: dict[str, BBox] = {
    "france":   BBox(lat_min=43.0, lon_min=4.0,  lat_max=49.0, lon_max=8.0),
    "benelux":  BBox(lat_min=49.5, lon_min=2.5,  lat_max=53.5, lon_max=7.0),
    "iberia":   BBox(lat_min=36.0, lon_min=-9.0, lat_max=43.5, lon_max=3.0),
    "alps":     BBox(lat_min=44.0, lon_min=5.0,  lat_max=48.0, lon_max=15.0),
    "provence": BBox(lat_min=43.2, lon_min=4.5,  lat_max=44.2, lon_max=6.5),
}

# RSSI normalisation constants (dBm)
RSSI_FLOOR_DBM = -120.0
RSSI_RANGE_DB  =   80.0   # −120 → −40 dBm spans [0, 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rssi_to_norm(rssi_dbm: float) -> float:
    """Normalise RSSI to [0, 1]. Clips to [RSSI_FLOOR, RSSI_FLOOR+RANGE]."""
    return float(np.clip((rssi_dbm - RSSI_FLOOR_DBM) / RSSI_RANGE_DB, 0.0, 1.0))


def random_bbox_within(zone: BBox, size_km: float = 20.0) -> BBox:
    """Sample a random bbox of ~size_km within a zone."""
    size_deg_lat = size_km / 111.32
    size_deg_lon = size_km / (111.32 * math.cos(math.radians(
        (zone.lat_min + zone.lat_max) / 2
    )))
    lat_min = random.uniform(zone.lat_min, zone.lat_max - size_deg_lat)
    lon_min = random.uniform(zone.lon_min, zone.lon_max - size_deg_lon)
    return BBox(
        lat_min=lat_min,
        lon_min=lon_min,
        lat_max=lat_min + size_deg_lat,
        lon_max=lon_min + size_deg_lon,
    )


def random_emitter_in_bbox(bbox: BBox, alt_range_m: tuple = (2, 30)) -> dict:
    return {
        "lat":   random.uniform(bbox.lat_min, bbox.lat_max),
        "lon":   random.uniform(bbox.lon_min, bbox.lon_max),
        "alt_m": random.uniform(*alt_range_m),
    }


def _post_json(url: str, payload: dict, timeout: int = 30) -> dict | None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        print(f"  WARN: POST {url} failed: {exc}")
        return None


def query_radiomap(
    simengine_url: str,
    emitter: dict,
    bbox: BBox,
    freq_hz: float,
    power_dbm: float,
    resolution_px: int,
) -> np.ndarray | None:
    """Query sim-engine for ground truth radio map. Returns [H, W] float32 in dBm."""
    result = _post_json(f"{simengine_url}/radiomap", {
        "freq_hz":      freq_hz,
        "emitter":      emitter,
        "bbox":         bbox.as_dict(),
        "resolution_px": resolution_px,
        "power_dbm":    power_dbm,
    })
    if result is None:
        return None
    return np.array(result["radiomap"], dtype=np.float32)


def query_rssi_single(
    simengine_url: str,
    emitter: dict,
    sensor_lat: float,
    sensor_lon: float,
    sensor_alt_m: float,
    freq_hz: float,
    power_dbm: float,
) -> float | None:
    """Query sim-engine for RSSI at a single sensor position. Returns dBm."""
    result = _post_json(f"{simengine_url}/rssi", {
        "freq_hz":   freq_hz,
        "emitter":   emitter,
        "sensor":    {"lat": sensor_lat, "lon": sensor_lon, "alt_m": sensor_alt_m},
        "power_dbm": power_dbm,
    })
    if result is None:
        return None
    return float(result["rssi_dbm"])


def build_sparse_obs(
    simengine_url: str,
    emitter: dict,
    bbox: BBox,
    freq_hz: float,
    power_dbm: float,
    resolution_px: int,
    n_sensors: int,
    alt_range_m: tuple = (1.5, 200.0),
    threshold_dbm: float = -110.0,
    max_workers: int = 8,
    shadow_std_db: float = 6.0,
    hw_offset_std_db: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample n_sensors random positions, query RSSI for each in parallel, and return:
      rssi_map  [H, W] float32 — normalised RSSI at sensor positions, 0 elsewhere
      rssi_mask [H, W] float32 — 1 where a valid measurement exists, else 0

    Measurement noise model (sim-to-real):
      shadow_std_db   — per-measurement log-normal shadow fading N(0, σ²)
      hw_offset_std_db — per-sensor hardware calibration offset, constant within a sensor
    """
    # Pre-sample all sensor positions + per-sensor hardware offset
    sensors = []
    for _ in range(n_sensors):
        hw_offset = float(np.random.normal(0.0, hw_offset_std_db)) if hw_offset_std_db > 0 else 0.0
        sensors.append((
            random.uniform(bbox.lat_min, bbox.lat_max),
            random.uniform(bbox.lon_min, bbox.lon_max),
            random.uniform(*alt_range_m),
            hw_offset,
        ))

    def _query(s):
        s_lat, s_lon, s_alt, hw_off = s
        rssi = query_rssi_single(simengine_url, emitter, s_lat, s_lon, s_alt, freq_hz, power_dbm)
        if rssi is None:
            return s_lat, s_lon, None
        # Add shadow fading (per-measurement) + hardware offset (per-sensor)
        shadow = float(np.random.normal(0.0, shadow_std_db)) if shadow_std_db > 0 else 0.0
        return s_lat, s_lon, rssi + shadow + hw_off

    rssi_map  = np.zeros((resolution_px, resolution_px), dtype=np.float32)
    rssi_mask = np.zeros((resolution_px, resolution_px), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=min(max_workers, n_sensors)) as pool:
        for s_lat, s_lon, rssi in pool.map(_query, sensors):
            if rssi is None or rssi < threshold_dbm:
                continue
            row = int(np.clip((bbox.lat_max - s_lat) / bbox.height_deg * resolution_px,
                              0, resolution_px - 1))
            col = int(np.clip((s_lon - bbox.lon_min) / bbox.width_deg * resolution_px,
                              0, resolution_px - 1))
            rssi_map[row, col]  = rssi_to_norm(rssi)
            rssi_mask[row, col] = 1.0

    return rssi_map, rssi_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic lunchfork dataset")
    p.add_argument("--n-scenes",           type=int,   default=1000)
    p.add_argument("--freq-range",         nargs=2, type=float,
                   default=[400e6, 500e6], metavar=("FREQ_MIN", "FREQ_MAX"))
    p.add_argument("--terrain-zones",      default="france",
                   help="Comma-separated zone names: " + ", ".join(TERRAIN_ZONES))
    p.add_argument("--propagation-model",  default="itm", choices=["friis", "itm"])
    p.add_argument("--output",             type=Path,
                   default=ROOT / "data/datasets/vhf_uhf_outdoor_v1")
    p.add_argument("--simengine-url",      default="http://localhost:9000")
    p.add_argument("--resolution",         type=int,   default=64,
                   help="Radio map resolution in pixels (must be divisible by 16)")
    p.add_argument("--scene-size-km",      type=float, default=20.0)
    p.add_argument("--power-dbm",          type=float, default=10.0)
    p.add_argument("--n-sensors-min",      type=int,   default=3,
                   help="Min sparse RSSI observations per scene")
    p.add_argument("--n-sensors-max",      type=int,   default=25,
                   help="Max sparse RSSI observations per scene")
    p.add_argument("--sensor-alt-min",     type=float, default=1.5)
    p.add_argument("--sensor-alt-max",     type=float, default=200.0)
    p.add_argument("--power-min",          type=float, default=None,
                   help="Min TX power dBm for random range (overrides --power-dbm if set)")
    p.add_argument("--power-max",          type=float, default=None,
                   help="Max TX power dBm for random range (overrides --power-dbm if set)")
    p.add_argument("--sigma-m",            type=float, default=0.0,
                   help="If >0, save Gaussian heatmap target with σ=sigma_m instead of ITM radio map")
    p.add_argument("--shadow-std",         type=float, default=6.0,
                   help="Shadow fading std-dev (dB). Added per-measurement as N(0,σ²). "
                        "0 = deterministic sim (old behaviour).")
    p.add_argument("--hw-offset-std",      type=float, default=3.0,
                   help="Per-sensor hardware calibration offset std-dev (dB). "
                        "Drawn once per sensor, constant across its measurements. "
                        "0 = no hardware bias.")
    p.add_argument("--val-split",          type=float, default=0.1,
                   help="Fraction of scenes to put in val/ subdirectory")
    p.add_argument("--resume",             action="store_true",
                   help="Skip existing scenes")
    p.add_argument("--seed",               type=int,   default=42)
    return p.parse_args()


def _gaussian_heatmap(em_row: float, em_col: float, H: int, W: int, sigma_px: float) -> np.ndarray:
    rows = np.arange(H, dtype=np.float32)[:, None]
    cols = np.arange(W, dtype=np.float32)[None, :]
    g = np.exp(-((rows - em_row)**2 + (cols - em_col)**2) / (2 * sigma_px**2))
    return (g / (g.max() + 1e-8)).astype(np.float32)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    assert args.resolution % 16 == 0, \
        f"--resolution must be divisible by 16 (got {args.resolution})"

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "val").mkdir(exist_ok=True)

    zone_names = [z.strip() for z in args.terrain_zones.split(",")]
    zones = [TERRAIN_ZONES[z] for z in zone_names if z in TERRAIN_ZONES]
    if not zones:
        print(f"ERROR: No valid zones. Available: {list(TERRAIN_ZONES.keys())}")
        sys.exit(1)

    # Power: fixed if --power-dbm set, else random in [power_min, power_max]
    use_random_power = args.power_min is not None and args.power_max is not None
    power_str = (f"{args.power_min:.0f}–{args.power_max:.0f}dBm (random)"
                 if use_random_power else f"{args.power_dbm:.0f}dBm")

    # Gaussian heatmap target
    m_per_px = args.scene_size_km * 1000 / args.resolution
    sigma_px = args.sigma_m / m_per_px if args.sigma_m > 0 else 0.0

    terrain_dir = ROOT / "data/terrain"
    geo = GeoPreprocessor(terrain_dir)

    n_val   = max(1, int(args.n_scenes * args.val_split))
    n_train = args.n_scenes - n_val

    print(f"\n=== Dataset generation ===")
    print(f"Scenes      : {args.n_scenes}  (train={n_train}  val={n_val})")
    print(f"Resolution  : {args.resolution}px  ({m_per_px:.0f}m/px)")
    print(f"Freq range  : {args.freq_range[0]/1e6:.0f}–{args.freq_range[1]/1e6:.0f} MHz")
    print(f"Zones       : {zone_names} | Model: {args.propagation_model}")
    print(f"Sensors/scene: {args.n_sensors_min}–{args.n_sensors_max}  "
          f"alt {args.sensor_alt_min}–{args.sensor_alt_max}m  power={power_str}")
    print(f"Noise model  : shadow_std={args.shadow_std:.1f}dB  hw_offset_std={args.hw_offset_std:.1f}dB")
    if sigma_px > 0:
        print(f"Target      : gaussian heatmap  σ={args.sigma_m:.0f}m ({sigma_px:.2f}px)")
    else:
        print(f"Target      : normalised ITM radio map")
    print(f"Output      : {args.output}\n")

    # Check sim-engine connectivity
    try:
        with urllib.request.urlopen(f"{args.simengine_url}/health", timeout=5) as resp:
            health = json.loads(resp.read())
        print(f"sim-engine  : {health['status']} ({health['propagation_model']})")
        if health["propagation_model"] != args.propagation_model:
            print(f"  WARN: sim-engine runs '{health['propagation_model']}' "
                  f"but you requested '{args.propagation_model}'. "
                  f"Restart sim-engine with PROPAGATION_MODEL={args.propagation_model}.")
    except Exception as exc:
        print(f"ERROR: Cannot reach sim-engine at {args.simengine_url}: {exc}")
        print("Start docker compose and try again.")
        sys.exit(1)

    metadata = {
        "n_scenes":          args.n_scenes,
        "n_train":           n_train,
        "n_val":             n_val,
        "freq_range_hz":     args.freq_range,
        "zones":             zone_names,
        "propagation_model": args.propagation_model,
        "resolution_px":     args.resolution,
        "scene_size_km":     args.scene_size_km,
        "n_sensors_min":     args.n_sensors_min,
        "n_sensors_max":     args.n_sensors_max,
        "sigma_m":           args.sigma_m,
        "shadow_std_db":     args.shadow_std,
        "hw_offset_std_db":  args.hw_offset_std,
        "rssi_floor_dbm":    RSSI_FLOOR_DBM,
        "rssi_range_db":     RSSI_RANGE_DB,
        "seed":              args.seed,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))

    success, failed = 0, 0
    t_start = time.monotonic()

    for i in range(args.n_scenes):
        split   = "val" if i >= n_train else "train"
        out_dir = args.output / "val" if split == "val" else args.output
        scene_path = out_dir / f"scene_{i:06d}.npz"
        if args.resume and scene_path.exists():
            success += 1
            continue

        # Sample scene parameters
        zone      = random.choice(zones)
        bbox      = random_bbox_within(zone, size_km=args.scene_size_km)
        freq_hz   = random.uniform(args.freq_range[0], args.freq_range[1])
        emitter   = random_emitter_in_bbox(bbox)
        n_sensors = random.randint(args.n_sensors_min, args.n_sensors_max)
        power_dbm = (random.uniform(args.power_min, args.power_max)
                     if use_random_power else args.power_dbm)

        elapsed = time.monotonic() - t_start
        rate    = (i + 1) / max(elapsed, 1.0)
        eta_s   = (args.n_scenes - i - 1) / max(rate, 1e-3)
        print(
            f"  [{i+1:5d}/{args.n_scenes}] {split}  "
            f"freq={freq_hz/1e6:6.1f}MHz  pwr={power_dbm:4.0f}dBm  "
            f"bbox=({bbox.lat_min:.3f},{bbox.lon_min:.3f})+{args.scene_size_km:.0f}km  "
            f"rate={rate:.2f}/s  eta={eta_s/60:.1f}min"
        )

        # --- Terrain conditioning [3, H, W] ---
        conditioning = geo.get_conditioning_tensor(bbox, resolution_px=args.resolution)

        # Emitter pixel position
        emitter_row = (bbox.lat_max - emitter["lat"]) / bbox.height_deg * args.resolution
        emitter_col = (emitter["lon"] - bbox.lon_min) / bbox.width_deg * args.resolution

        if sigma_px > 0:
            # Gaussian heatmap target — no radiomap query needed (fast path)
            target_gt  = _gaussian_heatmap(emitter_row, emitter_col, args.resolution,
                                           args.resolution, sigma_px)
            radiomap_raw = np.zeros((args.resolution, args.resolution), dtype=np.float32)
        else:
            # Full ITM radio map target
            radiomap_gt = query_radiomap(
                args.simengine_url, emitter, bbox, freq_hz, power_dbm, args.resolution
            )
            if radiomap_gt is None:
                print("    SKIP: radiomap query failed")
                failed += 1
                continue
            target_gt   = np.clip(
                (radiomap_gt - RSSI_FLOOR_DBM) / RSSI_RANGE_DB, 0.0, 1.0
            ).astype(np.float32)
            radiomap_raw = radiomap_gt.astype(np.float32)

        # --- Sparse RSSI observations (model input, parallel) ---
        rssi_map, rssi_mask = build_sparse_obs(
            args.simengine_url, emitter, bbox, freq_hz, power_dbm,
            args.resolution, n_sensors,
            alt_range_m=(args.sensor_alt_min, args.sensor_alt_max),
            shadow_std_db=args.shadow_std,
            hw_offset_std_db=args.hw_offset_std,
        )
        n_valid = int(rssi_mask.sum())
        print(f"    obs: {n_valid}/{n_sensors} valid")

        np.savez_compressed(
            scene_path,
            # Model inputs
            conditioning = conditioning.astype(np.float32),    # [3, H, W]
            rssi_map     = rssi_map.astype(np.float32),        # [H, W]
            rssi_mask    = rssi_mask.astype(np.float32),       # [H, W]
            # Labels
            radiomap_gt  = target_gt.astype(np.float32),       # [H, W] (heatmap or norm)
            radiomap_raw = radiomap_raw,                        # [H, W] dBm or zeros
            # Metadata
            bbox         = np.array([bbox.lat_min, bbox.lon_min,
                                      bbox.lat_max, bbox.lon_max]),
            emitter_pos  = np.array([emitter["lat"], emitter["lon"], emitter["alt_m"]]),
            emitter_px   = np.array([emitter_row, emitter_col]),
            freq_hz      = np.float32(freq_hz),
            power_dbm    = np.float32(power_dbm),
        )
        success += 1

    elapsed_total = time.monotonic() - t_start
    print(f"\nDataset complete: {success} saved, {failed} failed "
          f"({elapsed_total:.0f}s total)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
