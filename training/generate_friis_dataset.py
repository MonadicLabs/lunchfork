#!/usr/bin/env python3
"""
training/generate_friis_dataset.py — Fast offline Friis synthetic dataset.

Generates UNet training scenes entirely in NumPy (no HTTP, no sim-engine).
Uses Friis free-space path loss as the propagation model.

~500 scenes/second on CPU — generates 5000 scenes in ~10 seconds.

Output format matches generate_dataset.py:
  conditioning  [3, H, W]  — zeros (no terrain)
  rssi_map      [H, W]     — sparse normalised RSSI at sensor positions
  rssi_mask     [H, W]     — 1 where a sensor exists
  radiomap_gt   [H, W]     — dense normalised Friis radio map (training label)

Usage:
  python training/generate_friis_dataset.py \\
    --n-scenes 5000 \\
    --output   data/datasets/friis_fast_v1/ \\
    --resolution 64

"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RSSI_FLOOR_DBM = -120.0
RSSI_RANGE_DB  =   80.0   # −120 → −40 dBm → [0, 1]
SPEED_OF_LIGHT = 2.998e8


def fspl_db(d_m: np.ndarray, freq_hz: float) -> np.ndarray:
    d = np.maximum(d_m, 1.0)
    return 20*np.log10(d) + 20*np.log10(freq_hz) - 20*math.log10(SPEED_OF_LIGHT/(4*math.pi))


def norm(rssi: np.ndarray) -> np.ndarray:
    return np.clip((rssi - RSSI_FLOOR_DBM) / RSSI_RANGE_DB, 0.0, 1.0).astype(np.float32)


def _gaussian_heatmap(em_row: float, em_col: float, H: int, W: int, sigma_px: float) -> np.ndarray:
    rows = np.arange(H, dtype=np.float32)[:, None]
    cols = np.arange(W, dtype=np.float32)[None, :]
    g = np.exp(-((rows - em_row)**2 + (cols - em_col)**2) / (2 * sigma_px**2))
    return (g / (g.max() + 1e-8)).astype(np.float32)


def generate_scene(
    rng: np.random.Generator,
    resolution: int,
    freq_range: tuple[float, float],
    scene_size_m: float,
    n_sensors_range: tuple[int, int],
    sensor_alt_range: tuple[float, float],
    emitter_alt_range: tuple[float, float],
    power_dbm: float | tuple[float, float],
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    sigma_px: float = 0.0,
) -> dict:
    freq_hz = rng.uniform(*freq_range)
    if isinstance(power_dbm, tuple):
        power_dbm = float(rng.uniform(*power_dbm))
    H = W = resolution

    # Grid in metres (relative to scene centre)
    xs = np.linspace(-scene_size_m/2, scene_size_m/2, W, dtype=np.float32)
    ys = np.linspace( scene_size_m/2, -scene_size_m/2, H, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)   # [H, W] each

    # Emitter position (random within inner 60% to avoid edge effects)
    margin = 0.2 * scene_size_m
    em_x = rng.uniform(-scene_size_m/2 + margin, scene_size_m/2 - margin)
    em_y = rng.uniform(-scene_size_m/2 + margin, scene_size_m/2 - margin)
    em_z = rng.uniform(*emitter_alt_range)

    # Ground-truth radio map (Friis, all grid points)
    dh = np.sqrt((xg - em_x)**2 + (yg - em_y)**2)
    d3d = np.sqrt(dh**2 + em_z**2)
    rssi_gt = power_dbm + tx_gain_dbi + rx_gain_dbi - fspl_db(d3d, freq_hz)
    radiomap_gt = norm(rssi_gt)   # [H, W]

    # Sparse sensor observations (n_sensors random positions)
    n_sensors = int(rng.integers(*n_sensors_range, endpoint=True))
    rssi_map  = np.zeros((H, W), dtype=np.float32)
    rssi_mask = np.zeros((H, W), dtype=np.float32)
    n_valid   = 0

    for _ in range(n_sensors):
        # Allow sensors outside grid centre (they may be far from emitter)
        sx = rng.uniform(-scene_size_m/2, scene_size_m/2)
        sy = rng.uniform(-scene_size_m/2, scene_size_m/2)
        sz = rng.uniform(*sensor_alt_range)

        dh_s = math.sqrt((sx - em_x)**2 + (sy - em_y)**2)
        d3d_s = math.sqrt(dh_s**2 + (sz - em_z)**2)
        rssi_s = power_dbm + tx_gain_dbi + rx_gain_dbi - float(fspl_db(np.array([d3d_s]), freq_hz)[0])

        # Drop below noise floor
        if rssi_s < RSSI_FLOOR_DBM + 5:
            continue

        # Add Gaussian measurement noise (3-8 dB)
        rssi_s += rng.normal(0, rng.uniform(3.0, 8.0))

        # Convert sensor position to pixel
        col = int((sx + scene_size_m/2) / scene_size_m * W)
        row = int((scene_size_m/2 - sy) / scene_size_m * H)
        col = np.clip(col, 0, W-1)
        row = np.clip(row, 0, H-1)

        rssi_map[row, col]  = float(norm(np.array([rssi_s]))[0])
        rssi_mask[row, col] = 1.0
        n_valid += 1

    # Emitter pixel coordinates
    em_col = (em_x + scene_size_m/2) / scene_size_m * W
    em_row = (scene_size_m/2 - em_y) / scene_size_m * H

    # Use Gaussian heatmap as target when sigma_px > 0 (BCE-trainable)
    if sigma_px > 0:
        radiomap_gt = _gaussian_heatmap(em_row, em_col, H, W, sigma_px)

    return {
        "conditioning": np.zeros((3, H, W), dtype=np.float32),
        "rssi_map":     rssi_map,
        "rssi_mask":    rssi_mask,
        "radiomap_gt":  radiomap_gt,
        "emitter_px":   np.array([em_row, em_col], dtype=np.float32),
        "n_valid_obs":  n_valid,
        "freq_hz":      np.float32(freq_hz),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-scenes",      type=int,   default=5000)
    p.add_argument("--output",        type=Path,  required=True)
    p.add_argument("--resolution",    type=int,   default=64)
    p.add_argument("--scene-size-m",  type=float, default=6000.0,
                   help="Scene extent in metres (default 6000 = ±3km, matching POWDER eval)")
    p.add_argument("--freq-min",      type=float, default=400e6)
    p.add_argument("--freq-max",      type=float, default=500e6)
    p.add_argument("--n-sensors-min", type=int,   default=3)
    p.add_argument("--n-sensors-max", type=int,   default=25)
    p.add_argument("--sensor-alt-min",  type=float, default=3.0)
    p.add_argument("--sensor-alt-max",  type=float, default=200.0)
    p.add_argument("--emitter-alt-min", type=float, default=1.5)
    p.add_argument("--emitter-alt-max", type=float, default=20.0)
    p.add_argument("--power-dbm",     type=float, default=None,
                   help="Fixed TX power in dBm. Mutually exclusive with --power-min/--power-max.")
    p.add_argument("--power-min",     type=float, default=5.0,
                   help="Min TX power in dBm for uniform random range (default 5)")
    p.add_argument("--power-max",     type=float, default=30.0,
                   help="Max TX power in dBm for uniform random range (default 30)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--val-split",     type=float, default=0.1)
    p.add_argument("--sigma-m",       type=float, default=0.0,
                   help="If >0, use Gaussian heatmap target with σ=sigma_m (metres). "
                        "0 = use Friis radio map (default).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "val").mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    sigma_px = args.sigma_m / (args.scene_size_m / args.resolution) if args.sigma_m > 0 else 0.0

    # Power: fixed if --power-dbm set, else random in [power_min, power_max]
    if args.power_dbm is not None:
        power_arg: float | tuple[float, float] = args.power_dbm
        power_str = f"{args.power_dbm:.0f}dBm"
    else:
        power_arg = (args.power_min, args.power_max)
        power_str = f"{args.power_min:.0f}–{args.power_max:.0f}dBm (random)"

    print(f"Generating {args.n_scenes} Friis scenes → {args.output}")
    print(f"  resolution={args.resolution}  scene={args.scene_size_m/1000:.1f}km  "
          f"freq={args.freq_min/1e6:.0f}-{args.freq_max/1e6:.0f}MHz  "
          f"sensors={args.n_sensors_min}-{args.n_sensors_max}  power={power_str}")
    if sigma_px > 0:
        print(f"  target=gaussian  sigma={args.sigma_m:.0f}m ({sigma_px:.2f}px)")

    n_val   = max(1, int(args.n_scenes * args.val_split))
    n_train = args.n_scenes - n_val

    for i in range(args.n_scenes):
        split  = "val" if i >= n_train else "train"
        out_dir = args.output / "val" if split == "val" else args.output
        path   = out_dir / f"scene_{i:06d}.npz"

        scene = generate_scene(
            rng,
            resolution     = args.resolution,
            freq_range     = (args.freq_min, args.freq_max),
            scene_size_m   = args.scene_size_m,
            n_sensors_range= (args.n_sensors_min, args.n_sensors_max),
            sensor_alt_range   = (args.sensor_alt_min, args.sensor_alt_max),
            emitter_alt_range  = (args.emitter_alt_min, args.emitter_alt_max),
            power_dbm      = power_arg,
            tx_gain_dbi    = 2.15,
            rx_gain_dbi    = 0.0,
            sigma_px       = sigma_px,
        )

        np.savez_compressed(path, **{k: v for k, v in scene.items()
                                     if k != "n_valid_obs"})

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{args.n_scenes}  (split={split})")

    meta = {
        "source":           "Friis synthetic (offline, no sim-engine)",
        "n_scenes":         args.n_scenes,
        "n_train":          n_train,
        "n_val":            n_val,
        "resolution_px":    args.resolution,
        "scene_size_m":     args.scene_size_m,
        "freq_range_hz":    [args.freq_min, args.freq_max],
        "n_sensors_range":  [args.n_sensors_min, args.n_sensors_max],
        "seed":             args.seed,
    }
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone: {args.n_scenes} scenes in {args.output}")


if __name__ == "__main__":
    import time
    t0 = time.monotonic()
    main()
    print(f"Elapsed: {time.monotonic()-t0:.1f}s")
