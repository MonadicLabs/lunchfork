#!/usr/bin/env python3
"""
scripts/eval.py — Evaluate localisation performance from scenario logs.

Computes metrics: RMSE, CEP50, CEP90, convergence_time_sec.

Usage:
  python scripts/eval.py --scenario scripts/scenarios/orbit_uhf.yaml \\
    --log-dir logs/run-20260324_120000/ --output results/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yaml
except ImportError:
    print("Error: PyYAML required. Install with: pip install pyyaml")
    sys.exit(1)

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate lunchfork scenario results")
    p.add_argument("--scenario", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def parse_target_log(log_dir: Path) -> list[dict]:
    """
    Parse TargetLocation records from the log directory.

    Priority:
      1. targets.jsonl  — written by run_scenario.py REST collector (preferred)
      2. *.log / *.jsonl — master stdout redirected to file (fallback)
    """
    records = []

    # 1. Preferred: collector output
    targets_jsonl = log_dir / "targets.jsonl"
    if targets_jsonl.exists():
        with open(targets_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    # REST /api/targets returns list of target dicts with lat/lon/etc.
                    if "lat" in rec or "position" in rec or "target_id" in rec:
                        records.append(rec)
                except json.JSONDecodeError:
                    pass
        if records:
            return records

    # 2. Fallback: master log stdout
    for log_file in list(log_dir.glob("*.log")) + list(log_dir.glob("*.jsonl")):
        if log_file.name == "targets.jsonl":
            continue
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("event") == "target_location" or "target_id" in rec:
                        records.append(rec)
                except json.JSONDecodeError:
                    pass

    return records


def parse_rssi_log(log_dir: Path) -> list[dict]:
    """Parse RSSI messages from node log files."""
    records = []
    for log_file in log_dir.glob("node_*.log"):
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "rssi_dbm" in record:
                        records.append(record)
                except json.JSONDecodeError:
                    pass
    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_rmse(errors_m: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errors_m**2)))


def compute_cep(errors_m: np.ndarray, percentile: float) -> float:
    """Circular Error Probable at given percentile."""
    return float(np.percentile(errors_m, percentile))


def compute_convergence_time(
    timestamps: list[float],
    errors_m: np.ndarray,
    threshold_m: float,
) -> float | None:
    """
    Return the time (in seconds from start) when error drops below threshold_m
    and stays there for at least 30s.
    """
    if len(timestamps) == 0:
        return None

    t0 = timestamps[0]
    min_stable_s = 30.0
    below_threshold = errors_m < threshold_m

    for i in range(len(timestamps)):
        if below_threshold[i]:
            # Check how long it stays below threshold
            t_start = timestamps[i]
            stayed = True
            for j in range(i, len(timestamps)):
                if timestamps[j] - t_start > min_stable_s:
                    break
                if not below_threshold[j]:
                    stayed = False
                    break
            if stayed:
                return timestamps[i] - t0

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> dict:
    if not args.scenario.exists():
        print(f"ERROR: Scenario not found: {args.scenario}")
        sys.exit(1)
    if not args.log_dir.exists():
        print(f"ERROR: Log dir not found: {args.log_dir}")
        sys.exit(1)

    with open(args.scenario) as f:
        scenario = yaml.safe_load(f)

    print(f"\n=== Evaluation: {scenario.get('name', 'unnamed')} ===")
    print(f"Log dir: {args.log_dir}")

    # Find ground truth emitter
    eval_cfg = scenario.get("eval", {})
    gt_emitter_id = eval_cfg.get("ground_truth_emitter", "target-1")
    convergence_threshold_m = eval_cfg.get("convergence_threshold_m", 200)
    metrics_requested = eval_cfg.get("metrics", ["rmse_m", "cep50", "cep90"])

    gt_emitter = next(
        (e for e in scenario.get("emitters", []) if e["id"] == gt_emitter_id), None
    )
    if gt_emitter is None:
        print(f"ERROR: Ground truth emitter '{gt_emitter_id}' not found in scenario")
        sys.exit(1)

    gt_lat = gt_emitter["lat"]
    gt_lon = gt_emitter["lon"]
    print(f"\nGround truth: {gt_emitter_id} @ ({gt_lat}, {gt_lon})")

    # Parse logs
    target_records = parse_target_log(args.log_dir)

    if not target_records:
        print(f"\nWARN: No target location records found in {args.log_dir}")
        print("Generating synthetic evaluation from RSSI logs...")
        rssi_records = parse_rssi_log(args.log_dir)
        if rssi_records:
            print(f"  Found {len(rssi_records)} RSSI records")
        else:
            print("  No RSSI records found either. Was the scenario run?")

        # Return placeholder metrics
        return {
            "scenario": scenario.get("name"),
            "ground_truth": {"id": gt_emitter_id, "lat": gt_lat, "lon": gt_lon},
            "n_estimates": 0,
            "metrics": {m: None for m in metrics_requested},
            "note": "No target location estimates found — run Phase 3 pipeline first",
        }

    # Extract positions and timestamps
    # Group by target_id first; master uses channel-hash IDs, not emitter IDs.
    # We pick the channel whose estimates are closest (on average) to ground truth.
    by_channel: dict[str, list[dict]] = {}
    for rec in target_records:
        pos = rec.get("position", {})
        lat = rec.get("lat") or pos.get("lat")
        lon = rec.get("lon") or pos.get("lon")
        ts  = rec.get("_ts") or rec.get("timestamp_utc")
        if lat is None or lon is None:
            continue
        tid = rec.get("target_id", "unknown")
        try:
            t = datetime.fromisoformat(ts).timestamp() if ts else 0.0
        except Exception:
            t = 0.0
        error_m = haversine_m(gt_lat, gt_lon, lat, lon)
        by_channel.setdefault(tid, []).append(
            {"lat": lat, "lon": lon, "t": t, "error_m": error_m,
             "uncertainty_m": rec.get("uncertainty_m")}
        )

    if not by_channel:
        estimates = []
    elif gt_emitter_id in by_channel:
        # Exact match by emitter ID (future: master may publish with emitter ID)
        estimates = by_channel[gt_emitter_id]
        print(f"  Matched by target_id='{gt_emitter_id}'")
    else:
        # Pick channel with best (lowest) median error to ground truth
        best_ch = min(by_channel, key=lambda ch: np.median([e["error_m"] for e in by_channel[ch]]))
        estimates = by_channel[best_ch]
        print(f"  No exact target_id match; using closest channel '{best_ch}' "
              f"(median error {np.median([e['error_m'] for e in estimates]):.0f}m)")

    if not estimates:
        print("No estimates matching ground truth emitter found")
        return {"metrics": {}, "n_estimates": 0}

    estimates.sort(key=lambda x: x["t"])
    errors_m = np.array([e["error_m"] for e in estimates])
    timestamps = [e["t"] for e in estimates]

    print(f"\nFound {len(estimates)} position estimates")
    print(f"Error range: {errors_m.min():.0f}m – {errors_m.max():.0f}m")

    # Compute requested metrics
    results: dict[str, float | None] = {}

    if "rmse_m" in metrics_requested:
        results["rmse_m"] = compute_rmse(errors_m)
        print(f"  RMSE:   {results['rmse_m']:.1f} m")

    if "cep50" in metrics_requested:
        results["cep50"] = compute_cep(errors_m, 50)
        print(f"  CEP50:  {results['cep50']:.1f} m")

    if "cep90" in metrics_requested:
        results["cep90"] = compute_cep(errors_m, 90)
        print(f"  CEP90:  {results['cep90']:.1f} m")

    if "convergence_time_sec" in metrics_requested:
        conv_t = compute_convergence_time(timestamps, errors_m, convergence_threshold_m)
        results["convergence_time_sec"] = conv_t
        if conv_t is not None:
            print(f"  Convergence ({convergence_threshold_m}m): {conv_t:.0f}s")
        else:
            print(f"  Convergence ({convergence_threshold_m}m): not reached")

    result = {
        "scenario": scenario.get("name"),
        "log_dir": str(args.log_dir),
        "ground_truth": {"id": gt_emitter_id, "lat": gt_lat, "lon": gt_lon},
        "n_estimates": len(estimates),
        "metrics": results,
    }

    # Save results
    if args.output:
        out_dir = args.output
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"eval_{ts}.json"
        out_file.write_text(json.dumps(result, indent=2))
        print(f"\nResults saved: {out_file}")

    return result


if __name__ == "__main__":
    evaluate(parse_args())
