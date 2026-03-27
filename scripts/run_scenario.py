#!/usr/bin/env python3
"""
scripts/run_scenario.py — Load and run a SITL scenario.

1. Parse YAML scenario file
2. Register emitters with sim-engine via REST
3. Start node-sitl processes (or configure running Docker nodes)
4. Run for scenario duration
5. Collect logs to logs/ directory

Usage:
  python scripts/run_scenario.py scripts/scenarios/orbit_uhf.yaml
  python scripts/run_scenario.py scripts/scenarios/orbit_uhf.yaml --duration 300
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yaml
except ImportError:
    print("Error: PyYAML required. Install with: pip install pyyaml")
    sys.exit(1)


SIMENGINE_URL = os.environ.get("SIMENGINE_URL", "http://localhost:9000")
LOG_DIR = ROOT / "logs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a lunchfork SITL scenario")
    p.add_argument("scenario", type=Path, help="Path to scenario YAML file")
    p.add_argument("--duration", type=int, default=None, help="Override duration in seconds")
    p.add_argument("--simengine", default=SIMENGINE_URL, help="sim-engine base URL")
    p.add_argument("--log-dir", type=Path, default=None, help="Log output directory")
    p.add_argument(
        "--no-nodes",
        action="store_true",
        help="Only register emitters, do not start node processes",
    )
    return p.parse_args()


def http_post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def http_delete(url: str) -> None:
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        pass


def http_get(url: str) -> dict | list:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def wait_for_simengine(url: str, timeout: int = 60) -> bool:
    """Wait for sim-engine health endpoint to respond."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = http_get(f"{url}/health")
            if isinstance(result, dict) and result.get("status") == "ok":
                print(f"sim-engine ready: {url}")
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"ERROR: sim-engine not available at {url} after {timeout}s")
    return False


def register_emitters(scenario: dict, simengine_url: str) -> list[str]:
    """Register all scenario emitters with sim-engine. Returns list of emitter IDs."""
    emitter_ids = []
    for emitter in scenario.get("emitters", []):
        payload = {
            "id": emitter["id"],
            "lat": emitter["lat"],
            "lon": emitter["lon"],
            "alt_m": emitter.get("alt_m", 5),
            "freq_hz": emitter["freq_hz"],
            "power_dbm": emitter.get("power_dbm", 10),
        }
        try:
            result = http_post(f"{simengine_url}/emitter", payload)
            emitter_ids.append(result["id"])
            print(f"  Registered emitter {result['id']} at ({emitter['lat']}, {emitter['lon']})")
        except Exception as exc:
            print(f"  ERROR registering emitter {emitter['id']}: {exc}")
    return emitter_ids


def deregister_emitters(emitter_ids: list[str], simengine_url: str) -> None:
    """Remove all emitters from sim-engine."""
    for eid in emitter_ids:
        try:
            http_delete(f"{simengine_url}/emitter/{eid}")
            print(f"  Removed emitter {eid}")
        except Exception as exc:
            print(f"  WARN: could not remove emitter {eid}: {exc}")


def build_node_env(node: dict, scenario: dict, simengine_url: str) -> dict[str, str]:
    """Build environment variables for a node-sitl process."""
    env = dict(os.environ)
    env["NODE_ID"] = node["id"]
    env["NODE_TYPE"] = node["type"]
    env["SIMENGINE_URL"] = simengine_url
    env["UPDATE_RATE_HZ"] = "1"
    env["LOG_LEVEL"] = "INFO"
    env["COMM_TRANSPORT"] = os.environ.get("COMM_TRANSPORT", "mqtt")
    env["COMM_BROKER_URL"] = os.environ.get("COMM_BROKER_URL", "mqtt://localhost:1883")

    traj = node.get("trajectory", {})
    if isinstance(traj, str):
        env["TRAJECTORY_TYPE"] = traj
    elif isinstance(traj, dict):
        traj_type = traj.get("type", "static")
        env["TRAJECTORY_TYPE"] = traj_type
        if traj_type == "orbit":
            env["TRAJECTORY_ORBIT_LAT"] = str(traj.get("center_lat", node.get("lat", 43.5)))
            env["TRAJECTORY_ORBIT_LON"] = str(traj.get("center_lon", node.get("lon", 5.45)))
            env["TRAJECTORY_ORBIT_RADIUS_M"] = str(traj.get("radius_m", 800))
            env["TRAJECTORY_ORBIT_ALT_M"] = str(traj.get("alt_m", 120))
            env["TRAJECTORY_ORBIT_PERIOD_SEC"] = str(traj.get("period_sec", 120))
            env["TRAJECTORY_ORBIT_HELIX"] = str(traj.get("helix", False)).lower()
        elif traj_type == "static":
            env["TRAJECTORY_ORBIT_LAT"] = str(node.get("lat", 43.5))
            env["TRAJECTORY_ORBIT_LON"] = str(node.get("lon", 5.45))
            env["TRAJECTORY_ORBIT_ALT_M"] = str(node.get("alt_m", 180))

    if "lat" in node and "trajectory" not in node:
        env["TRAJECTORY_TYPE"] = "static"
        env["TRAJECTORY_ORBIT_LAT"] = str(node["lat"])
        env["TRAJECTORY_ORBIT_LON"] = str(node["lon"])
        env["TRAJECTORY_ORBIT_ALT_M"] = str(node.get("alt_m", 0))

    return env


def start_node_processes(
    scenario: dict, simengine_url: str, log_dir: Path
) -> list[subprocess.Popen]:
    """Start node-sitl subprocesses."""
    node_script = ROOT / "containers" / "node-sitl" / "main.py"
    procs = []

    for node in scenario.get("nodes", []):
        env = build_node_env(node, scenario, simengine_url)
        log_file = log_dir / f"node_{node['id']}.log"
        log_file_handle = open(log_file, "w")

        print(f"  Starting node {node['id']} ({node['type']}) → {log_file}")
        try:
            proc = subprocess.Popen(
                [sys.executable, str(node_script)],
                env=env,
                stdout=log_file_handle,
                stderr=subprocess.STDOUT,
            )
            procs.append(proc)
        except Exception as exc:
            print(f"  ERROR starting node {node['id']}: {exc}")

    return procs


MASTER_URL = os.environ.get("MASTER_URL", "http://localhost:8080")
COLLECT_INTERVAL_S = 5.0  # poll master /api/targets every N seconds


def collect_target_estimates(
    master_url: str,
    log_dir: Path,
    duration: float,
    poll_interval: float = COLLECT_INTERVAL_S,
) -> None:
    """
    Poll master /api/targets at regular intervals and write TargetLocation
    records to targets.jsonl in the log directory.

    Also polls /api/nodes for node status and /api/channels for PF stats.
    Runs synchronously — call from a thread.
    """
    targets_log = open(log_dir / "targets.jsonl", "w")
    nodes_log   = open(log_dir / "nodes.jsonl",   "w")
    channels_log = open(log_dir / "channels.jsonl", "w")

    t_start = time.time()
    n_records = 0

    try:
        while time.time() - t_start < duration:
            ts = datetime.now().isoformat()
            for endpoint, fh in [
                ("/api/targets",  targets_log),
                ("/api/nodes",    nodes_log),
                ("/api/channels", channels_log),
            ]:
                try:
                    with urllib.request.urlopen(f"{master_url}{endpoint}", timeout=5) as r:
                        records = json.loads(r.read())
                    if not isinstance(records, list):
                        records = [records]
                    for rec in records:
                        rec["_ts"] = ts
                        fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    if endpoint == "/api/targets":
                        n_records += len(records)
                except Exception:
                    pass
            time.sleep(poll_interval)
    finally:
        targets_log.close()
        nodes_log.close()
        channels_log.close()

    return n_records


def run_scenario(args: argparse.Namespace) -> None:
    import threading

    scenario_path = args.scenario.resolve()
    if not scenario_path.exists():
        print(f"ERROR: Scenario file not found: {scenario_path}")
        sys.exit(1)

    with open(scenario_path) as f:
        scenario = yaml.safe_load(f)

    duration = args.duration or scenario.get("duration_sec", 600)
    simengine_url = args.simengine
    master_url = os.environ.get("MASTER_URL", MASTER_URL)

    # Prepare log directory
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or LOG_DIR / f"run-{run_ts}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save scenario copy to log dir
    (log_dir / "scenario.yaml").write_text(scenario_path.read_text())

    print(f"\n=== lunchfork SITL scenario: {scenario.get('name', 'unnamed')} ===")
    print(f"Duration: {duration}s | sim-engine: {simengine_url} | master: {master_url}")
    print(f"Log dir: {log_dir}\n")

    # Wait for sim-engine
    if not wait_for_simengine(simengine_url):
        print("Aborting: sim-engine not available. Start docker compose first.")
        sys.exit(1)

    # Register emitters
    print("Registering emitters...")
    emitter_ids = register_emitters(scenario, simengine_url)
    print(f"  {len(emitter_ids)} emitter(s) registered\n")

    # Start background collector thread (polls master REST API)
    collector_results = {"n_records": 0}
    def _collector():
        collector_results["n_records"] = collect_target_estimates(
            master_url, log_dir, duration + 10, COLLECT_INTERVAL_S
        )
    collector_thread = threading.Thread(target=_collector, daemon=True)
    collector_thread.start()

    # Start node processes
    procs: list[subprocess.Popen] = []
    if not args.no_nodes:
        print("Starting node-sitl processes...")
        procs = start_node_processes(scenario, simengine_url, log_dir)
        print(f"  {len(procs)} node(s) started\n")
        time.sleep(2)  # Let nodes initialise

    # Run for duration
    print(f"Running scenario for {duration}s... (Ctrl-C to stop early)")
    print(f"Collecting target estimates every {COLLECT_INTERVAL_S:.0f}s from {master_url}/api/targets\n")
    start = time.time()
    try:
        while time.time() - start < duration:
            elapsed = time.time() - start
            remaining = duration - elapsed
            print(
                f"\r  Elapsed: {elapsed:.0f}s / {duration}s "
                f"({remaining:.0f}s remaining)",
                end="", flush=True,
            )
            time.sleep(5)
        print()
    except KeyboardInterrupt:
        elapsed = time.time() - start
        print(f"\nInterrupted after {elapsed:.0f}s")

    # Cleanup
    print("\nCleaning up...")
    for proc in procs:
        proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("Deregistering emitters...")
    deregister_emitters(emitter_ids, simengine_url)

    # Count records from file (collector thread may still be running)
    targets_file = log_dir / "targets.jsonl"
    n_captured = sum(1 for _ in open(targets_file) if _.strip()) if targets_file.exists() else "?"

    print(f"\nScenario complete.")
    print(f"  Target estimates captured: {n_captured}")
    print(f"  Logs: {log_dir}")
    print(f"\nEvaluate with:")
    print(f"  python scripts/eval.py --scenario {scenario_path} --log-dir {log_dir}")


if __name__ == "__main__":
    run_scenario(parse_args())
