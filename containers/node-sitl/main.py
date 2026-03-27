"""
node-sitl — Simulated SDR node for lunchfork SITL testing.

Reads trajectory from environment variables, queries sim-engine for RSSI,
and publishes RssiMessage + NodeStatus via CommTransport.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import aiohttp
import structlog

# Ensure shared is importable
sys.path.insert(0, "/app")

from shared.comm import get_transport
from shared.messages import (
    FreqChannel,
    NodePosition,
    NodeStatus,
    RssiMessage,
)

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NODE_ID = os.environ.get("NODE_ID", "sim-node-001")
NODE_TYPE = os.environ.get("NODE_TYPE", "ground")  # ground|uav|vehicle
SIMENGINE_URL = os.environ.get("SIMENGINE_URL", "http://sim-engine:9000")
UPDATE_RATE_HZ = float(os.environ.get("UPDATE_RATE_HZ", "1"))

TRAJECTORY_TYPE = os.environ.get("TRAJECTORY_TYPE", "static")  # static|orbit|gps_replay

# Static / orbit parameters
STATIC_LAT = float(os.environ.get("TRAJECTORY_ORBIT_LAT", "43.535"))
STATIC_LON = float(os.environ.get("TRAJECTORY_ORBIT_LON", "5.455"))
ORBIT_RADIUS_M = float(os.environ.get("TRAJECTORY_ORBIT_RADIUS_M", "800"))
ORBIT_ALT_M = float(os.environ.get("TRAJECTORY_ORBIT_ALT_M", "120"))
ORBIT_PERIOD_SEC = float(os.environ.get("TRAJECTORY_ORBIT_PERIOD_SEC", "120"))
ORBIT_HELIX = os.environ.get("TRAJECTORY_ORBIT_HELIX", "false").lower() == "true"
ORBIT_HELIX_DELTA_M = float(os.environ.get("TRAJECTORY_ORBIT_HELIX_DELTA_M", "20"))

# GPS replay
GPS_REPLAY_FILE = os.environ.get("TRAJECTORY_GPS_FILE", "")

# Frequency channels to monitor
# Format: "freq_hz:bandwidth_hz:label,..." — comma separated
FREQ_CHANNELS_ENV = os.environ.get(
    "FREQ_CHANNELS",
    "433920000:25000:UHF-433",
)


def parse_freq_channels() -> list[FreqChannel]:
    channels = []
    for entry in FREQ_CHANNELS_ENV.split(","):
        parts = entry.strip().split(":")
        if len(parts) >= 2:
            channels.append(
                FreqChannel(
                    center_hz=int(parts[0]),
                    bandwidth_hz=int(parts[1]),
                    label=parts[2] if len(parts) > 2 else None,
                )
            )
    return channels or [FreqChannel(center_hz=433920000, bandwidth_hz=25000, label="UHF-433")]


FREQ_CHANNELS = parse_freq_channels()

# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


def deg_to_m_factor(lat: float) -> tuple[float, float]:
    """Return (metres_per_lat_deg, metres_per_lon_deg) at given latitude."""
    lat_m = 111320.0
    lon_m = 111320.0 * math.cos(math.radians(lat))
    return lat_m, lon_m


class StaticTrajectory:
    def __init__(self, lat: float, lon: float, alt_m: float) -> None:
        self._lat = lat
        self._lon = lon
        self._alt_m = alt_m

    def position_at(self, t: float) -> NodePosition:
        return NodePosition(lat=self._lat, lon=self._lon, alt_m=self._alt_m)


class OrbitTrajectory:
    """
    Circular orbit (optionally helical).

    Centre: (STATIC_LAT, STATIC_LON)
    Radius: ORBIT_RADIUS_M
    Altitude: ORBIT_ALT_M (base), increases linearly if helix=True
    """

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: float,
        alt_m: float,
        period_sec: float,
        helix: bool = False,
        helix_delta_m: float = 20.0,
    ) -> None:
        self._clat = center_lat
        self._clon = center_lon
        self._radius_m = radius_m
        self._alt_m = alt_m
        self._period = period_sec
        self._helix = helix
        self._helix_delta_m = helix_delta_m

    def position_at(self, t: float) -> NodePosition:
        angle = 2 * math.pi * (t % self._period) / self._period
        lat_m_per_deg, lon_m_per_deg = deg_to_m_factor(self._clat)

        dlat = self._radius_m * math.sin(angle) / lat_m_per_deg
        dlon = self._radius_m * math.cos(angle) / lon_m_per_deg

        if self._helix:
            # Altitude oscillates between alt_m and alt_m + helix_delta_m
            alt = self._alt_m + self._helix_delta_m * (1.0 - math.cos(angle)) / 2.0
        else:
            alt = self._alt_m

        return NodePosition(
            lat=self._clat + dlat,
            lon=self._clon + dlon,
            alt_m=alt,
        )


class GpsReplayTrajectory:
    """
    Replay a GPS track from a CSV or GPX file.

    CSV format: timestamp_utc,lat,lon,alt_m (header row)
    GPX format: standard GPX 1.1 trackpoints
    """

    def __init__(self, filepath: str) -> None:
        self._track: list[tuple[float, float, float, float]] = []  # (t_rel, lat, lon, alt)
        self._duration = 0.0
        self._load(filepath)

    def _load(self, filepath: str) -> None:
        import csv
        from pathlib import Path

        path = Path(filepath)
        if not path.exists():
            logger.warning("gps_replay.file_not_found", filepath=filepath)
            return

        if path.suffix.lower() == ".gpx":
            self._load_gpx(path)
        else:
            self._load_csv(path)

    def _load_csv(self, path: Any) -> None:
        import csv

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return

        t0 = datetime.fromisoformat(rows[0]["timestamp_utc"]).timestamp()
        for row in rows:
            t = datetime.fromisoformat(row["timestamp_utc"]).timestamp() - t0
            self._track.append((t, float(row["lat"]), float(row["lon"]), float(row.get("alt_m", 0))))

        self._duration = self._track[-1][0] if self._track else 0.0
        logger.info("gps_replay.loaded_csv", n_points=len(self._track), duration_s=self._duration)

    def _load_gpx(self, path: Any) -> None:
        import xml.etree.ElementTree as ET

        tree = ET.parse(path)
        root = tree.getroot()
        ns = {"gpx": "http://www.topografix.com/GPX/1/1"}

        points = root.findall(".//gpx:trkpt", ns)
        if not points:
            return

        t0 = None
        for pt in points:
            lat = float(pt.attrib["lat"])
            lon = float(pt.attrib["lon"])
            ele_el = pt.find("gpx:ele", ns)
            alt = float(ele_el.text) if ele_el is not None else 0.0
            time_el = pt.find("gpx:time", ns)
            if time_el is not None:
                t = datetime.fromisoformat(time_el.text.replace("Z", "+00:00")).timestamp()
                if t0 is None:
                    t0 = t
                self._track.append((t - t0, lat, lon, alt))
            else:
                idx = len(self._track)
                self._track.append((float(idx), lat, lon, alt))

        self._duration = self._track[-1][0] if self._track else 0.0
        logger.info("gps_replay.loaded_gpx", n_points=len(self._track), duration_s=self._duration)

    def position_at(self, t: float) -> NodePosition:
        if not self._track:
            return NodePosition(lat=STATIC_LAT, lon=STATIC_LON, alt_m=ORBIT_ALT_M)

        # Loop replay
        if self._duration > 0:
            t = t % self._duration

        # Linear interpolation
        for i in range(len(self._track) - 1):
            t0, lat0, lon0, alt0 = self._track[i]
            t1, lat1, lon1, alt1 = self._track[i + 1]
            if t0 <= t <= t1:
                alpha = (t - t0) / max(t1 - t0, 1e-9)
                return NodePosition(
                    lat=lat0 + alpha * (lat1 - lat0),
                    lon=lon0 + alpha * (lon1 - lon0),
                    alt_m=alt0 + alpha * (alt1 - alt0),
                )

        t_last, lat_last, lon_last, alt_last = self._track[-1]
        return NodePosition(lat=lat_last, lon=lon_last, alt_m=alt_last)


def build_trajectory():
    if TRAJECTORY_TYPE == "orbit":
        return OrbitTrajectory(
            center_lat=STATIC_LAT,
            center_lon=STATIC_LON,
            radius_m=ORBIT_RADIUS_M,
            alt_m=ORBIT_ALT_M,
            period_sec=ORBIT_PERIOD_SEC,
            helix=ORBIT_HELIX,
            helix_delta_m=ORBIT_HELIX_DELTA_M,
        )
    if TRAJECTORY_TYPE == "gps_replay" and GPS_REPLAY_FILE:
        return GpsReplayTrajectory(GPS_REPLAY_FILE)
    # Default: static
    return StaticTrajectory(lat=STATIC_LAT, lon=STATIC_LON, alt_m=ORBIT_ALT_M)


# ---------------------------------------------------------------------------
# Main node loop
# ---------------------------------------------------------------------------


async def query_rssi(
    session: aiohttp.ClientSession,
    emitter_id: str,
    sensor_pos: NodePosition,
    freq_hz: float,
) -> float | None:
    """Query sim-engine for RSSI at sensor_pos given emitter emitter_id."""
    try:
        # First get emitter position
        async with session.get(f"{SIMENGINE_URL}/emitters") as resp:
            emitters = await resp.json()

        emitter = next((e for e in emitters if e["id"] == emitter_id), None)
        if emitter is None:
            return None

        payload = {
            "freq_hz": freq_hz,
            "emitter": {
                "lat": emitter["lat"],
                "lon": emitter["lon"],
                "alt_m": emitter["alt_m"],
            },
            "sensor": {
                "lat": sensor_pos.lat,
                "lon": sensor_pos.lon,
                "alt_m": sensor_pos.alt_m,
            },
            "power_dbm": emitter["power_dbm"],
        }
        async with session.post(f"{SIMENGINE_URL}/rssi", json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["rssi_dbm"]
    except Exception as exc:
        logger.warning("node.rssi_query_failed", error=str(exc))
    return None


async def query_rssi_direct(
    session: aiohttp.ClientSession,
    emitter_lat: float,
    emitter_lon: float,
    emitter_alt_m: float,
    power_dbm: float,
    sensor_pos: NodePosition,
    freq_hz: float,
) -> float | None:
    """Query sim-engine for RSSI with explicit emitter position."""
    try:
        payload = {
            "freq_hz": freq_hz,
            "emitter": {
                "lat": emitter_lat,
                "lon": emitter_lon,
                "alt_m": emitter_alt_m,
            },
            "sensor": {
                "lat": sensor_pos.lat,
                "lon": sensor_pos.lon,
                "alt_m": sensor_pos.alt_m,
            },
            "power_dbm": power_dbm,
        }
        async with session.post(f"{SIMENGINE_URL}/rssi", json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["rssi_dbm"]
    except Exception as exc:
        logger.warning("node.rssi_query_failed", error=str(exc))
    return None


async def main() -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO").lower()
    logger.info(
        "node_sitl.starting",
        node_id=NODE_ID,
        node_type=NODE_TYPE,
        trajectory=TRAJECTORY_TYPE,
        freq_channels=[ch.center_hz for ch in FREQ_CHANNELS],
    )

    transport = get_transport()
    await transport.connect()

    trajectory = build_trajectory()
    start_time = time.monotonic()
    interval = 1.0 / max(UPDATE_RATE_HZ, 0.1)

    # State for status reporting
    rssi_count = 0
    last_status_time = start_time
    status_interval = 10.0  # publish NodeStatus every 10s

    # Wait for sim-engine to be ready
    async with aiohttp.ClientSession() as session:
        for attempt in range(30):
            try:
                async with session.get(f"{SIMENGINE_URL}/health") as resp:
                    if resp.status == 200:
                        logger.info("node.simengine_ready")
                        break
            except Exception:
                pass
            logger.info("node.waiting_for_simengine", attempt=attempt + 1)
            await asyncio.sleep(2.0)

    logger.info("node.loop_starting", interval_s=interval)

    async with aiohttp.ClientSession() as session:
        while True:
            loop_start = time.monotonic()
            t = loop_start - start_time
            pos = trajectory.position_at(t)

            # Get list of emitters from sim-engine
            emitters = []
            try:
                async with session.get(f"{SIMENGINE_URL}/emitters") as resp:
                    if resp.status == 200:
                        emitters = await resp.json()
            except Exception as exc:
                logger.warning("node.emitters_fetch_failed", error=str(exc))

            # For each frequency channel and emitter, publish an RSSI measurement
            for channel in FREQ_CHANNELS:
                threshold_dbm = float(os.environ.get("FREQ_CLUSTER_THRESHOLD_DBM", "-90"))

                if emitters:
                    for emitter in emitters:
                        # Skip emitters whose frequency is outside this channel's bandwidth
                        emitter_freq = float(emitter.get("freq_hz", channel.center_hz))
                        half_bw = channel.bandwidth_hz / 2.0
                        if abs(emitter_freq - channel.center_hz) > half_bw:
                            continue
                        rssi = await query_rssi_direct(
                            session,
                            emitter["lat"],
                            emitter["lon"],
                            emitter["alt_m"],
                            emitter["power_dbm"],
                            pos,
                            float(channel.center_hz),
                        )
                        if rssi is not None and rssi > threshold_dbm:
                            msg = RssiMessage(
                                node_id=NODE_ID,
                                node_type=NODE_TYPE,  # type: ignore[arg-type]
                                timestamp_utc=datetime.now(timezone.utc),
                                position=pos,
                                freq_channel=channel,
                                rssi_dbm=rssi,
                                snr_db=rssi - (threshold_dbm - 10),
                                is_simulated=True,
                            )
                            topic = f"rssi/{NODE_ID}"
                            await transport.publish(topic, msg)
                            rssi_count += 1
                            logger.debug(
                                "node.published_rssi",
                                node_id=NODE_ID,
                                freq_hz=channel.center_hz,
                                rssi_dbm=round(rssi, 1),
                                lat=round(pos.lat, 6),
                                lon=round(pos.lon, 6),
                                alt_m=round(pos.alt_m, 1),
                            )
                else:
                    # No emitters registered — publish noise floor
                    noise_floor = threshold_dbm - 5.0
                    msg = RssiMessage(
                        node_id=NODE_ID,
                        node_type=NODE_TYPE,  # type: ignore[arg-type]
                        timestamp_utc=datetime.now(timezone.utc),
                        position=pos,
                        freq_channel=channel,
                        rssi_dbm=noise_floor,
                        snr_db=-5.0,
                        is_simulated=True,
                    )
                    await transport.publish(f"rssi/{NODE_ID}", msg)

            # Periodic NodeStatus
            now = time.monotonic()
            if now - last_status_time >= status_interval:
                elapsed = now - last_status_time
                rate = rssi_count / max(elapsed, 1.0)
                status = NodeStatus(
                    node_id=NODE_ID,
                    timestamp_utc=datetime.now(timezone.utc),
                    node_type=NODE_TYPE,  # type: ignore[arg-type]
                    position=pos,
                    sdr_ok=True,
                    gps_ok=True,
                    comm_ok=await transport.healthcheck(),
                    rssi_rate_hz=round(rate, 2),
                )
                await transport.publish(f"node/status/{NODE_ID}", status)
                rssi_count = 0
                last_status_time = now

            # Sleep for remainder of interval
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, interval - elapsed)
            await asyncio.sleep(sleep_time)


if __name__ == "__main__":
    asyncio.run(main())
