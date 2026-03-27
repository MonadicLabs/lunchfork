"""
master — Localisation + Inference + WebUI service.

Subscribes to RSSI measurements, clusters by frequency,
maintains sliding windows, runs the diffusion inference pipeline,
updates the particle filter, and serves a Leaflet.js WebUI.

Phase 1/2: clustering + sliding window + log
Phase 3:   + diffusion inference (stub or real model)
           + particle filter
           + TargetLocation publish
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure shared is importable
sys.path.insert(0, "/app")

from shared.comm import get_transport
from shared.geo import BBox, GeoPreprocessor
from shared.messages import (
    FreqChannel,
    NodePosition,
    NodeStatus,
    RadioMapUpdate,
    RssiMessage,
    TargetLocation,
)
from shared.models import DiffusionModel, SuperResolutionModel, UNetModel
from shared.models.grid_likelihood import GridLikelihoodModel, RssiObservation

# Pipeline imports — relative to containers/master/
sys.path.insert(0, str(Path(__file__).parent))
from pipeline.clustering import FreqClusterer
from pipeline.particle_filter import BBox as PFBBox
from pipeline.particle_filter import ParticleFilter
from pipeline.sliding_window import SlidingWindow

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

FREQ_CLUSTER_BW_HZ = int(os.environ.get("FREQ_CLUSTER_BW_HZ", "25000"))
FREQ_CLUSTER_THRESHOLD_DBM = float(os.environ.get("FREQ_CLUSTER_THRESHOLD_DBM", "-90"))
SLIDING_WINDOW_SEC = float(os.environ.get("SLIDING_WINDOW_SEC", "30"))
PF_N_PARTICLES = int(os.environ.get("PF_N_PARTICLES", "500"))
PF_MOTION_MODEL = os.environ.get("PF_MOTION_MODEL", "constant_velocity")
ZOOM_TRIGGER_STD_M = float(os.environ.get("ZOOM_TRIGGER_STD_M", "500"))
GRID_COARSE_SIZE = int(os.environ.get("GRID_COARSE_SIZE", "256"))
GRID_FINE_SIZE = int(os.environ.get("GRID_FINE_SIZE", "256"))
SR_FACTOR = int(os.environ.get("SR_FACTOR", "4"))

MODEL_DIFFUSION_PATH = os.environ.get("MODEL_DIFFUSION_PATH", "data/checkpoints/diffusion_vhf_v1.onnx")
MODEL_SR_PATH = os.environ.get("MODEL_SR_PATH", "data/checkpoints/sr_vhf_v1.onnx")
MODEL_UNET_PATH = os.environ.get("MODEL_UNET_PATH", "data/checkpoints/unet_uhf_v1.onnx")
TERRAIN_CACHE_DIR = os.environ.get("TERRAIN_CACHE_DIR", "data/terrain")

INFERENCE_INTERVAL_SEC = float(os.environ.get("INFERENCE_INTERVAL_SEC", "10.0"))
MIN_MEASUREMENTS_INFERENCE = int(os.environ.get("MIN_MEASUREMENTS_INFERENCE", "5"))

# Default search radius for coarse inference (metres)
COARSE_SEARCH_RADIUS_M = float(os.environ.get("COARSE_SEARCH_RADIUS_M", "5000"))

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").lower()
STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class AppState:
    """Shared mutable state for the master service."""

    def __init__(self) -> None:
        self.clusterer = FreqClusterer(cluster_bw_hz=FREQ_CLUSTER_BW_HZ)
        self.sliding_window = SlidingWindow(window_sec=SLIDING_WINDOW_SEC)
        self.particle_filter = ParticleFilter(
            n_particles=PF_N_PARTICLES,
            motion_model=PF_MOTION_MODEL,
        )

        # Grid likelihood model (always available — no checkpoint needed)
        self.grid_model = GridLikelihoodModel(terrain_cache_dir=TERRAIN_CACHE_DIR)

        # Lazy-loaded ML models and geo preprocessor
        self._diffusion_model: DiffusionModel | None = None
        self._unet_model: UNetModel | None = None
        self._sr_model: SuperResolutionModel | None = None
        self._geo: GeoPreprocessor | None = None

        # Whether real model checkpoints are available
        self.has_diffusion_model: bool = Path(MODEL_DIFFUSION_PATH).exists()
        self.has_unet_model: bool = Path(MODEL_UNET_PATH).exists()
        self.has_sr_model: bool = Path(MODEL_SR_PATH).exists()

        backend_name = (
            "diffusion" if self.has_diffusion_model
            else "unet" if self.has_unet_model
            else "grid_likelihood"
        )
        logger.info("master.inference_backend", backend=backend_name)

        # Track last-known node status
        self.node_states: dict[str, dict] = {}

        # Track estimated target locations per channel_id
        self.target_locations: dict[str, dict] = {}

        # Radio maps per channel_id (for API)
        self.radio_maps: dict[str, dict] = {}

        # WebSocket clients for push
        self.ws_clients: list[WebSocket] = []

        # Statistics
        self.total_rssi_messages = 0
        self.total_inferences = 0
        self.start_time = time.monotonic()

        # CommTransport (set during startup)
        self.transport = None

    @property
    def unet_model(self) -> UNetModel:
        if self._unet_model is None:
            self._unet_model = UNetModel(MODEL_UNET_PATH)
        return self._unet_model

    @property
    def diffusion_model(self) -> DiffusionModel:
        if self._diffusion_model is None:
            self._diffusion_model = DiffusionModel(MODEL_DIFFUSION_PATH)
        return self._diffusion_model

    @property
    def sr_model(self) -> SuperResolutionModel:
        if self._sr_model is None:
            self._sr_model = SuperResolutionModel(MODEL_SR_PATH)
        return self._sr_model

    @property
    def geo(self) -> GeoPreprocessor:
        if self._geo is None:
            self._geo = GeoPreprocessor(TERRAIN_CACHE_DIR)
        return self._geo


state = AppState()
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inference")

# ---------------------------------------------------------------------------
# WebSocket manager
# ---------------------------------------------------------------------------


async def broadcast(event: dict) -> None:
    """Broadcast a JSON event to all connected WebSocket clients."""
    dead: list[WebSocket] = []
    payload = json.dumps(event, default=str)
    for ws in list(state.ws_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Inference pipeline (runs in ThreadPoolExecutor)
# ---------------------------------------------------------------------------


_UNET_RSSI_FLOOR_DBM = -120.0  # must match training/generate_*_dataset.py
_UNET_RSSI_RANGE_DB  =   80.0  # normalises [-120, -40] dBm → [0, 1]


def _build_sparse_rssi(
    snapshot: list[RssiMessage],
    bbox: PFBBox,
    grid_size: int,
    threshold_dbm: float,
) -> list[tuple[float, float, float]]:
    """
    Convert RSSI measurements to sparse (x_px, y_px, rssi_norm) tuples
    for the UNet / diffusion model.

    rssi_norm uses the training normalisation: [-120, -40] dBm → [0, 1].
    Observations below threshold_dbm are dropped (noise floor filter).
    """
    points = []

    for msg in snapshot:
        if msg.rssi_dbm < threshold_dbm:
            continue
        # Map lat/lon to pixel coords
        x_px = (msg.position.lon - bbox.lon_min) / bbox.width_deg * grid_size
        y_px = (bbox.lat_max - msg.position.lat) / bbox.height_deg * grid_size
        rssi_norm = (msg.rssi_dbm - _UNET_RSSI_FLOOR_DBM) / _UNET_RSSI_RANGE_DB
        rssi_norm = float(np.clip(rssi_norm, 0.0, 1.0))
        points.append((x_px, y_px, rssi_norm))

    return points


def _build_rssi_observations(snapshot: list[RssiMessage]) -> list[RssiObservation]:
    """Convert a snapshot of RssiMessages to RssiObservation namedtuples."""
    return [
        RssiObservation(
            sensor_lat=msg.position.lat,
            sensor_lon=msg.position.lon,
            sensor_alt_m=msg.position.alt_m,
            rssi_dbm=msg.rssi_dbm,
            freq_hz=float(msg.freq_channel.center_hz),
        )
        for msg in snapshot
    ]


def _run_coarse_inference(
    channel_id: str,
    snapshot: list[RssiMessage],
    search_bbox: PFBBox,
) -> np.ndarray | None:
    """
    Run the coarse inference for a channel.
    Uses the diffusion model if a checkpoint is available, otherwise falls back
    to the grid likelihood model (Friis-based, no ML needed).
    Returns radio map [1, H, W] or None on failure.
    Runs synchronously in a ThreadPoolExecutor thread.
    """
    try:
        if not state.has_diffusion_model:
            if state.has_unet_model:
                # UNet path: single forward pass, much faster than diffusion
                from shared.geo import BBox as GeoBBox
                geo_bbox = GeoBBox(
                    lat_min=search_bbox.lat_min, lon_min=search_bbox.lon_min,
                    lat_max=search_bbox.lat_max, lon_max=search_bbox.lon_max,
                )
                cond   = state.geo.get_conditioning_tensor(geo_bbox, GRID_COARSE_SIZE)
                sparse = _build_sparse_rssi(
                    snapshot, search_bbox, GRID_COARSE_SIZE, FREQ_CLUSTER_THRESHOLD_DBM
                )
                radiomap = state.unet_model.infer(cond, sparse)
                logger.debug("inference.coarse_unet", channel_id=channel_id,
                             n_obs=len(sparse))
                return radiomap

            # Model-free fallback: Friis likelihood surface
            observations = _build_rssi_observations(snapshot)
            bbox_dict = search_bbox.as_dict()
            radiomap = state.grid_model.infer_from_observations(
                observations, bbox_dict, GRID_COARSE_SIZE
            )
            logger.debug(
                "inference.coarse_grid_fallback",
                channel_id=channel_id,
                n_obs=len(observations),
            )
            return radiomap

        # Diffusion model path
        from shared.geo import BBox as GeoBBox
        geo_bbox = GeoBBox(
            lat_min=search_bbox.lat_min,
            lon_min=search_bbox.lon_min,
            lat_max=search_bbox.lat_max,
            lon_max=search_bbox.lon_max,
        )

        cond = state.geo.get_conditioning_tensor(geo_bbox, GRID_COARSE_SIZE)
        sparse = _build_sparse_rssi(
            snapshot, search_bbox, GRID_COARSE_SIZE, FREQ_CLUSTER_THRESHOLD_DBM
        )
        radiomap = state.diffusion_model.infer(cond, sparse)
        return radiomap
    except Exception as exc:
        logger.error("inference.coarse_failed", channel_id=channel_id, error=str(exc))
        return None


def _run_fine_inference(
    channel_id: str,
    snapshot: list[RssiMessage],
    fine_bbox: PFBBox,
) -> np.ndarray | None:
    """
    Run fine-resolution inference + optional super-resolution.
    Falls back to grid likelihood when no diffusion checkpoint is available.
    Falls back to bilinear upscale when no SR checkpoint is available.
    Returns radio map [1, H*SR, W*SR] or None on failure.
    """
    try:
        if not state.has_diffusion_model:
            if state.has_unet_model:
                from shared.geo import BBox as GeoBBox
                geo_bbox = GeoBBox(
                    lat_min=fine_bbox.lat_min, lon_min=fine_bbox.lon_min,
                    lat_max=fine_bbox.lat_max, lon_max=fine_bbox.lon_max,
                )
                cond   = state.geo.get_conditioning_tensor(geo_bbox, GRID_FINE_SIZE)
                sparse = _build_sparse_rssi(
                    snapshot, fine_bbox, GRID_FINE_SIZE, FREQ_CLUSTER_THRESHOLD_DBM
                )
                rm_f = state.unet_model.infer(cond, sparse)
                logger.debug("inference.fine_unet", channel_id=channel_id,
                             n_obs=len(sparse))
            else:
                # Model-free fallback: higher-resolution Friis likelihood surface
                observations = _build_rssi_observations(snapshot)
                bbox_dict = fine_bbox.as_dict()
                rm_f = state.grid_model.infer_from_observations(
                    observations, bbox_dict, GRID_FINE_SIZE
                )
                logger.debug(
                    "inference.fine_grid_fallback",
                    channel_id=channel_id,
                    n_obs=len(observations),
                )
            # Bilinear upscale to SR resolution (no SR model needed)
            rm_hr_np = rm_f.squeeze()
            target_size = GRID_FINE_SIZE * SR_FACTOR
            rm_hr = np.zeros((1, target_size, target_size), dtype=np.float32)
            # Simple nearest-neighbour upscale via numpy (avoids scipy dependency)
            indices_y = np.linspace(0, rm_hr_np.shape[0] - 1, target_size)
            indices_x = np.linspace(0, rm_hr_np.shape[1] - 1, target_size)
            yi = np.round(indices_y).astype(int).clip(0, rm_hr_np.shape[0] - 1)
            xi = np.round(indices_x).astype(int).clip(0, rm_hr_np.shape[1] - 1)
            rm_hr[0] = rm_hr_np[np.ix_(yi, xi)]
            return rm_hr

        # Diffusion model path
        from shared.geo import BBox as GeoBBox
        geo_bbox = GeoBBox(
            lat_min=fine_bbox.lat_min,
            lon_min=fine_bbox.lon_min,
            lat_max=fine_bbox.lat_max,
            lon_max=fine_bbox.lon_max,
        )

        cond_f = state.geo.get_conditioning_tensor(geo_bbox, GRID_FINE_SIZE)
        sparse = _build_sparse_rssi(
            snapshot, fine_bbox, GRID_FINE_SIZE, FREQ_CLUSTER_THRESHOLD_DBM
        )
        rm_f = state.diffusion_model.infer(cond_f, sparse)

        if state.has_sr_model:
            mnt_hr = state.geo.get_mnt_hires(geo_bbox, GRID_FINE_SIZE * SR_FACTOR)
            rm_hr = state.sr_model.upscale(rm_f, mnt_hr)
        else:
            # No SR checkpoint: return fine map at original resolution
            rm_hr = rm_f

        return rm_hr
    except Exception as exc:
        logger.error("inference.fine_failed", channel_id=channel_id, error=str(exc))
        return None


def _estimate_search_bbox(snapshot: list[RssiMessage]) -> PFBBox | None:
    """
    Estimate the coarse search area from node positions in the snapshot.
    Uses a circle centred on the centroid of all node positions.
    """
    if not snapshot:
        return None

    lats = [m.position.lat for m in snapshot]
    lons = [m.position.lon for m in snapshot]
    clat = float(np.mean(lats))
    clon = float(np.mean(lons))

    return PFBBox.from_center(clat, clon, COARSE_SEARCH_RADIUS_M)


# ---------------------------------------------------------------------------
# Inference loop (background asyncio task)
# ---------------------------------------------------------------------------


async def inference_loop() -> None:
    """
    Periodically runs the diffusion + PF pipeline for each active channel.
    """
    loop = asyncio.get_running_loop()
    last_inference: dict[str, float] = {}

    logger.info("inference_loop.started", interval_sec=INFERENCE_INTERVAL_SEC)

    while True:
        await asyncio.sleep(INFERENCE_INTERVAL_SEC)

        channel_ids = state.clusterer.get_channel_states().keys()

        for channel_id in list(channel_ids):
            now = time.monotonic()
            last = last_inference.get(channel_id, 0.0)
            if now - last < INFERENCE_INTERVAL_SEC * 0.9:
                continue

            snapshot = state.sliding_window.get_snapshot(channel_id)
            if len(snapshot) < MIN_MEASUREMENTS_INFERENCE:
                logger.debug(
                    "inference_loop.skip_insufficient_data",
                    channel_id=channel_id,
                    n=len(snapshot),
                )
                continue

            last_inference[channel_id] = now

            # Determine search bbox
            pos_std_m = state.particle_filter.position_std_m(channel_id)

            if pos_std_m < ZOOM_TRIGGER_STD_M:
                # Zoomed inference: use PF confidence bbox
                search_bbox = state.particle_filter.get_confidence_bbox(channel_id, sigma=2.0)
                if search_bbox is None:
                    search_bbox = _estimate_search_bbox(snapshot)
                run_fine = True
            else:
                search_bbox = _estimate_search_bbox(snapshot)
                run_fine = False

            if search_bbox is None:
                continue

            logger.info(
                "inference_loop.running",
                channel_id=channel_id,
                n_measurements=len(snapshot),
                pos_std_m=round(pos_std_m, 0) if pos_std_m != float("inf") else -1,
                fine=run_fine,
            )

            if run_fine:
                radiomap = await loop.run_in_executor(
                    executor, _run_fine_inference, channel_id, snapshot, search_bbox
                )
                bbox_used = search_bbox
            else:
                radiomap = await loop.run_in_executor(
                    executor, _run_coarse_inference, channel_id, snapshot, search_bbox
                )
                bbox_used = search_bbox

            if radiomap is None:
                continue

            # Compute dt_sec since last PF update
            dt_sec = INFERENCE_INTERVAL_SEC

            # Update particle filter
            state.particle_filter.update(channel_id, radiomap, bbox_used, dt_sec)
            state.total_inferences += 1

            # Adapt sliding window
            from pipeline.sliding_window import TrackerState
            pf_std = state.particle_filter.position_std_m(channel_id)
            tracker_state_obj = TrackerState(position_std_m=pf_std)
            state.sliding_window.adapt(channel_id, tracker_state_obj)

            # Get target estimates
            targets = state.particle_filter.get_targets(channel_id)

            for target in targets:
                # Publish TargetLocation via CommTransport
                if state.transport is not None:
                    channel_state = state.clusterer.get_channel_states().get(channel_id)
                    if channel_state:
                        freq_ch = channel_state.to_freq_channel()
                        loc_msg = TargetLocation(
                            target_id=target["target_id"],
                            timestamp_utc=datetime.now(timezone.utc),
                            position=NodePosition(
                                lat=target["lat"],
                                lon=target["lon"],
                                alt_m=target["alt_m"],
                            ),
                            uncertainty_m=target["uncertainty_m"],
                            covariance=target["covariance"],
                            n_particles=target["n_particles"],
                            freq_channel=freq_ch,
                            track_state=target["track_state"],
                        )
                        try:
                            await state.transport.publish(
                                f"loc/target/{channel_id}", loc_msg
                            )
                        except Exception as exc:
                            logger.warning(
                                "inference_loop.publish_failed",
                                channel_id=channel_id,
                                error=str(exc),
                            )

                # Update in-memory target state
                state.target_locations[channel_id] = target

                # Push to WebSocket
                await broadcast(
                    {
                        "type": "target_location",
                        "target_id": target["target_id"],
                        "position": {
                            "lat": target["lat"],
                            "lon": target["lon"],
                            "alt_m": target["alt_m"],
                        },
                        "uncertainty_m": target["uncertainty_m"],
                        "track_state": target["track_state"],
                        "n_particles": target["n_particles"],
                    }
                )

            # Mark lost channels
            state.particle_filter.mark_lost(channel_id)

            # Store radio map for API
            rm_squeeze = radiomap.squeeze().astype(np.float32)
            rm_b64 = base64.b64encode(rm_squeeze.tobytes()).decode("ascii")
            state.radio_maps[channel_id] = {
                "channel_id": channel_id,
                "bbox": bbox_used.as_dict(),
                "shape": list(rm_squeeze.shape),
                "data_b64": rm_b64,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Broadcast channel updates
            await broadcast(
                {
                    "type": "channels_update",
                    "channels": [
                        ch.model_dump() for ch in state.clusterer.get_channels()
                    ],
                }
            )


# ---------------------------------------------------------------------------
# RSSI / NodeStatus message handlers
# ---------------------------------------------------------------------------


async def on_rssi_message(topic: str, payload: dict) -> None:
    """Handle incoming RssiMessage from any node."""
    try:
        msg = RssiMessage.model_validate(payload)
    except Exception as exc:
        logger.warning("master.rssi_parse_error", topic=topic, error=str(exc))
        return

    state.total_rssi_messages += 1

    # Cluster by frequency
    channel_id = state.clusterer.push(msg)

    # Add to sliding window
    state.sliding_window.push(channel_id, msg)

    logger.debug(
        "master.rssi_received",
        node_id=msg.node_id,
        channel_id=channel_id,
        freq_hz=msg.freq_channel.center_hz,
        rssi_dbm=round(msg.rssi_dbm, 1),
        n_window=len(state.sliding_window.get_snapshot(channel_id)),
    )

    # Push lightweight node position update to WebSocket clients
    await broadcast(
        {
            "type": "node_rssi",
            "node_id": msg.node_id,
            "node_type": msg.node_type,
            "lat": msg.position.lat,
            "lon": msg.position.lon,
            "alt_m": msg.position.alt_m,
            "rssi_dbm": msg.rssi_dbm,
            "freq_hz": msg.freq_channel.center_hz,
            "channel_id": channel_id,
            "timestamp": msg.timestamp_utc.isoformat(),
        }
    )


async def on_node_status(topic: str, payload: dict) -> None:
    """Handle incoming NodeStatus."""
    try:
        msg = NodeStatus.model_validate(payload)
    except Exception as exc:
        logger.warning("master.status_parse_error", topic=topic, error=str(exc))
        return

    state.node_states[msg.node_id] = {
        "node_id": msg.node_id,
        "node_type": msg.node_type,
        "position": msg.position.model_dump(),
        "sdr_ok": msg.sdr_ok,
        "gps_ok": msg.gps_ok,
        "comm_ok": msg.comm_ok,
        "rssi_rate_hz": msg.rssi_rate_hz,
        "last_seen": msg.timestamp_utc.isoformat(),
    }

    logger.info(
        "master.node_status",
        node_id=msg.node_id,
        rssi_rate_hz=msg.rssi_rate_hz,
        sdr_ok=msg.sdr_ok,
    )

    await broadcast({"type": "node_status", **state.node_states[msg.node_id]})


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated on_event)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("master.starting", log_level=LOG_LEVEL)

    transport = get_transport()
    await transport.connect()
    state.transport = transport
    logger.info("master.transport_connected")

    await transport.subscribe("rssi/+", on_rssi_message)
    await transport.subscribe("node/status/+", on_node_status)
    logger.info("master.subscribed")

    # Start inference loop
    inference_task = asyncio.create_task(inference_loop())

    yield

    # Shutdown
    inference_task.cancel()
    try:
        await inference_task
    except asyncio.CancelledError:
        pass

    await transport.disconnect()
    executor.shutdown(wait=False)
    logger.info("master.stopped")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="lunchfork master", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def api_health() -> dict:
    uptime_s = time.monotonic() - state.start_time
    channels = state.clusterer.get_channels()
    transport_ok = await state.transport.healthcheck() if state.transport else False
    return {
        "status": "ok",
        "uptime_s": round(uptime_s, 1),
        "total_rssi_messages": state.total_rssi_messages,
        "total_inferences": state.total_inferences,
        "n_channels": len(channels),
        "n_nodes": len(state.node_states),
        "n_targets": len(state.target_locations),
        "transport_ok": transport_ok,
        "channels": [ch.model_dump() for ch in channels],
        "window_stats": state.sliding_window.stats(),
    }


@app.get("/api/nodes")
async def api_nodes() -> list[dict]:
    return list(state.node_states.values())


@app.get("/api/targets")
async def api_targets() -> list[dict]:
    return list(state.target_locations.values())


@app.get("/api/radiomap/{freq_hz}")
async def api_radiomap(freq_hz: int) -> JSONResponse:
    channel_id = state.clusterer.get_channel_id_for_freq(freq_hz)
    if channel_id is None or channel_id not in state.radio_maps:
        return JSONResponse(
            status_code=404,
            content={"detail": "No radio map for this frequency"},
        )
    return JSONResponse(state.radio_maps[channel_id])


@app.get("/api/channels")
async def api_channels() -> list[dict]:
    ch_states = state.clusterer.get_channel_states()
    result = []
    for cid, cs in ch_states.items():
        snap = state.sliding_window.get_snapshot(cid)
        pf_std = state.particle_filter.position_std_m(cid)
        result.append(
            {
                "channel_id": cid,
                "center_hz": cs.center_hz,
                "bandwidth_hz": cs.bandwidth_hz,
                "label": cs.label,
                "message_count": cs.message_count,
                "n_window": len(snap),
                "pf_std_m": round(pf_std, 0) if pf_std != float("inf") else None,
                "last_seen": cs.last_seen.isoformat(),
            }
        )
    return result


@app.post("/api/scenario")
async def api_scenario(body: dict) -> dict:
    """Development endpoint: inject a scenario configuration."""
    logger.info("master.scenario_loaded", name=body.get("name", "unknown"))
    return {"status": "accepted", "scenario": body.get("name")}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    state.ws_clients.append(ws)
    logger.info("ws.client_connected", n_clients=len(state.ws_clients))
    try:
        # Send current state snapshot on connect
        await ws.send_json(
            {
                "type": "snapshot",
                "nodes": list(state.node_states.values()),
                "targets": list(state.target_locations.values()),
                "channels": [ch.model_dump() for ch in state.clusterer.get_channels()],
            }
        )
        while True:
            # Keep connection alive — handle incoming pings
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("ws.client_error", error=str(exc))
    finally:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)
        logger.info("ws.client_disconnected", n_clients=len(state.ws_clients))


# ---------------------------------------------------------------------------
# Static files (WebUI)
# ---------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        log_level=LOG_LEVEL,
        reload=False,
    )
