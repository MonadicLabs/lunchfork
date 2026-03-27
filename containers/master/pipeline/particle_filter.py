"""
master.pipeline.particle_filter — Multi-target particle filter for RF localisation.

One ParticleFilter instance manages all tracked channels.
Each channel gets its own set of particles.

State vector per particle: [lat, lon, vlat, vlon]
  lat, lon  — position in decimal degrees
  vlat, vlon — velocity in degrees/second

Likelihood: bilinear interpolation of the radio map at each particle position.

Motion models:
  random_walk — no velocity, pure diffusion noise
  constant_velocity — CV model with process noise; switch to CT if curvature detected

Resampling: systematic resampling when ESS < N/2.
"""

from __future__ import annotations

import math
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read from env, overridable at construction)
# ---------------------------------------------------------------------------

PF_N_PARTICLES = int(os.environ.get("PF_N_PARTICLES", "500"))
PF_MOTION_MODEL = os.environ.get("PF_MOTION_MODEL", "constant_velocity")
PF_ADAPTIVE_MOTION = os.environ.get("PF_ADAPTIVE_MOTION", "true").lower() == "true"

# Process noise (degrees/step)
# Acts as jitter/kernel spreading after resampling.  Too small → particle impoverishment.
# Too large → particles escape the likelihood peak between updates.
# 0.0005° ≈ 55m: enough to spread resampled clusters over the Friis peak width,
# small enough not to escape the peak (Friis peak is ~100–500m wide at UHF).
SIGMA_POS_DEG = 0.0005       # ~55m per step
SIGMA_VEL_DEG_S = 0.000001   # ~0.1 m/s velocity drift per √s

# Minimum measurements before initialising particles
MIN_MEASUREMENTS_INIT = 3

# Degrees to metres conversion helpers
M_PER_LAT_DEG = 111320.0


def m_per_lon_deg(lat: float) -> float:
    return 111320.0 * math.cos(math.radians(lat))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# BBox helper (duplicated here to avoid circular import)
# ---------------------------------------------------------------------------


@dataclass
class BBox:
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

    @property
    def center_lat(self) -> float:
        return (self.lat_min + self.lat_max) / 2.0

    @property
    def center_lon(self) -> float:
        return (self.lon_min + self.lon_max) / 2.0

    @property
    def width_deg(self) -> float:
        return self.lon_max - self.lon_min

    @property
    def height_deg(self) -> float:
        return self.lat_max - self.lat_min

    def as_dict(self) -> dict:
        return {
            "lat_min": self.lat_min,
            "lon_min": self.lon_min,
            "lat_max": self.lat_max,
            "lon_max": self.lon_max,
        }

    @classmethod
    def from_center(cls, lat: float, lon: float, radius_m: float) -> "BBox":
        lat_delta = radius_m / M_PER_LAT_DEG
        lon_delta = radius_m / m_per_lon_deg(lat)
        return cls(
            lat_min=lat - lat_delta,
            lon_min=lon - lon_delta,
            lat_max=lat + lat_delta,
            lon_max=lon + lon_delta,
        )


# ---------------------------------------------------------------------------
# Particle state per channel
# ---------------------------------------------------------------------------


@dataclass
class ChannelPFState:
    """Particle filter state for one frequency channel."""

    channel_id: str
    n_particles: int
    # particles: [N, 4] — (lat, lon, vlat, vlon)
    particles: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float64))
    # weights: [N]
    weights: np.ndarray = field(default_factory=lambda: np.ones(0, dtype=np.float64))
    # Track state
    track_state: Literal["init", "confirmed", "lost"] = "init"
    update_count: int = 0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Velocity magnitude history for adaptive motion
    vel_history: list[float] = field(default_factory=list)

    def is_initialised(self) -> bool:
        return len(self.particles) == self.n_particles


# ---------------------------------------------------------------------------
# Core PF operations (stateless, operate on arrays)
# ---------------------------------------------------------------------------


def _latlon_to_pixel(
    lats: np.ndarray, lons: np.ndarray, bbox: BBox, h: int, w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lat/lon arrays to (row, col) pixel coordinates within bbox."""
    rows = (bbox.lat_max - lats) / bbox.height_deg * h
    cols = (lons - bbox.lon_min) / bbox.width_deg * w
    return rows, cols


def _bilinear_interp(
    radiomap: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """
    Bilinear interpolation of radiomap [H, W] at fractional (row, col) positions.

    Out-of-bounds positions get value 0 (lowest likelihood).
    """
    h, w = radiomap.shape

    rows = rows.clip(0, h - 1 - 1e-9)
    cols = cols.clip(0, w - 1 - 1e-9)

    r0 = np.floor(rows).astype(int)
    c0 = np.floor(cols).astype(int)
    r1 = (r0 + 1).clip(0, h - 1)
    c1 = (c0 + 1).clip(0, w - 1)

    dr = rows - r0
    dc = cols - c0

    vals = (
        radiomap[r0, c0] * (1 - dr) * (1 - dc)
        + radiomap[r0, c1] * (1 - dr) * dc
        + radiomap[r1, c0] * dr * (1 - dc)
        + radiomap[r1, c1] * dr * dc
    )

    # Clamp out-of-bounds particles
    oob = (rows <= 0) | (rows >= h - 1) | (cols <= 0) | (cols >= w - 1)
    vals = np.where(oob, 0.0, vals)

    return vals


def _systematic_resample(weights: np.ndarray) -> np.ndarray:
    """
    Systematic resampling. Returns indices of resampled particles.
    weights must be normalised and sum to 1.
    """
    n = len(weights)
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # fix floating point
    positions = (np.arange(n) + np.random.random()) / n
    indices = np.searchsorted(cumsum, positions)
    return indices.clip(0, n - 1)


def _ess(weights: np.ndarray) -> float:
    """Effective Sample Size."""
    return 1.0 / float(np.sum(weights**2) + 1e-30)


def _position_estimate(
    particles: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Weighted mean and standard deviation of particle positions.

    Returns (lat_mean, lon_mean, lat_std_m, lon_std_m).
    """
    lat_mean = float(np.sum(weights * particles[:, 0]))
    lon_mean = float(np.sum(weights * particles[:, 1]))

    lat_var = float(np.sum(weights * (particles[:, 0] - lat_mean) ** 2))
    lon_var = float(np.sum(weights * (particles[:, 1] - lon_mean) ** 2))

    lat_std_m = math.sqrt(max(lat_var, 0.0)) * M_PER_LAT_DEG
    lon_std_m = math.sqrt(max(lon_var, 0.0)) * m_per_lon_deg(lat_mean)

    return lat_mean, lon_mean, lat_std_m, lon_std_m


# ---------------------------------------------------------------------------
# ParticleFilter class
# ---------------------------------------------------------------------------


class ParticleFilter:
    """
    Multi-target particle filter.

    Usage:
        pf = ParticleFilter()
        pf.update(channel_id, radiomap, bbox)
        targets = pf.get_targets(channel_id)
    """

    def __init__(
        self,
        n_particles: int = PF_N_PARTICLES,
        motion_model: str = PF_MOTION_MODEL,
        adaptive_motion: bool = PF_ADAPTIVE_MOTION,
    ) -> None:
        self._n = n_particles
        self._motion_model = motion_model
        self._adaptive_motion = adaptive_motion
        self._channels: dict[str, ChannelPFState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        channel_id: str,
        radiomap: np.ndarray,
        bbox: BBox,
        dt_sec: float = 1.0,
    ) -> None:
        """
        Update the particle filter for channel_id with a new radio map.

        radiomap: [1, H, W] or [H, W] float32, values in [0, 1] (signal likelihood)
        bbox: geographic extent of the radio map
        dt_sec: time elapsed since last update (for motion model)
        """
        # Normalise radiomap to [H, W]
        rm = radiomap.squeeze()
        if rm.ndim != 2:
            logger.warning("pf.bad_radiomap_shape", shape=radiomap.shape)
            return

        h, w = rm.shape

        with self._lock:
            state = self._get_or_create(channel_id, bbox)

            if not state.is_initialised():
                self._initialise_particles(state, bbox)

            # Motion step: propagate particles
            self._motion_step(state, dt_sec)

            # Likelihood step: weight by radio map interpolation
            lats = state.particles[:, 0]
            lons = state.particles[:, 1]

            rows, cols = _latlon_to_pixel(lats, lons, bbox, h, w)
            likelihoods = _bilinear_interp(rm, rows, cols)

            # Avoid all-zero weights
            likelihoods = np.maximum(likelihoods, 1e-6)

            # Replace weights (not multiplicative) because GridLikelihoodModel and
            # DiffusionModel both produce a FULL posterior from the entire observation
            # window — multiplying would double-count previous observations.
            state.weights = likelihoods.copy()
            w_sum = state.weights.sum()
            if w_sum < 1e-30:
                # Weight collapse — reinitialise
                logger.warning("pf.weight_collapse", channel_id=channel_id)
                if state.track_state == "confirmed":
                    # Reinitialise around last confirmed estimate (not full bbox)
                    lat_c, lon_c, _, _ = _position_estimate(state.particles, state.weights)
                    recovery_bbox = BBox.from_center(lat_c, lon_c, 2000.0)
                    self._initialise_particles(state, recovery_bbox)
                else:
                    self._initialise_particles(state, bbox)
                state.weights = np.ones(self._n, dtype=np.float64) / self._n
            else:
                state.weights /= w_sum

            # Resample if ESS too low
            ess = _ess(state.weights)
            if ess < self._n / 2:
                indices = _systematic_resample(state.weights)
                state.particles = state.particles[indices].copy()
                state.weights = np.ones(self._n, dtype=np.float64) / self._n

            # Update track state
            state.update_count += 1
            state.last_update = datetime.now(timezone.utc)

            _, _, lat_std_m, lon_std_m = _position_estimate(state.particles, state.weights)
            pos_std_m = math.sqrt(lat_std_m**2 + lon_std_m**2) / math.sqrt(2)

            if state.update_count >= 3 and pos_std_m < 2000.0:
                state.track_state = "confirmed"
            elif state.update_count >= 1:
                state.track_state = "init"

            logger.debug(
                "pf.updated",
                channel_id=channel_id,
                update_count=state.update_count,
                ess=round(ess, 0),
                pos_std_m=round(pos_std_m, 0),
                track_state=state.track_state,
            )

    def get_targets(self, channel_id: str) -> list[dict]:
        """
        Return current target estimate for channel_id.

        Returns a list with zero or one dict containing position estimate.
        Dict keys match TargetLocation model (without constructing it here
        to avoid import cycles).
        """
        with self._lock:
            if channel_id not in self._channels:
                return []
            state = self._channels[channel_id]
            if not state.is_initialised() or state.update_count == 0:
                return []

            lat, lon, lat_std_m, lon_std_m = _position_estimate(
                state.particles, state.weights
            )
            uncertainty_m = math.sqrt(lat_std_m**2 + lon_std_m**2) / math.sqrt(2)

            # 2x2 covariance in metres²
            cov_m = [
                [lat_std_m**2, 0.0],
                [0.0, lon_std_m**2],
            ]

            return [
                {
                    "target_id": channel_id,
                    "timestamp_utc": state.last_update.isoformat(),
                    "lat": lat,
                    "lon": lon,
                    "alt_m": 0.0,
                    "uncertainty_m": uncertainty_m,
                    "covariance": cov_m,
                    "n_particles": self._n,
                    "track_state": state.track_state,
                }
            ]

    def get_confidence_bbox(self, channel_id: str, sigma: float = 2.0) -> BBox | None:
        """
        Return a BBox centred on the particle cloud, extending sigma standard deviations.
        Returns None if channel not initialised.
        """
        with self._lock:
            if channel_id not in self._channels:
                return None
            state = self._channels[channel_id]
            if not state.is_initialised():
                return None

            lat, lon, lat_std_m, lon_std_m = _position_estimate(
                state.particles, state.weights
            )
            radius_m = sigma * math.sqrt(lat_std_m**2 + lon_std_m**2) / math.sqrt(2)
            radius_m = max(radius_m, 500.0)  # minimum 500m zoom area

            return BBox.from_center(lat, lon, radius_m)

    def position_std_m(self, channel_id: str) -> float:
        """Return 1-sigma position uncertainty in metres, or inf if not initialised."""
        with self._lock:
            if channel_id not in self._channels:
                return float("inf")
            state = self._channels[channel_id]
            if not state.is_initialised():
                return float("inf")
            _, _, lat_std_m, lon_std_m = _position_estimate(
                state.particles, state.weights
            )
            return math.sqrt(lat_std_m**2 + lon_std_m**2) / math.sqrt(2)

    def channel_ids(self) -> list[str]:
        with self._lock:
            return list(self._channels.keys())

    def mark_lost(self, channel_id: str, timeout_sec: float = 120.0) -> None:
        """Mark channel as lost if no updates within timeout_sec."""
        with self._lock:
            if channel_id in self._channels:
                state = self._channels[channel_id]
                now = datetime.now(timezone.utc)
                age_s = (now - state.last_update).total_seconds()
                if age_s > timeout_sec:
                    state.track_state = "lost"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, channel_id: str, bbox: BBox) -> ChannelPFState:
        if channel_id not in self._channels:
            self._channels[channel_id] = ChannelPFState(
                channel_id=channel_id, n_particles=self._n
            )
            logger.info("pf.new_channel", channel_id=channel_id)
        return self._channels[channel_id]

    def _initialise_particles(self, state: ChannelPFState, bbox: BBox) -> None:
        """Uniformly initialise particles over the bbox."""
        n = self._n
        lats = np.random.uniform(bbox.lat_min, bbox.lat_max, n)
        lons = np.random.uniform(bbox.lon_min, bbox.lon_max, n)
        vlats = np.zeros(n, dtype=np.float64)
        vlons = np.zeros(n, dtype=np.float64)
        state.particles = np.stack([lats, lons, vlats, vlons], axis=1).astype(np.float64)
        state.weights = np.ones(n, dtype=np.float64) / n
        logger.info(
            "pf.initialised",
            channel_id=state.channel_id,
            n=n,
            bbox=bbox.as_dict(),
        )

    def _motion_step(self, state: ChannelPFState, dt_sec: float) -> None:
        """Propagate particles forward by dt_sec."""
        n = self._n
        particles = state.particles

        if self._motion_model == "random_walk":
            # Pure diffusion: add noise to position
            sigma_pos = SIGMA_POS_DEG * math.sqrt(dt_sec)
            particles[:, 0] += np.random.randn(n) * sigma_pos
            particles[:, 1] += np.random.randn(n) * sigma_pos

        else:  # constant_velocity
            # Position update from velocity
            particles[:, 0] += particles[:, 2] * dt_sec
            particles[:, 1] += particles[:, 3] * dt_sec

            # Process noise on position and velocity
            sigma_pos = SIGMA_POS_DEG * 0.1 * math.sqrt(dt_sec)
            sigma_vel = SIGMA_VEL_DEG_S * math.sqrt(dt_sec)

            particles[:, 0] += np.random.randn(n) * sigma_pos
            particles[:, 1] += np.random.randn(n) * sigma_pos
            particles[:, 2] += np.random.randn(n) * sigma_vel
            particles[:, 3] += np.random.randn(n) * sigma_vel

            # Track velocity magnitude
            vel_m_s = np.sqrt(
                (particles[:, 2] * M_PER_LAT_DEG) ** 2
                + (particles[:, 3] * m_per_lon_deg(float(particles[:, 0].mean()))) ** 2
            )
            state.vel_history.append(float(np.median(vel_m_s)))
            if len(state.vel_history) > 20:
                state.vel_history.pop(0)

        # Keep particles from wandering too far (hard clamp to 5° radius)
        particles[:, 0] = particles[:, 0].clip(-90, 90)
        particles[:, 1] = particles[:, 1].clip(-180, 180)


__all__ = ["ParticleFilter", "BBox", "haversine_m"]
