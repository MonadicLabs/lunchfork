"""
master.pipeline.sliding_window — Per-channel circular buffer of RSSI measurements.

Maintains a sliding time window of RssiMessage objects per frequency channel.
Window depth is configurable; can be adapted dynamically based on tracker state.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from shared.messages import RssiMessage

logger = structlog.get_logger(__name__)

SLIDING_WINDOW_SEC = float(os.environ.get("SLIDING_WINDOW_SEC", "30"))
SLIDING_WINDOW_ADAPTIVE = os.environ.get("SLIDING_WINDOW_ADAPTIVE", "true").lower() == "true"

# Min/max window depth for adaptive mode
WINDOW_SEC_MIN = 5.0
WINDOW_SEC_MAX = 120.0


class TrackerState:
    """Minimal tracker state passed to SlidingWindow.adapt()."""

    def __init__(
        self,
        position_std_m: float,
        velocity_m_s: float = 0.0,
        acceleration_m_s2: float = 0.0,
    ) -> None:
        self.position_std_m = position_std_m
        self.velocity_m_s = velocity_m_s
        self.acceleration_m_s2 = acceleration_m_s2


class _ChannelBuffer:
    """Circular buffer of RssiMessage with time-based eviction."""

    def __init__(self, window_sec: float) -> None:
        self._window_sec = window_sec
        self._buf: deque["RssiMessage"] = deque()
        self._lock = threading.Lock()

    @property
    def window_sec(self) -> float:
        return self._window_sec

    @window_sec.setter
    def window_sec(self, value: float) -> None:
        self._window_sec = max(WINDOW_SEC_MIN, min(WINDOW_SEC_MAX, value))

    def push(self, msg: "RssiMessage") -> None:
        with self._lock:
            self._buf.append(msg)
            self._evict()

    def snapshot(self) -> list["RssiMessage"]:
        """Return a copy of the current window contents, oldest first."""
        with self._lock:
            self._evict()
            return list(self._buf)

    def __len__(self) -> int:
        with self._lock:
            self._evict()
            return len(self._buf)

    def _evict(self) -> None:
        """Remove measurements older than window_sec."""
        now = datetime.now(timezone.utc)
        cutoff_s = self._window_sec
        while self._buf:
            oldest = self._buf[0]
            # Handle both timezone-aware and naive datetimes
            ts = oldest.timestamp_utc
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_s = (now - ts).total_seconds()
            if age_s > cutoff_s:
                self._buf.popleft()
            else:
                break


class SlidingWindow:
    """
    Manages per-channel sliding windows of RSSI measurements.

    Usage:
        sw = SlidingWindow()
        sw.push("ch-abc123", msg)
        snapshot = sw.get_snapshot("ch-abc123")
        sw.adapt("ch-abc123", tracker_state)
    """

    def __init__(self, window_sec: float = SLIDING_WINDOW_SEC) -> None:
        self._default_window_sec = window_sec
        self._channels: dict[str, _ChannelBuffer] = {}
        self._lock = threading.Lock()

    def push(self, channel_id: str, msg: "RssiMessage") -> None:
        """Add a measurement to the channel buffer."""
        buf = self._get_or_create(channel_id)
        buf.push(msg)

    def get_snapshot(self, channel_id: str) -> list["RssiMessage"]:
        """
        Return all measurements in the current window for channel_id.
        Returns an empty list if the channel is unknown.
        """
        buf = self._get_or_create(channel_id)
        return buf.snapshot()

    def adapt(self, channel_id: str, tracker_state: TrackerState) -> None:
        """
        Dynamically adjust the window depth for channel_id based on tracker state.

        Rules:
        - High acceleration → shorten window (target is manoeuvring)
        - High velocity → shorten window
        - Low position uncertainty (converged) → shorten window to track closely
        - High uncertainty (init phase) → extend window to accumulate observations
        """
        if not SLIDING_WINDOW_ADAPTIVE:
            return

        buf = self._get_or_create(channel_id)
        current = buf.window_sec

        std_m = tracker_state.position_std_m
        vel = tracker_state.velocity_m_s
        acc = tracker_state.acceleration_m_s2

        if acc > 5.0 or vel > 30.0:
            # Fast-moving / manoeuvring target: short window
            new_window = max(WINDOW_SEC_MIN, current * 0.7)
        elif std_m < 200.0:
            # Converged: use moderate window for fine tracking
            new_window = max(10.0, min(30.0, current))
        elif std_m > 1000.0:
            # Initialising: extend window to gather more data
            new_window = min(WINDOW_SEC_MAX, current * 1.2)
        else:
            new_window = current  # no change

        if abs(new_window - current) > 1.0:
            logger.debug(
                "sliding_window.adapted",
                channel_id=channel_id,
                old_sec=round(current, 1),
                new_sec=round(new_window, 1),
                std_m=round(std_m, 0),
            )
            buf.window_sec = new_window

    def get_window_sec(self, channel_id: str) -> float:
        return self._get_or_create(channel_id).window_sec

    def channel_ids(self) -> list[str]:
        with self._lock:
            return list(self._channels.keys())

    def stats(self) -> dict[str, dict]:
        """Return per-channel statistics for debugging."""
        result = {}
        with self._lock:
            for cid, buf in self._channels.items():
                snap = buf.snapshot()
                result[cid] = {
                    "n_messages": len(snap),
                    "window_sec": buf.window_sec,
                    "n_nodes": len({m.node_id for m in snap}),
                }
        return result

    def _get_or_create(self, channel_id: str) -> _ChannelBuffer:
        with self._lock:
            if channel_id not in self._channels:
                self._channels[channel_id] = _ChannelBuffer(self._default_window_sec)
                logger.debug("sliding_window.new_channel", channel_id=channel_id)
            return self._channels[channel_id]
