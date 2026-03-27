"""
master.pipeline.clustering — Frequency channel clustering.

Groups RssiMessage objects by detected frequency channel.
Creates new channels dynamically on first detection.
Merges channels that are within FREQ_CLUSTER_BW_HZ of each other.
"""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

# Avoid circular import by using TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.messages import FreqChannel, RssiMessage

logger = structlog.get_logger(__name__)

FREQ_CLUSTER_BW_HZ = int(os.environ.get("FREQ_CLUSTER_BW_HZ", "25000"))
FREQ_CLUSTER_THRESHOLD_DBM = float(os.environ.get("FREQ_CLUSTER_THRESHOLD_DBM", "-90"))


@dataclass
class ChannelState:
    """Internal state for a tracked frequency channel."""

    channel_id: str
    center_hz: int
    bandwidth_hz: int
    label: str | None
    first_seen: datetime
    last_seen: datetime
    message_count: int = 0

    def to_freq_channel(self) -> "FreqChannel":
        from shared.messages import FreqChannel
        return FreqChannel(
            center_hz=self.center_hz,
            bandwidth_hz=self.bandwidth_hz,
            label=self.label,
        )


class FreqClusterer:
    """
    Groups RSSI measurements by frequency channel.

    A new channel is created when a measurement arrives at a frequency
    more than FREQ_CLUSTER_BW_HZ away from any existing channel.

    Channels within FREQ_CLUSTER_BW_HZ of each other are merged
    (using the weighted-average centre frequency).

    Thread-safe via internal lock.
    """

    def __init__(self, cluster_bw_hz: int = FREQ_CLUSTER_BW_HZ) -> None:
        self._cluster_bw_hz = cluster_bw_hz
        self._channels: dict[str, ChannelState] = {}
        self._lock = threading.Lock()

    def push(self, msg: "RssiMessage") -> str:
        """
        Register a measurement and return its channel_id.

        Creates a new channel if no existing channel is within cluster_bw_hz.
        """
        if msg.rssi_dbm < FREQ_CLUSTER_THRESHOLD_DBM:
            # Below threshold — still assign to nearest channel but don't create new ones
            with self._lock:
                nearest = self._find_nearest_channel(msg.freq_channel.center_hz)
                if nearest:
                    return nearest.channel_id

        freq_hz = msg.freq_channel.center_hz

        with self._lock:
            nearest = self._find_nearest_channel(freq_hz)

            if nearest is not None:
                # Update existing channel (optionally merge)
                nearest.last_seen = msg.timestamp_utc
                nearest.message_count += 1
                # Update centre frequency using rolling average
                n = nearest.message_count
                nearest.center_hz = int((nearest.center_hz * (n - 1) + freq_hz) / n)
                channel_id = nearest.channel_id
            else:
                # Create new channel
                channel_id = f"ch-{uuid.uuid4().hex[:6]}"
                label = msg.freq_channel.label or _freq_label(freq_hz)
                state = ChannelState(
                    channel_id=channel_id,
                    center_hz=freq_hz,
                    bandwidth_hz=msg.freq_channel.bandwidth_hz,
                    label=label,
                    first_seen=msg.timestamp_utc,
                    last_seen=msg.timestamp_utc,
                    message_count=1,
                )
                self._channels[channel_id] = state
                logger.info(
                    "clusterer.new_channel",
                    channel_id=channel_id,
                    center_hz=freq_hz,
                    label=label,
                )

            # Merge nearby channels after update
            self._merge_nearby_channels()

        return channel_id

    def get_channels(self) -> list["FreqChannel"]:
        """Return the current list of detected frequency channels."""
        with self._lock:
            return [state.to_freq_channel() for state in self._channels.values()]

    def get_channel_states(self) -> dict[str, ChannelState]:
        """Return internal channel states (for debugging/API)."""
        with self._lock:
            return dict(self._channels)

    def get_channel_id_for_freq(self, freq_hz: int) -> str | None:
        """Return the channel_id for the given frequency, or None."""
        with self._lock:
            nearest = self._find_nearest_channel(freq_hz)
            return nearest.channel_id if nearest else None

    def _find_nearest_channel(self, freq_hz: int) -> ChannelState | None:
        """Return the nearest channel within cluster_bw_hz, or None."""
        best: ChannelState | None = None
        best_dist = float("inf")

        for state in self._channels.values():
            dist = abs(state.center_hz - freq_hz)
            if dist <= self._cluster_bw_hz and dist < best_dist:
                best = state
                best_dist = dist

        return best

    def _merge_nearby_channels(self) -> None:
        """
        Merge channels whose centres are within cluster_bw_hz of each other.

        Keeps the channel with the most messages; removes the other.
        """
        ids = list(self._channels.keys())
        merged: set[str] = set()

        for i in range(len(ids)):
            if ids[i] in merged:
                continue
            for j in range(i + 1, len(ids)):
                if ids[j] in merged:
                    continue
                a = self._channels[ids[i]]
                b = self._channels[ids[j]]
                if abs(a.center_hz - b.center_hz) <= self._cluster_bw_hz:
                    # Merge b into a (keep the one with more messages)
                    if b.message_count > a.message_count:
                        a, b = b, a
                        ids[i], ids[j] = ids[j], ids[i]
                    # Merge b into a
                    total = a.message_count + b.message_count
                    a.center_hz = int(
                        (a.center_hz * a.message_count + b.center_hz * b.message_count) / total
                    )
                    a.message_count = total
                    a.last_seen = max(a.last_seen, b.last_seen)
                    merged.add(ids[j])
                    logger.info(
                        "clusterer.merged_channels",
                        kept=a.channel_id,
                        removed=b.channel_id,
                    )

        for cid in merged:
            del self._channels[cid]


def _freq_label(freq_hz: int) -> str:
    """Generate a human-readable label for a frequency."""
    if freq_hz < 30_000_000:
        return f"HF-{freq_hz // 1_000_000}MHz"
    if freq_hz < 300_000_000:
        return f"VHF-{freq_hz // 1_000_000}MHz"
    if freq_hz < 3_000_000_000:
        return f"UHF-{freq_hz // 1_000_000}MHz"
    return f"SHF-{freq_hz // 1_000_000}MHz"
