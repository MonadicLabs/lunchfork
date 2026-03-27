"""
shared.comm — CommTransport abstraction layer.

No component should import paho-mqtt or aiohttp directly.
Always use get_transport() to obtain the configured transport.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Awaitable, Callable

from shared.messages import BaseMessage


class CommTransport(ABC):
    """Abstract communication transport interface."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the broker / endpoint."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close the connection."""
        ...

    @abstractmethod
    async def publish(self, topic: str, message: BaseMessage) -> None:
        """Publish a message to the given topic."""
        ...

    @abstractmethod
    async def subscribe(
        self,
        topic_pattern: str,
        callback: Callable[[str, BaseMessage], Awaitable[None]],
    ) -> None:
        """
        Subscribe to a topic pattern.

        The callback receives (topic: str, raw_payload: dict).
        Wildcard patterns follow MQTT convention: + (single level), # (multi-level).
        """
        ...

    @abstractmethod
    async def healthcheck(self) -> bool:
        """Return True if the transport is connected and functional."""
        ...


def get_transport() -> CommTransport:
    """
    Read COMM_TRANSPORT from environment and return the configured implementation.

    COMM_TRANSPORT: mqtt | websocket
    COMM_BROKER_URL: broker URL (e.g. mqtt://localhost:1883 or ws://localhost:8082)
    """
    transport_type = os.environ.get("COMM_TRANSPORT", "mqtt").lower()

    if transport_type == "mqtt":
        from shared.comm.mqtt import MqttTransport
        return MqttTransport()

    if transport_type == "websocket":
        from shared.comm.websocket import WebSocketTransport
        return WebSocketTransport()

    raise ValueError(
        f"Unknown COMM_TRANSPORT: {transport_type!r}. "
        "Supported values: mqtt, websocket"
    )


__all__ = ["CommTransport", "get_transport"]
