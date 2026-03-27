"""
shared.comm.mqtt — MQTT transport implementation using paho-mqtt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from typing import Awaitable, Callable
from urllib.parse import urlparse

import paho.mqtt.client as mqtt
import structlog

from shared.comm import CommTransport
from shared.messages import BaseMessage

logger = structlog.get_logger(__name__)


def _mqtt_wildcard_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert MQTT wildcard pattern (+, #) to a Python regex."""
    escaped = re.escape(pattern)
    # + matches exactly one level (no slashes)
    escaped = escaped.replace(r"\+", r"[^/]+")
    # # matches zero or more levels (with optional leading slash)
    escaped = escaped.replace(r"\#", r".*")
    return re.compile(f"^{escaped}$")


class MqttTransport(CommTransport):
    """
    MQTT-based CommTransport using paho-mqtt.

    Configuration via environment variables:
      COMM_BROKER_URL  — e.g. mqtt://localhost:1883
    """

    def __init__(self) -> None:
        broker_url = os.environ.get("COMM_BROKER_URL", "mqtt://localhost:1883")
        parsed = urlparse(broker_url)
        self._host: str = parsed.hostname or "localhost"
        self._port: int = parsed.port or 1883
        self._client_id: str = f"lunchfork-{uuid.uuid4().hex[:8]}"

        self._client: mqtt.Client = mqtt.Client(
            client_id=self._client_id,
            protocol=mqtt.MQTTv311,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        self._connected_event: asyncio.Event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

        # Map from topic_pattern → (regex, callback)
        self._subscriptions: dict[
            str, tuple[re.Pattern[str], Callable[[str, dict], Awaitable[None]]]
        ] = {}

    # ------------------------------------------------------------------
    # paho callbacks (called from paho's background thread)
    # ------------------------------------------------------------------

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: object,
        flags: dict,
        rc: int,
    ) -> None:
        if rc == 0:
            logger.info("mqtt.connected", host=self._host, port=self._port)
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._connected_event.set)
            # Re-subscribe after reconnect
            for pattern in self._subscriptions:
                client.subscribe(pattern)
        else:
            logger.error("mqtt.connect_failed", rc=rc)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: object,
        rc: int,
    ) -> None:
        logger.warning("mqtt.disconnected", rc=rc)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._connected_event.clear)

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: object,
        msg: mqtt.MQTTMessage,
    ) -> None:
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("mqtt.bad_payload", topic=topic, error=str(exc))
            return

        for pattern, (regex, callback) in self._subscriptions.items():
            if regex.match(topic):
                if self._loop is not None:
                    asyncio.run_coroutine_threadsafe(callback(topic, payload), self._loop)

    # ------------------------------------------------------------------
    # CommTransport interface
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._client.connect_async(self._host, self._port, keepalive=60)
        self._client.loop_start()
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"Could not connect to MQTT broker at {self._host}:{self._port} within 30s"
            )

    async def disconnect(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("mqtt.transport_stopped")

    async def publish(self, topic: str, message: BaseMessage) -> None:
        payload = message.model_dump_json()
        result = self._client.publish(topic, payload, qos=1)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error("mqtt.publish_failed", topic=topic, rc=result.rc)

    async def subscribe(
        self,
        topic_pattern: str,
        callback: Callable[[str, dict], Awaitable[None]],
    ) -> None:
        regex = _mqtt_wildcard_to_regex(topic_pattern)
        self._subscriptions[topic_pattern] = (regex, callback)
        result, _ = self._client.subscribe(topic_pattern, qos=1)
        if result != mqtt.MQTT_ERR_SUCCESS:
            logger.error("mqtt.subscribe_failed", pattern=topic_pattern, rc=result)
        else:
            logger.info("mqtt.subscribed", pattern=topic_pattern)

    async def healthcheck(self) -> bool:
        return self._connected_event.is_set()
