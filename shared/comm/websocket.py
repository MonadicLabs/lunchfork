"""
shared.comm.websocket — WebSocket-based CommTransport for development.

Acts as a simple hub: one server accepts connections and broadcasts.
Each client connects, subscribes by sending a JSON subscribe message,
and receives matching published messages.

COMM_BROKER_URL controls the hub address:
  ws://host:port  — connect to an existing hub
  ws://0.0.0.0:port — also starts a local hub (server mode)

For dev, run one process with COMM_WS_SERVER=true to start the hub.
Others connect as clients.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import Awaitable, Callable
from urllib.parse import urlparse

import aiohttp
import structlog

from shared.comm import CommTransport
from shared.messages import BaseMessage

logger = structlog.get_logger(__name__)

# Module-level in-process hub for single-process dev/test
_hub_subscribers: dict[
    str, list[tuple[re.Pattern[str], Callable[[str, dict], Awaitable[None]]]]
] = {}


def _mqtt_wildcard_to_regex(pattern: str) -> re.Pattern[str]:
    escaped = re.escape(pattern)
    escaped = escaped.replace(r"\+", r"[^/]+")
    escaped = escaped.replace(r"\#", r".*")
    return re.compile(f"^{escaped}$")


class WebSocketTransport(CommTransport):
    """
    WebSocket-based CommTransport for development.

    If COMM_WS_MODE=server (or no broker URL), uses an in-process hub.
    If COMM_WS_MODE=client, connects to a remote WebSocket hub.

    Message format over the wire:
      { "type": "publish", "topic": "...", "payload": { ... } }
    """

    def __init__(self) -> None:
        broker_url = os.environ.get("COMM_BROKER_URL", "")
        self._mode = os.environ.get("COMM_WS_MODE", "inprocess")
        self._client_id = f"ws-{uuid.uuid4().hex[:8]}"

        if broker_url and self._mode != "inprocess":
            parsed = urlparse(broker_url)
            self._ws_url = f"ws://{parsed.hostname}:{parsed.port or 8082}/ws"
        else:
            self._ws_url = ""

        # Local subscriptions: pattern → (regex, callback)
        self._subscriptions: dict[
            str, tuple[re.Pattern[str], Callable[[str, dict], Awaitable[None]]]
        ] = {}

        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._recv_task: asyncio.Task | None = None
        self._connected = False

    async def connect(self) -> None:
        if self._mode == "inprocess":
            self._connected = True
            logger.info("ws_transport.inprocess_mode", client_id=self._client_id)
            return

        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(self._ws_url)
            self._connected = True
            self._recv_task = asyncio.create_task(self._receive_loop())
            logger.info("ws_transport.connected", url=self._ws_url)
        except Exception as exc:
            logger.error("ws_transport.connect_failed", url=self._ws_url, error=str(exc))
            raise

    async def disconnect(self) -> None:
        self._connected = False
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()

    async def publish(self, topic: str, message: BaseMessage) -> None:
        payload = json.loads(message.model_dump_json())

        if self._mode == "inprocess":
            # Dispatch directly to registered callbacks
            await self._dispatch(topic, payload)
            return

        if self._ws is None or self._ws.closed:
            logger.warning("ws_transport.publish_no_connection", topic=topic)
            return

        envelope = {"type": "publish", "topic": topic, "payload": payload}
        await self._ws.send_json(envelope)

    async def subscribe(
        self,
        topic_pattern: str,
        callback: Callable[[str, dict], Awaitable[None]],
    ) -> None:
        regex = _mqtt_wildcard_to_regex(topic_pattern)
        self._subscriptions[topic_pattern] = (regex, callback)

        if self._mode != "inprocess" and self._ws and not self._ws.closed:
            await self._ws.send_json(
                {"type": "subscribe", "pattern": topic_pattern, "client_id": self._client_id}
            )
        logger.info("ws_transport.subscribed", pattern=topic_pattern)

    async def healthcheck(self) -> bool:
        return self._connected

    async def _dispatch(self, topic: str, payload: dict) -> None:
        for pattern, (regex, callback) in self._subscriptions.items():
            if regex.match(topic):
                try:
                    await callback(topic, payload)
                except Exception as exc:
                    logger.error(
                        "ws_transport.callback_error",
                        pattern=pattern,
                        topic=topic,
                        error=str(exc),
                    )

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        envelope = json.loads(msg.data)
                        if envelope.get("type") == "publish":
                            topic = envelope["topic"]
                            payload = envelope["payload"]
                            await self._dispatch(topic, payload)
                    except (json.JSONDecodeError, KeyError) as exc:
                        logger.warning("ws_transport.bad_envelope", error=str(exc))
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("ws_transport.recv_loop_error", error=str(exc))
