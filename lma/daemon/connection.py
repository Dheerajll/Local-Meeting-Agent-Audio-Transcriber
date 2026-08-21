"""
Daemon WebSocket connection management.

Handles:
- Connecting to the backend control WebSocket
- Sending keepalive pings
- Receiving and parsing messages
- Sending command acknowledgments
- Graceful disconnection
"""

import asyncio
import json

import websockets #type:ignore
from websockets.exceptions import ConnectionClosed #type:ignore

from lma.core.config import LMAConfig

# Keepalive settings
PING_INTERVAL = 15  # Send application-level ping every 15 seconds


class DaemonConnection:
    """
    Manages the WebSocket connection to the backend control channel.

    Lifecycle:
        connect() → wait_for_command() → (auto-disconnects via context manager)
    """

    def __init__(self, config: LMAConfig):
        self._config = config

    @property
    def control_url(self) -> str:
        """Build the control WebSocket URL from config."""
        backend_url = self._config.backend_url
        ws_url = (
            backend_url
            .replace("http://", "ws://")
            .replace("https://", "wss://")
        )
        return f"{ws_url}/ws/lma/control?token={self._config.lma_token}"

    async def connect_and_wait_for_command(self) -> dict | None: #type:ignore
        """
        Connect to the backend control WebSocket and wait for a command.

        Returns:
            The command payload when received, or None if the
            connection closed without delivering a command.
        """
        print("🔌 Connecting to backend control channel...")

        try:
            # Disable websockets library's internal ping/pong
            # (same approach as BackendClient)
            async with websockets.connect(
                self.control_url,
                ping_interval=None,
                ping_timeout=None,
            ) as ws:
                print("✓ Connected. Waiting for commands...")

                last_ping_time = asyncio.get_event_loop().time()

                while True:
                    message = await self._receive_with_keepalive(
                        ws, last_ping_time
                    )

                    if message is None:
                        # Timeout — update ping timer and continue
                        last_ping_time = asyncio.get_event_loop().time()
                        continue

                    # Parse message
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    msg_type = payload.get("type")

                    # Keepalive response — ignore
                    if msg_type == "pong":
                        continue

                    # The only command we handle
                    if msg_type == "join_meeting":
                        meeting_id = payload.get("meeting_id")
                        print(
                            f"📥 Received join_meeting "
                            f"(meeting_id={meeting_id})"
                        )

                        # Acknowledge before disconnecting
                        await self._send_ack(ws, payload)

                        # Return command; context manager closes WebSocket
                        return payload

                    print(f"   Unknown message type: {msg_type}")

        except (ConnectionClosed, OSError, ConnectionRefusedError):
            raise  # Let the service layer handle reconnection

    async def _receive_with_keepalive(
        self,
        ws,
        last_ping_time: float,
    ) -> str | None:  #type:ignore
        """
        Wait for a message with a short timeout.
        If timeout expires, send a keepalive ping if needed.

        Returns:
            The received message string, or None if timeout expired.
        """
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=1.0)
            return message
        except asyncio.TimeoutError:
            # No message — send keepalive ping if needed
            now = asyncio.get_event_loop().time()
            if now - last_ping_time >= PING_INTERVAL:
                try:
                    await ws.send(json.dumps({"type": "ping"}))
                except ConnectionClosed:
                    raise
            return None

    async def _send_ack(self, ws, command: dict) -> None:
        """Send command acknowledgment back to the backend."""
        meeting_id = command.get("meeting_id")
        await ws.send(json.dumps({
            "type": "command_ack",
            "command": "join_meeting",
            "meeting_id": meeting_id,
            "status": "ok",
        }))
        print("📤 Sent command_ack")