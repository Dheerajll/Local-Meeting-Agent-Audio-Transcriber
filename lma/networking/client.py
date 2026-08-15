"""
WebSocket client for communicating with the MeetQ backend.

Handles:
- Connecting with LMA token authentication
- Sending transcript chunks as JSON
- Receiving ACKs from the backend
- Automatic reconnection on failure
- Graceful shutdown
"""

import json
import asyncio
import threading
import time
from dataclasses import asdict
from queue import Queue, Empty
import websockets
from websockets.exceptions import ConnectionClosed 

from lma.core.config import get_token, get_backend_url
from lma.core.schemas import TranscriptChunk
from lma.core.exceptions import ConfigError, BackendError


class BackendClient:
    """
    Manages a persistent WebSocket connection to the backend.

    Usage:
        client = BackendClient(meeting_id="abc123")
        client.start()
        client.send_chunk(transcript_chunk)
        ...
        client.stop()
    """

    def __init__(
        self,
        meeting_id: str,
        reconnect_interval: float = 5.0,
        max_reconnect_attempts: int = 10,
    ):
        self.meeting_id = meeting_id
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts

        self._send_queue: Queue = Queue()
        self._stop_event = threading.Event()
        self._connected = threading.Event()
        self._thread: threading.Thread | None = None
        self._ws = None

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> None:
        """Start the WebSocket client in a background thread."""
        if self._thread is not None:
            raise RuntimeError("BackendClient is already running.")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="backend-ws-client",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the WebSocket client."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def send_chunk(self, chunk: TranscriptChunk) -> None:
        """
        Queue a transcript chunk for sending to the backend.
        This is non-blocking — the actual send happens in the WS thread.
        """
        self._send_queue.put(chunk)

    def _run(self) -> None:
        """Main loop: connect, send, receive, reconnect."""
        attempts = 0

        while not self._stop_event.is_set():
            try:
                asyncio.run(self._connect_and_stream())
                # If we get here, connection closed cleanly
                break

            except (ConnectionClosed, OSError, BackendError) as exc:
                attempts += 1
                if attempts >= self.max_reconnect_attempts:
                    print(f"❌ Max reconnection attempts reached: {exc}")
                    break

                print(
                    f"⚠️  Connection lost ({exc}). "
                    f"Reconnecting in {self.reconnect_interval}s "
                    f"(attempt {attempts}/{self.max_reconnect_attempts})..."
                )
                self._connected.clear()
                self._stop_event.wait(self.reconnect_interval)

            except ConfigError as exc:
                print(f"❌ Configuration error: {exc}")
                break

            except Exception as exc:
                print(f"❌ Unexpected WebSocket error: {exc}")
                break

        self._connected.clear()

    async def _connect_and_stream(self) -> None:
        """Establish WebSocket connection and stream chunks."""
        token = get_token()
        backend_url = get_backend_url()

        # Convert http(s):// to ws(s)://
        ws_url = backend_url.replace("http://", "ws://").replace("https://", "wss://")
        uri = f"{ws_url}/ws/lma/{self.meeting_id}?token={token}"

        print(f"🔌 Connecting to backend: {ws_url}/ws/lma/{self.meeting_id}")

        async with websockets.connect(uri) as ws:
            self._ws = ws
            self._connected.set()
            print("✓ Connected to backend")

            # Send a handshake message
            await ws.send(json.dumps({
                "type": "handshake",
                "meeting_id": self.meeting_id,
            }))

            while not self._stop_event.is_set():
                # Check for chunks to send
                try:
                    chunk = self._send_queue.get(timeout=0.5)
                except Empty:
                    continue

                # Serialize and send
                payload = self._serialize_chunk(chunk)
                await ws.send(payload)

                # Wait for ACK (with timeout)
                try:
                    ack = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    ack_data = json.loads(ack)
                    if ack_data.get("status") == "ack":
                        print(f"  ✓ Chunk {chunk.chunk_id} acknowledged")
                    else:
                        print(f"  ⚠️  Chunk {chunk.chunk_id}: {ack_data}")
                except asyncio.TimeoutError:
                    print(f"  ⚠️  No ACK for chunk {chunk.chunk_id}")

                self._send_queue.task_done()

    @staticmethod
    def _serialize_chunk(chunk: TranscriptChunk) -> str:
        """Convert a TranscriptChunk to JSON for transmission."""
        data = asdict(chunk)
        data["reason"] = chunk.reason.value
        data["type"] = "transcript_chunk"
        return json.dumps(data, ensure_ascii=False)