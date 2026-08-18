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

        ws_url = backend_url.replace("http://", "ws://").replace("https://", "wss://")
        uri = f"{ws_url}/ws/lma/{self.meeting_id}?token={token}"

        print(f"🔌 Connecting to backend: {ws_url}/ws/lma/{self.meeting_id}")

        # FIX 1: Completely disable the websockets library's internal ping/pong.
        # This prevents the 40-second timeout since Uvicorn doesn't always reply to them.
        async with websockets.connect(
            uri, 
            ping_interval=None, 
            ping_timeout=None
        ) as ws:
            self._ws = ws
            self._connected.set()
            print("✓ Connected to backend")

            # Send handshake and wait for ACK
            await ws.send(json.dumps({
                "type": "handshake",
                "meeting_id": self.meeting_id,
            }))
            
            try:
                handshake_resp = await asyncio.wait_for(ws.recv(), timeout=10.0)
                if json.loads(handshake_resp).get("status") == "handshake_ack":
                    print("✓ Handshake confirmed")
            except Exception as e:
                print(f"⚠️  Handshake issue: {e}")

            last_ping = asyncio.get_event_loop().time()
            PING_INTERVAL = 15  # Send application-level JSON ping every 15 seconds

            while not self._stop_event.is_set():
                
                # FIX 2: Use non-blocking get_nowait() so we don't freeze the asyncio event loop!
                try:
                    chunk = self._send_queue.get_nowait()
                except Empty:
                    # Send application-level keepalive ping
                    now = asyncio.get_event_loop().time()
                    if now - last_ping > PING_INTERVAL:
                        try:
                            await ws.send(json.dumps({"type": "ping"}))
                            last_ping = now
                        except websockets.exceptions.ConnectionClosed:
                            break
                    # Yield to the event loop briefly instead of blocking
                    await asyncio.sleep(0.1)
                    continue

                # Send the chunk
                payload = self._serialize_chunk(chunk)
                await ws.send(payload)

                # Wait for ACK (skip "pong" or other non-ack messages)
                try:
                    while True:
                        ack = await asyncio.wait_for(ws.recv(), timeout=15.0)
                        ack_data = json.loads(ack)
                        
                        if ack_data.get("status") == "pong":
                            continue 
                        break
                        
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