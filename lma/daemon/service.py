"""
Daemon service — main loop and lifecycle management.
"""

import asyncio

from websockets.exceptions import ConnectionClosed #type:ignore

from lma.core.config import load_config
from lma.core.exceptions import ConfigError
from lma.daemon.connection import DaemonConnection
from lma.daemon.runner import run_meeting

# Reconnection settings
INITIAL_RECONNECT_DELAY = 2.0
MAX_RECONNECT_DELAY = 60.0
RECONNECT_BACKOFF_FACTOR = 2.0


class DaemonService:
    """
    Main daemon service.
    """

    def __init__(self):
        self._config = load_config()
        self._connection = DaemonConnection(self._config)
        self._running = True

    async def run(self) -> None:
        """Main daemon loop."""
        print("🟢 LMA Daemon starting...")
        print(f"   Backend: {self._config.backend_url}")

        reconnect_delay = INITIAL_RECONNECT_DELAY

        while self._running:
            try:
                # Steps 1-3: Connect, wait for command, ack, disconnect
                command = await self._connection.connect_and_wait_for_command()

                if command is None:
                    continue

                # Successful connection: reset backoff
                reconnect_delay = INITIAL_RECONNECT_DELAY

                # Extract command details
                meeting_id = command.get("meeting_id")
                meeting_url = command.get("meeting_url")
                language = command.get("language", "en")

                # ──────────────────────────────────────────────
                # FIX: Run the meeting in a separate thread
                # ──────────────────────────────────────────────
                # Playwright's Sync API refuses to run inside an active
                # asyncio event loop. By using run_in_executor, we run
                # the blocking orchestrator in a thread pool thread,
                # which has no event loop — so Playwright works fine.
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None,run_meeting,meeting_id,meeting_url,language) #type:ignore

                print("🔄 Reconnecting to backend...\n")

            except (ConnectionClosed, OSError, ConnectionRefusedError) as e:
                if not self._running:
                    break
                print(f"⚠️  Connection error: {e}")
                print(f"   Retrying in {reconnect_delay:.0f}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(
                    reconnect_delay * RECONNECT_BACKOFF_FACTOR,
                    MAX_RECONNECT_DELAY,
                )

            except ConfigError as e:
                print(f"❌ Configuration error: {e}")
                break

            except Exception as e:
                if not self._running:
                    break
                print(f"❌ Unexpected error: {e}")
                print(f"   Retrying in {reconnect_delay:.0f}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(
                    reconnect_delay * RECONNECT_BACKOFF_FACTOR,
                    MAX_RECONNECT_DELAY,
                )

        print("👋 LMA Daemon stopped.")

    def stop(self) -> None:
        """Signal the daemon to stop."""
        self._running = False


def run_daemon() -> None:
    """Entry point for the `lma daemon` CLI command."""
    config = load_config()

    if not config.lma_token:
        print("✗ LMA token not configured.")
        print("  Run: lma config set-token <token>")
        return

    if not config.backend_url:
        print("✗ Backend URL not configured.")
        print("  Run: lma config set-backend <url>")
        return

    daemon = DaemonService()

    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        print("\n👋 LMA Daemon stopped.")