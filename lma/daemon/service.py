"""
Daemon service — main loop and lifecycle management.

Orchestrates the daemon's lifecycle:
    connect → wait for command → ack → disconnect
    → run meeting → reconnect → repeat

Handles:
- Configuration validation
- Reconnection with exponential backoff
- Graceful shutdown
"""

import asyncio

from websockets.exceptions import ConnectionClosed

from lma.core.config import load_config
from lma.core.exceptions import ConfigError
from lma.daemon.connection import DaemonConnection
from lma.daemon.runner import run_meeting

# Reconnection settings
INITIAL_RECONNECT_DELAY = 2.0       # First retry after 2 seconds
MAX_RECONNECT_DELAY = 60.0          # Cap retries at 60 seconds
RECONNECT_BACKOFF_FACTOR = 2.0      # Double delay each attempt


class DaemonService:
    """
    Main daemon service.

    Runs a simple loop:
        1. Connect to backend control WebSocket
        2. Wait for join_meeting command
        3. Disconnect
        4. Run the meeting (blocking)
        5. Reconnect
        6. Repeat
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
                    # Connection closed without a command — retry
                    continue

                # Successful connection: reset backoff
                reconnect_delay = INITIAL_RECONNECT_DELAY

                # Step 4: Run the meeting (blocking)
                meeting_id = command.get("meeting_id")
                meeting_url = command.get("meeting_url")
                language = command.get("language", "en")

                run_meeting(meeting_id, meeting_url, language)

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


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def run_daemon() -> None:
    """Entry point for the `lma daemon` CLI command."""
    config = load_config()

    # Validate configuration before starting
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