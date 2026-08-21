"""
LMA Daemon — Background service that listens for backend commands.

Public API:
    run_daemon() — Entry point for `lma daemon` CLI command
"""

from lma.daemon.service import run_daemon

__all__ = ["run_daemon"]