"""
LMA configuration management.

Stores and retrieves:
- backend_url: The FastAPI backend address
- lma_token: The unique device token for authentication
- device_name: A human-readable label for this machine

Config file location:
~/Library/Application Support/local-meeting-agent/config.json
"""

import json
import platform
from pathlib import Path
from dataclasses import dataclass, field, asdict

from lma.core.paths import APP_SUPPORT_DIR
from lma.core.exceptions import ConfigError

CONFIG_FILE = APP_SUPPORT_DIR / "config.json"

# Default backend URL (local development)
DEFAULT_BACKEND_URL = "http://localhost:8000"


@dataclass
class LMAConfig:
    """Represents the LMA configuration."""
    backend_url: str = DEFAULT_BACKEND_URL
    lma_token: str | None = None
    device_name: str = field(
        default_factory=lambda: platform.node()
    )


def load_config() -> LMAConfig:
    """
    Load configuration from disk.
    Returns default config if file doesn't exist.
    """
    if not CONFIG_FILE.exists():
        return LMAConfig()

    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return LMAConfig(
            backend_url=data.get("backend_url", DEFAULT_BACKEND_URL),
            lma_token=data.get("lma_token"),
            device_name=data.get("device_name", platform.node()),
        )
    except (json.JSONDecodeError, KeyError) as exc:
        raise ConfigError(f"Corrupt config file: {exc}")


def save_config(config: LMAConfig) -> None:
    """
    Persist configuration to disk.
    Creates parent directories if needed.
    Sets file permissions to owner-only (600) for security.
    """
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    # Restrict permissions: owner read/write only
    CONFIG_FILE.chmod(0o600)


def get_token() -> str:
    """
    Get the LMA token. Raises ConfigError if not set.
    """
    config = load_config()
    if not config.lma_token:
        raise ConfigError(
            "LMA token not configured.\n"
            "Run: lma config set-token <your-token>"
        )
    return config.lma_token


def set_token(token: str) -> None:
    """Save the LMA token."""
    token = token.strip()
    if not token:
        raise ConfigError("Token cannot be empty.")

    config = load_config()
    config.lma_token = token
    save_config(config)


def get_backend_url() -> str:
    """Get the backend URL."""
    config = load_config()
    return config.backend_url


def set_backend_url(url: str) -> None:
    """Save the backend URL."""
    url = url.strip().rstrip("/")
    if not url:
        raise ConfigError("Backend URL cannot be empty.")

    config = load_config()
    config.backend_url = url
    save_config(config)


def get_device_name() -> str:
    """Get the device name."""
    config = load_config()
    return config.device_name


def is_configured() -> bool:
    """Check if the LMA has a token and backend URL set."""
    config = load_config()
    return bool(config.lma_token and config.backend_url)