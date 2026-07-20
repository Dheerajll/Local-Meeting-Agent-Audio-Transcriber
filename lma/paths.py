from pathlib import Path
import platform
from  datetime import datetime

APP_NAME = "local-meeting-agent"

def get_cache_dir():
    system = platform.system()

    if system == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Caches"
            / APP_NAME
        )
    elif system == "Linux":

        return (
            Path.home()
            / ".cache"
            / APP_NAME
        )
    elif system == "Windows":
        return (
            Path.home()
            / "AppData"
            / "Local"
            / APP_NAME
        )
    raise RuntimeError(
        f"Unsupported system: {system}"
    )

# For browser persistent context

APP_SUPPORT_DIR = (
    Path.home()
    / "Library"
    / "Application Support"
    / "local-meeting-agent"
)


BROWSER_PROFILE_DIR = (
    APP_SUPPORT_DIR
    / "browser-profile"
)


# For recording dirs

RECORDINGS_DIR = (
    APP_SUPPORT_DIR
    / "recordings"
)


def ensure_directories():

    APP_SUPPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    BROWSER_PROFILE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RECORDINGS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def create_session_dir():

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    session = (
        RECORDINGS_DIR /
        f"meeting_{timestamp}"
    )

    session.mkdir(
        parents=True,
        exist_ok=True
    )

    return session

# Main application cache

CACHE_DIR = get_cache_dir()


# Runtime directories

MODEL_DIR = CACHE_DIR / "models"

LOG_DIR = CACHE_DIR / "logs"

TEMP_DIR = CACHE_DIR / "temp"


# HuggingFace cache

HF_CACHE_DIR = MODEL_DIR / "huggingface"