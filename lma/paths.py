from pathlib import Path
import platform
from  datetime import datetime
from lma.schemas import SessionPaths
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


# For session dirs

SESSIONS_DIR = (
    APP_SUPPORT_DIR
    / "sessions"
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

    SESSIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def create_session_dir(session_id:str|None = None)->SessionPaths:

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M"
    )

    meetings_dir = (
        SESSIONS_DIR /
        f"meeting_{timestamp}_{session_id}"
    )
    audio_dir = (meetings_dir) / "audio"
    transcripts_dir = (meetings_dir) / "transcripts"
    logs_dir = (meetings_dir) / "logs"

    meetings_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    audio_dir.mkdir(parents=True,exist_ok=True)
    transcripts_dir.mkdir(parents=True,exist_ok=True)
    logs_dir.mkdir(parents=True,exist_ok=True)

    return SessionPaths(root=meetings_dir,audio=audio_dir,transcripts=transcripts_dir,logs=logs_dir)

# Main application cache

CACHE_DIR = get_cache_dir()


# Runtime directories

MODEL_DIR = CACHE_DIR / "models"

LOG_DIR = CACHE_DIR / "logs"

TEMP_DIR = CACHE_DIR / "temp"


# HuggingFace cache

HF_CACHE_DIR = MODEL_DIR / "huggingface"