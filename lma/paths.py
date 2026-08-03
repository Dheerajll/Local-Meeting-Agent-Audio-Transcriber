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

SESSIONS = (
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

    SESSIONS.mkdir(
        parents=True,
        exist_ok=True
    )


def create_session_dir():

    timestamp = datetime.now().strftime(
        "%Y%m%d"
    )

    MEETINGS = (
        SESSIONS /
        f"meeting_{timestamp}"
    )
    AUDIO = (MEETINGS) / "audio"
    TRANSCRIPT = (MEETINGS) / "transcripts"
    LOGS = (MEETINGS) / "logs"

    MEETINGS.mkdir(
        parents=True,
        exist_ok=True
    )
    AUDIO.mkdir(parents=True,exist_ok=True)
    TRANSCRIPT.mkdir(parents=True,exist_ok=True)
    LOGS.mkdir(parents=True,exist_ok=True)

    return SessionPaths(root=MEETINGS,audio=AUDIO,transcripts=TRANSCRIPT,logs=LOGS)

# Main application cache

CACHE_DIR = get_cache_dir()


# Runtime directories

MODEL_DIR = CACHE_DIR / "models"

LOG_DIR = CACHE_DIR / "logs"

TEMP_DIR = CACHE_DIR / "temp"


# HuggingFace cache

HF_CACHE_DIR = MODEL_DIR / "huggingface"