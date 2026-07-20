"""
Application-wide constants.
"""

from enum import Enum


# ---------- Audio ----------

SAMPLE_RATE = 16_000

CHANNELS = 1

SAMPLE_WIDTH = 2          # int16

FRAME_SAMPLES = 512

FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH


# ---------- Chunking ----------

OVERLAP_MS = 3_000

SOFT_LIMIT_MS = 40_000

HARD_LIMIT_MS = 60_000

RESUME_WINDOW_MS = 2_000

FINALIZE_SILENCE_MS = 5_000


# ---------- VAD ----------

VAD_THRESHOLD = 0.4

VAD_MIN_SILENCE_MS = 600

VAD_SPEECH_PAD_MS = 100


# ---------- Recorder ----------

class RecorderState(Enum):

    IDLE = "idle"

    RECORDING = "recording"

    WAITING_FOR_RESUME = "waiting_for_resume"



class ChunkReason(Enum):

    NATURAL_SILENCE = "natural_silence"

    SOFT_LIMIT = "soft_limit"

    HARD_LIMIT = "hard_limit"

    STREAM_ENDED = "stream_ended"


#========================================#
#New constants#

# ---------- Audio ----------

SAMPLE_RATE = 16_000

CHANNELS = 1

SAMPLE_WIDTH = 2

FRAME_SAMPLES = 512

FRAME_DURATION_MS = (
    FRAME_SAMPLES * 1000
) // SAMPLE_RATE

FRAME_BYTES = (
    FRAME_SAMPLES *
    SAMPLE_WIDTH
)

READ_SIZE = 4096