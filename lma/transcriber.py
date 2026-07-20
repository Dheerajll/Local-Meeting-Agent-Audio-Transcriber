"""
Local speech-to-text via mlx-whisper.

Pure function: raw PCM bytes in, transcript out. No queues, no threading,
no file writing — this only knows how to turn audio into words. Apple
Silicon only (mlx-whisper), same as the rest of this project.
"""

import numpy as np
import mlx_whisper

from .constants import SAMPLE_RATE

MODEL_PATH = "models"  # cached during setup — setup.py isn't part of this commit


def transcribe_pcm(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> dict:
    """
    Transcribe raw 16-bit mono PCM audio.

    Returns whatever mlx_whisper.transcribe() gives back (text, language,
    segments, etc.) rather than reshaping it here — the caller decides what
    to keep and what to drop into TranscriptChunk.metadata.
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"transcribe_pcm expects {SAMPLE_RATE}Hz audio, got {sample_rate}Hz"
        )

    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL_PATH)


def average_confidence(result: dict) -> float:
    """
    mlx_whisper doesn't give a single confidence number — it gives
    per-segment avg_logprob. Convert that into a single 0-1 score by
    averaging across segments and exponentiating (logprob -> probability).
    Falls back to 0.0 if there are no segments (e.g. silence).
    """
    segments = result.get("segments") or []
    if not segments:
        return 0.0

    avg_logprob = sum(s.get("avg_logprob", 0.0) for s in segments) / len(segments)
    return float(np.exp(avg_logprob))