from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .constants import ChunkReason

import numpy as np

@dataclass(slots=True)
class AudioChunk:

    chunk_id: int

    pcm_bytes: bytes

    sample_rate: int

    channels: int

    start_ms: int

    end_ms: int

    overlap_ms: int

    reason: ChunkReason

    forced: bool

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )



@dataclass(slots=True)
class TranscriptChunk:

    chunk_id: int

    raw_text: str

    confidence: float

    language: str

    start_ms: int

    end_ms: int

    reason: ChunkReason

    forced: bool




@dataclass(slots=True)
class MeetingSession:

    session_id: str

    meeting_url: str

    started_at: datetime

    browser_profile: Path

@dataclass(slots=True, frozen=True)
class PCMFrame:
    """
    One fixed-size PCM frame produced by FFmpegCapture.

    This is the smallest unit flowing through the audio pipeline.
    """

    data: bytes

    timestamp_ms: int


@dataclass(slots=True, frozen=True)
class FinalizedChunk:
    """
    Immutable snapshot returned by ChunkBuffer when a chunk is finalized.
    """

    pcm_bytes: bytes

    duration_ms: int

    overlap_ms: int

    effective_duration_ms : int 


@dataclass(slots=True)
class SessionPaths:

    root: Path

    audio: Path

    transcripts: Path

    logs: Path


@dataclass(slots=True)
class SpeakerSegment:
    start_ms: int
    end_ms: int
    speaker: str



@dataclass
class GlobalSpeaker:
    speaker_id: int
    embedding: np.ndarray
