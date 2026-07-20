from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .constants import ChunkReason



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

    metadata: dict = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )



@dataclass(slots=True)
class MeetingSession:

    session_id: str

    meeting_url: str

    started_at: datetime

    browser_profile: Path