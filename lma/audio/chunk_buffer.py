from lma.audio.buffer import RollingAudioBuffer
from lma.constants import (
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    OVERLAP_MS,
)
from lma.schemas import FinalizedChunk

BYTES_PER_MS = (SAMPLE_RATE * SAMPLE_WIDTH) // 1000


class ChunkBuffer:
    """
    Manages audio data belonging to one logical speech chunk.

    Responsibilities:
    - store current chunk audio
    - maintain previous overlap
    - calculate duration
    - finalize chunk audio
    """

    def __init__(self,overlap_ms: int = OVERLAP_MS):

        self.overlap_bytes = (overlap_ms * BYTES_PER_MS)

        self.current = RollingAudioBuffer()

        self.previous_overlap = RollingAudioBuffer()


    def start(self):
        """
        Start a new chunk using previous overlap.
        """
        self.current = RollingAudioBuffer(self.previous_overlap.read())


    def add(self, pcm: bytes):
        self.current.append(pcm)


    def duration_ms(self) -> int:

        return (len(self.current)//BYTES_PER_MS
        )


    def silence_duration_ms(self,last_speech_position: int) -> int:

        silence_bytes = (len(self.current)-last_speech_position)

        return (silence_bytes//BYTES_PER_MS)


    def finalize(self) -> FinalizedChunk:
        """
        Returns finished chunk audio and updates overlap.
        """

        audio = self.current.read()


        overlap = self.current.tail(self.overlap_bytes)

        overlap_ms = len(overlap) // BYTES_PER_MS

        self.previous_overlap.replace(overlap)

        duration_ms = len(audio) // BYTES_PER_MS

        self.current.clear()


        return FinalizedChunk(
            pcm_bytes = audio,
            duration_ms = duration_ms,
            overlap_ms = overlap_ms,
            effective_duration_ms= duration_ms - overlap_ms
        )


    def clear(self):

        self.current.clear()