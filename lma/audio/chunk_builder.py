from lma.schemas import AudioChunk, FinalizedChunk
from lma.constants import ChunkReason


class ChunkBuilder:
    """
    Builds immutable AudioChunk objects.

    Owns chunk numbering and converts finalized audio
    into an AudioChunk ready for publishing.
    """

    def __init__(self):
        self._next_chunk_id = 1

    def build(self,*,chunk: FinalizedChunk,start_ms: int,reason: ChunkReason,forced: bool,sample_rate: int,channels: int,) -> AudioChunk:

        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1

        end_ms = start_ms + chunk.effective_duration_ms

        return AudioChunk(
            chunk_id=chunk_id,
            pcm_bytes=chunk.pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
            start_ms=start_ms,
            end_ms=end_ms,
            overlap_ms=chunk.overlap_ms,
            reason=reason,
            forced=forced,
        )