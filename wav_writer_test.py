from pathlib import Path

from lma.audio.wav_writer import WavWriter
from lma.schemas import AudioChunk
from lma.constants import ChunkReason
from lma.paths import create_session_dir
chunk = AudioChunk(
    chunk_id=1,
    pcm_bytes=b"\x00" * 32000,
    sample_rate=16000,
    channels=1,
    start_ms=0,
    end_ms=1000,
    overlap_ms=0,
    reason=ChunkReason.NATURAL_SILENCE,
    forced=False,
)
session = create_session_dir("01")
writer = WavWriter(Path(session.audio))

path = writer.write(chunk)

print(path)
print(path.exists())

writer.delete(path)

print(path.exists())