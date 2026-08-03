from pathlib import Path

from lma.constants import ChunkReason
from lma.schemas import TranscriptChunk
from lma.transcription.transcript_writer import TranscriptWriter


output_dir = Path("test_transcripts")

writer = TranscriptWriter(output_dir)

transcript = TranscriptChunk(
    chunk_id=1,
    raw_text="Hello everyone, let's start the meeting.",
    confidence=0.94,
    language="en",
    start_ms=1200,
    end_ms=4800,
    reason=ChunkReason.NATURAL_SILENCE,
    forced=False,
)

path = writer.write(transcript)

print("Created:", path)
print("Exists:", path.exists())

print()
print(path.read_text(encoding="utf-8"))