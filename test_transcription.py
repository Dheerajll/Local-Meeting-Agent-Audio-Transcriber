from pathlib import Path

from lma.schemas import AudioChunk
from lma.transcription.transcriber import WhisperTranscriber
from lma.constants import ChunkReason


MODEL_PATH = Path(
    "/Users/dheeraj/Library/Caches/local-meeting-agent/"
    "models/huggingface/"
    "models--mlx-community--whisper-large-v3-mlx/"
    "snapshots/49e6aa286ad60c14352c404340ded53710378a11"
)

AUDIO_FILE = Path("test_chunks/chunk_001.wav")


chunk = AudioChunk(
    chunk_id=1,
    pcm_bytes=b"",  # not used by transcriber
    sample_rate=16000,
    channels=1,
    start_ms=0,
    end_ms=60000,
    overlap_ms=0,
    reason=ChunkReason.NATURAL_SILENCE,
    forced=False,
)


transcriber = WhisperTranscriber(
    model_path=MODEL_PATH,
    language="ne",
)

transcript = transcriber.transcribe(
    AUDIO_FILE,
    chunk,
)

print()
print("Transcript:")
print(transcript)

print()
print("Text:", transcript.raw_text)
print("Confidence:", transcript.confidence)
print("Language:", transcript.language)
print("Chunk:", transcript.chunk_id)
print("Timing:", transcript.start_ms, "->", transcript.end_ms)