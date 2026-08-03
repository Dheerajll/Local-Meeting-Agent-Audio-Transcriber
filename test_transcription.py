from pathlib import Path

import mlx_whisper

from lma.transcription.whisper import WhisperTranscriber


MODEL_NAME = "mlx-community/whisper-large-v3-mlx"

CHUNK_1 = (
    Path(__file__).parent
    / "test_chunks"
    / "chunk_001.wav"
)

CHUNK_2 = (
    Path(__file__).parent
    / "test_chunks"
    / "chunk_002.wav"
)


transcriber = WhisperTranscriber(
    model_name=MODEL_NAME,
    language="ne",
)


print("\n--- CHUNK 1 ---")

text1 = transcriber.transcribe(
    CHUNK_1
)

print(text1)


print("\n--- CHUNK 2 ---")

text2 = transcriber.transcribe(
    CHUNK_2
)

print(text2)