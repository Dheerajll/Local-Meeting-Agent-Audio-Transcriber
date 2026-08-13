from pathlib import Path

from lma.core.schemas import AudioChunk
from lma.transcription.diarizer import Diarizer
from lma.transcription.speaker_manager import SpeakerManager
from lma.transcription.whisper_transcriber import WhisperTranscriber
from lma.transcription.word_aligner import WordAligner
from lma.core.constants import ChunkReason
from lma.transcription.cache import get_model_snapshot_path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_PATH = ("mlx-community/whisper-large-v3-mlx")

snapshot_path = get_model_snapshot_path(MODEL_PATH)

CHUNKS = [
    Path("test_chunks/chunk_01.wav"),
    Path("test_chunks/chunk_02.wav"),
]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading components...")

    diarizer = Diarizer()

    speaker_manager = SpeakerManager(
        similarity_threshold=0.70,
    )

    word_aligner = WordAligner()

    transcriber = WhisperTranscriber(
        model_path=snapshot_path,
        diarizer=diarizer,
        speaker_manager=speaker_manager,
        word_aligner=word_aligner,
    )

    print("✓ Components loaded")

    for index, audio_path in enumerate(
        CHUNKS,
        start=1,
    ):

        print()
        print("=" * 80)
        print(f"CHUNK {index}")
        print(audio_path)
        print("=" * 80)

        # -------------------------------------------------
        # Construct AudioChunk
        #
        # Adjust these fields if your AudioChunk schema
        # requires additional values.
        # -------------------------------------------------

        chunk = AudioChunk(
            chunk_id=0,
            pcm_bytes=b"",
           sample_rate=16000,
            channels=1,
            start_ms=0,
            end_ms=0,
            overlap_ms=300,
            reason=ChunkReason.NATURAL_SILENCE,
            forced=True,

        )

        print()
        print("Running transcription pipeline...")

        transcript = transcriber.transcribe(
            audio_path,
            chunk,
        )

        print("✓ Transcription completed")

        print()
        print("=" * 80)
        print("SPEAKER-LABELLED TRANSCRIPT")
        print("=" * 80)

        print(
            transcript.raw_text
        )

        print()
        print(
            f"confidence: "
            f"{transcript.confidence:.4f}"
        )

        print(
            f"language: "
            f"{transcript.language}"
        )


if __name__ == "__main__":
    main()