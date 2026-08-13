from pathlib import Path

from lma.transcription.diarizer import Diarizer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHUNKS = [
    Path("test_chunks/chunk_01.wav"),
    Path("test_chunks/chunk_02.wav"),
]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading diarization pipeline...")

    diarizer = Diarizer()

    print("✓ Pipeline loaded")

    for chunk_index, audio_path in enumerate(
        CHUNKS,
        start=1,
    ):

        print()
        print("=" * 70)
        print(f"CHUNK {chunk_index}")
        print(audio_path)
        print("=" * 70)

        print()
        print("Running diarization...")

        result = diarizer.diarize(
            audio_path
        )

        print("✓ Diarization completed")

        # -------------------------------------------------
        # Speakers
        # -------------------------------------------------

        print()
        print("LOCAL SPEAKERS")
        print("-" * 70)

        for speaker in result.speakers:
            print(speaker)

        # -------------------------------------------------
        # Embeddings
        # -------------------------------------------------

        print()
        print("EMBEDDINGS")
        print("-" * 70)

        print(
            "shape:",
            result.embeddings.shape,
        )

        # -------------------------------------------------
        # Segments
        # -------------------------------------------------

        print()
        print("DIARIZATION SEGMENTS")
        print("-" * 70)

        for segment in result.segments:

            print(
                f"{segment.start_ms:>7} ms"
                f" -> "
                f"{segment.end_ms:<7} ms"
                f"  {segment.speaker_id}"
            )

        print()

        print(
            f"Total segments: "
            f"{len(result.segments)}"
        )

        print(
            f"Total speakers: "
            f"{len(result.speakers)}"
        )


if __name__ == "__main__":
    main()