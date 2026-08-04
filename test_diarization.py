from pathlib import Path

from pyannote.audio import Pipeline


AUDIO_FILE = (
    Path(__file__).parent
    / "chunk_005.wav"
)

MODEL = "pyannote/speaker-diarization-community-1"


def main():

    print(
        f"Audio file: {AUDIO_FILE}"
    )

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_FILE}"
        )

    print(
        "Loading diarization pipeline..."
    )

    pipeline = Pipeline.from_pretrained(
        MODEL,
    )

    print(
        "✓ Diarization pipeline loaded"
    )

    print(
        "Running diarization..."
    )

    result = pipeline(str(AUDIO_FILE))

    print(
        "✓ Diarization completed"
    )

    diarization = (
        result.exclusive_speaker_diarization
    )

    print(
        "\n--- SPEAKER SEGMENTS ---"
    )

    for turn, _, speaker in diarization.itertracks(
        yield_label=True
    ):

        print(
            f"{turn.start:8.2f}s → "
            f"{turn.end:8.2f}s   "
            f"{speaker}"
        )


if __name__ == "__main__":
    main()