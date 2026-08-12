from pathlib import Path

import mlx_whisper

from lma.transcription.cache import get_model_snapshot_path
from lma.transcription.word_aligner import WordAligner


# ============================================================
# CONFIG
# ============================================================

AUDIO_PATH = Path("test_chunks/chunk_01.wav")

WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"

snapshot_path = get_model_snapshot_path(
    WHISPER_MODEL
)


# ============================================================
# WHISPER
# ============================================================

def run_whisper():

    print("Loading Whisper...")

    result = mlx_whisper.transcribe(
        str(AUDIO_PATH),
        path_or_hf_repo=snapshot_path,
        language=None,
        temperature=0.0,
        condition_on_previous_text=True,
        word_timestamps=True,
    )

    print("✓ Whisper completed")

    return result


# ============================================================
# BUILD DIARIZATION INPUT
# ============================================================

def run_diarization():

    from pyannote.audio import Pipeline

    DIARIZATION_MODEL = (
        "pyannote/speaker-diarization-community-1"
    )

    print("\nLoading diarization pipeline...")

    pipeline = Pipeline.from_pretrained(
        DIARIZATION_MODEL
    )

    print("✓ Pipeline loaded")

    print("\nRunning diarization...")

    output = pipeline(
        str(AUDIO_PATH)
    )

    print("✓ Diarization completed")

    # --------------------------------------------------------
    # Handle new pyannote DiarizeOutput
    # --------------------------------------------------------

    if hasattr(
        output,
        "speaker_diarization",
    ):
        annotation = output.speaker_diarization

    elif hasattr(
        output,
        "annotation",
    ):
        annotation = output.annotation

    elif hasattr(
        output,
        "itertracks",
    ):
        annotation = output

    else:
        raise TypeError(
            "Unsupported diarization output: "
            f"{type(output).__name__}"
        )

    segments = []

    for turn, _, speaker in annotation.itertracks(
        yield_label=True
    ):

        start_ms = round(
            turn.start * 1000
        )

        end_ms = round(
            turn.end * 1000
        )

        segments.append(
            (
                start_ms,
                end_ms,
                speaker,
            )
        )

    return segments


# ============================================================
# CONVERT SPEAKER LABELS
# ============================================================

def build_speaker_mapping(
    diarization_segments,
):

    speakers = sorted(
        {
            speaker
            for _, _, speaker
            in diarization_segments
        }
    )

    return {
        speaker: index
        for index, speaker
        in enumerate(speakers)
    }


# ============================================================
# PRINT FINAL ALIGNMENT
# ============================================================

def print_alignment(
    aligned_words,
):

    print("\n" + "=" * 70)
    print("FINAL SPEAKER ALIGNMENT")
    print("=" * 70)

    if not aligned_words:
        print("No aligned words.")
        return

    current_speaker = None
    current_words = []

    for word in aligned_words:

        if (
            current_speaker is None
            or word.speaker_id == current_speaker
        ):

            current_speaker = (
                word.speaker_id
            )

            current_words.append(
                word.word
            )

        else:

            print(
                f"[speaker {current_speaker}] "
                f"{' '.join(current_words)}"
            )

            current_speaker = (
                word.speaker_id
            )

            current_words = [
                word.word
            ]

    if current_words:

        print(
            f"[speaker {current_speaker}] "
            f"{' '.join(current_words)}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Whisper
    # --------------------------------------------------------

    whisper_result = run_whisper()

    # --------------------------------------------------------
    # Diarization
    # --------------------------------------------------------

    diarization_segments = (
        run_diarization()
    )

    # --------------------------------------------------------
    # Convert pyannote labels to stable integer IDs
    #
    # This keeps the test output simple:
    #
    # SPEAKER_00 -> 0
    # SPEAKER_01 -> 1
    # SPEAKER_02 -> 2
    # --------------------------------------------------------

    speaker_mapping = (
        build_speaker_mapping(
            diarization_segments
        )
    )

    diarization_segments = [
        (
            start_ms,
            end_ms,
            speaker_mapping[speaker],
        )
        for (
            start_ms,
            end_ms,
            speaker,
        ) in diarization_segments
    ]

    # --------------------------------------------------------
    # Production aligner
    # --------------------------------------------------------

    aligner = WordAligner()

    aligned_words = aligner.align(
        whisper_result,
        diarization_segments,
    )

    # --------------------------------------------------------
    # ONLY PRINT THE RESULTING SPEAKER SEGMENTS
    # --------------------------------------------------------

    print_alignment(
        aligned_words
    )


if __name__ == "__main__":
    main()