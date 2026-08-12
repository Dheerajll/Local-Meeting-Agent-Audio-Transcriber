from pathlib import Path

from mlx_whisper.audio import load_audio
from mlx_whisper import transcribe

from pyannote.audio import Pipeline
from lma.transcription.cache import get_model_snapshot_path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

AUDIO_FILE = (Path(__file__).parent/ "test_chunks"/ "chunk_01.wav")

WHISPER_MODEL = ("mlx-community/whisper-large-v3-mlx")

DIARIZATION_MODEL = ("pyannote/speaker-diarization-community-1")

snapshot_path = get_model_snapshot_path(WHISPER_MODEL)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def overlap_ms(start_a: int,end_a: int,start_b: int,end_b: int) -> int:

    start = max(start_a, start_b)
    end = min(end_a, end_b)

    return max(0, end - start)


def find_speaker(start_ms: int,end_ms: int,speaker_segments: list[dict]) -> tuple[str | None, int]:

    best_speaker = None
    best_overlap = 0

    for segment in speaker_segments:

        overlap = overlap_ms(start_ms,end_ms,segment["start_ms"],segment["end_ms"])

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = segment["speaker"]

    return best_speaker, best_overlap


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading Whisper...")

    # We are deliberately using mlx_whisper.transcribe
    # because it already handles the audio preprocessing
    # and returns timestamped segments.
    whisper_result = transcribe(
        str(AUDIO_FILE),
        path_or_hf_repo=str(snapshot_path),
        language="en",
        temperature=0.0,
        condition_on_previous_text=True,
    )

    print("✓ Whisper completed")

    print()
    print("Loading diarization pipeline...")

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)

    print("✓ Diarization pipeline loaded")

    print()
    print("Running diarization...")

    diarization_result = pipeline(
        str(AUDIO_FILE)
    )

    print("✓ Diarization completed")

    # -----------------------------------------------------
    # Extract speaker annotation
    # -----------------------------------------------------

    annotation = (diarization_result.exclusive_speaker_diarization)

    speaker_segments = []

    for turn, _, speaker in annotation.itertracks(
        yield_label=True
    ):

        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)

        speaker_segments.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker": speaker,
            }
        )

    # -----------------------------------------------------
    # Print diarization
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("DIARIZATION")
    print("=" * 70)

    for segment in speaker_segments:

        print(
            f"{segment['start_ms']:6d} → "
            f"{segment['end_ms']:6d} "
            f"{segment['speaker']}"
        )

    # -----------------------------------------------------
    # Whisper segments
    # -----------------------------------------------------

    whisper_segments = whisper_result["segments"]

    print()
    print("=" * 70)
    print("WHISPER + SPEAKER ALIGNMENT")
    print("=" * 70)

    annotated_lines = []

    for segment in whisper_segments:

        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)

        text = segment["text"].strip()

        speaker, overlap = find_speaker(
            start_ms,
            end_ms,
            speaker_segments,
        )

        if speaker is None:
            speaker = "UNKNOWN"

        print()
        print(
            f"{start_ms:6d} → "
            f"{end_ms:6d} "
            f"{speaker:12s} "
            f"overlap={overlap:5d}ms"
        )

        print(f"  {text}")

        annotated_lines.append(
            f"[{speaker}] {text}"
        )

    # -----------------------------------------------------
    # Final text
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("ANNOTATED TRANSCRIPT")
    print("=" * 70)

    print()

    print(
        "\n".join(annotated_lines)
    )


if __name__ == "__main__":
    main()