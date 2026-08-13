from pathlib import Path

import mlx_whisper
import numpy as np
from pyannote.audio import Pipeline

from lma.transcription.speaker_manager import SpeakerManager
from lma.transcription.word_aligner import WordAligner
from lma.transcription.cache import get_model_snapshot_path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHUNKS = [
    Path("test_chunks/chunk_01.wav"),
    Path("test_chunks/chunk_02.wav"),
]


SPEAKER_SIMILARITY_THRESHOLD = 0.7


# ---------------------------------------------------------
# Existing mlx-whisper configuration
# ---------------------------------------------------------
#
# Replace these with the SAME values/config you already
# use in the worker/transcription code.
WHISPER_MODEL = ("mlx-community/whisper-large-v3-mlx")

DIARIZATION_MODEL = ("pyannote/speaker-diarization-community-1")

snapshot_path = get_model_snapshot_path(WHISPER_MODEL)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading diarization pipeline...")

    diarization_pipeline = (
        Pipeline.from_pretrained(
            DIARIZATION_MODEL
        )
    )

    print("✓ Diarization loaded")

    manager = SpeakerManager(
        similarity_threshold=(
            SPEAKER_SIMILARITY_THRESHOLD
        )
    )

    aligner = WordAligner()

    for chunk_index, audio_path in enumerate(
        CHUNKS,
        start=1,
    ):

        print()
        print("=" * 70)
        print(f"CHUNK {chunk_index}")
        print(audio_path)
        print("=" * 70)

        # -------------------------------------------------
        # Whisper
        # -------------------------------------------------

        print()
        print("Running mlx-whisper...")

        whisper_result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=snapshot_path,
            word_timestamps=True,
        )

        print("✓ Whisper completed")

        # -------------------------------------------------
        # Diarization
        # -------------------------------------------------

        print()
        print("Running diarization...")

        diarization_result = (
            diarization_pipeline(
                str(audio_path)
            )
        )

        print("✓ Diarization completed")

        # -------------------------------------------------
        # Extract diarization segments
        #
        # Convert:
        #
        #     local speaker
        #
        # into:
        #
        #     (start_ms, end_ms, local_speaker_id)
        # -------------------------------------------------

        diarization_segments = []

        local_speakers = list(
            diarization_result.speaker_diarization.labels()
        )

        # -------------------------------------------------
        # Establish local -> global speaker IDs.
        #
        # Use the speaker embeddings produced by the
        # diarization pipeline.
        # -------------------------------------------------

        embeddings = (
            diarization_result.speaker_embeddings
        )

        local_to_global = {}

        for index, local_speaker in enumerate(
            local_speakers
        ):

            global_speaker_id, similarity, is_new = (
                manager.assign_speaker(
                    embeddings[index]
                )
            )

            local_to_global[
                local_speaker
            ] = global_speaker_id

            print(
                f"{local_speaker:<15}"
                f" -> GLOBAL_{global_speaker_id:<3}"
                f" similarity={similarity:.4f}"
                f" "
                f"{'NEW' if is_new else 'MATCH'}"
            )

        # -------------------------------------------------
        # Build diarization segments using GLOBAL IDs.
        # -------------------------------------------------

        for segment, _, local_speaker in (
            diarization_result.speaker_diarization.itertracks(
                yield_label=True
            )
        ):

            global_speaker_id = (
                local_to_global[
                    local_speaker
                ]
            )

            diarization_segments.append(
                (
                    round(
                        segment.start * 1000
                    ),
                    round(
                        segment.end * 1000
                    ),
                    global_speaker_id,
                )
            )

        # -------------------------------------------------
        # Word alignment
        # -------------------------------------------------

        print()
        print("Aligning words...")

        aligned_words = aligner.align(
            whisper_result,
            diarization_segments,
        )

        print(
            f"✓ Aligned {len(aligned_words)} words"
        )

        # -------------------------------------------------
        # Transcript
        # -------------------------------------------------

        print()
        print("=" * 70)
        print("DIARIZED TRANSCRIPT")
        print("=" * 70)

        current_speaker = None
        current_words = []

        for word in aligned_words:

            if (
                current_speaker is not None
                and word.speaker_id
                != current_speaker
            ):

                print(
                    f"GLOBAL_{current_speaker}: "
                    f"{' '.join(current_words)}"
                )

                current_words = []

            current_speaker = (
                word.speaker_id
            )

            current_words.append(
                word.word
            )

        if current_words:

            print(
                f"GLOBAL_{current_speaker}: "
                f"{' '.join(current_words)}"
            )


if __name__ == "__main__":
    main()