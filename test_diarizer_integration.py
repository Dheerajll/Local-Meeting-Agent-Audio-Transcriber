from pathlib import Path

import mlx_whisper

from lma.transcription.diarizer import Diarizer
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


WHISPER_MODEL = ("mlx-community/whisper-large-v3-mlx")


snapshot_path = get_model_snapshot_path(WHISPER_MODEL)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading components...")

    diarizer = Diarizer()

    speaker_manager = SpeakerManager(
        similarity_threshold=0.70,
    )

    aligner = WordAligner()

    print("✓ Components loaded")

    for chunk_index, audio_path in enumerate(
        CHUNKS,
        start=1,
    ):

        print()
        print("=" * 80)
        print(f"CHUNK {chunk_index}")
        print(audio_path)
        print("=" * 80)

        # -------------------------------------------------
        # DIARIZATION
        # -------------------------------------------------

        print()
        print("Running diarization...")

        diarization = diarizer.diarize(
            audio_path
        )

        print("✓ Diarization completed")

        print(
            f"Local speakers: "
            f"{diarization.speakers}"
        )

        print(
            f"Embeddings: "
            f"{diarization.embeddings.shape}"
        )

        # -------------------------------------------------
        # SPEAKER MANAGER
        # -------------------------------------------------

        print()
        print("Matching global speakers...")

        local_to_global = {}

        for index, local_speaker in enumerate(
            diarization.speakers
        ):

            (
                global_speaker_id,
                similarity,
                is_new,
            ) = speaker_manager.assign_speaker(
                diarization.embeddings[index]
            )

            local_to_global[
                local_speaker
            ] = global_speaker_id

            print(
                f"{local_speaker:<15}"
                f" -> GLOBAL_{global_speaker_id:<3}"
                f" similarity={similarity:.4f}"
                f" "
                f"[{'NEW' if is_new else 'MATCH'}]"
            )

        # -------------------------------------------------
        # CONVERT DIARIZATION SEGMENTS
        #
        # local speaker:
        #
        #     SPEAKER_00
        #
        # becomes:
        #
        #     GLOBAL_0
        # -------------------------------------------------

        global_segments = []

        for segment in diarization.segments:

            global_speaker_id = local_to_global[
                segment.speaker_id
            ]

            global_segments.append(
                (
                    segment.start_ms,
                    segment.end_ms,
                    global_speaker_id,
                )
            )

        # -------------------------------------------------
        # WHISPER
        # -------------------------------------------------

        print()
        print("Running Whisper...")

        whisper_result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=snapshot_path,
            temperature=0.0,
            condition_on_previous_text=True,
            word_timestamps=True
        )

        print("✓ Whisper completed")

        # -------------------------------------------------
        # WORD ALIGNMENT
        # -------------------------------------------------

        print()
        print("Aligning words...")

        aligned_words = aligner.align(
            whisper_result,
            global_segments,
        )

        print(
            f"✓ Aligned "
            f"{len(aligned_words)} words"
        )

        # -------------------------------------------------
        # PRINT RESULT
        # -------------------------------------------------

        print()
        print("=" * 80)
        print("SPEAKER-LABELLED TRANSCRIPT")
        print("=" * 80)

        current_speaker = None

        for word in aligned_words:

            if (
                word.speaker_id
                != current_speaker
            ):

                if current_speaker is not None:
                    print()

                print(
                    f"\n[GLOBAL_{word.speaker_id}] ",
                    end="",
                )

                current_speaker = (
                    word.speaker_id
                )

            print(
                word.word,
                end=" ",
            )

        print()

       


if __name__ == "__main__":
    main()