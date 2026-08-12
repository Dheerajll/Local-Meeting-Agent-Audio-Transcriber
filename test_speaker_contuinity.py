from pathlib import Path

import numpy as np
from pyannote.audio import Pipeline

from lma.transcription.speaker_manager import SpeakerManager


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHUNKS = [
    Path("test_chunks/chunk_01.wav"),
    Path("test_chunks/chunk_02.wav"),
]



# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def extract_speaker_embeddings(
    diarization_output,
):
    """
    Returns:

        speakers
        embeddings

    using pyannote's speaker embeddings from the
    diarization output.
    """

    speakers = list(
        diarization_output.speaker_diarization.labels()
    )

    embeddings = diarization_output.speaker_embeddings

    return speakers, embeddings


def print_similarity_matrix(
    manager: SpeakerManager,
    speakers,
    embeddings,
):
    print()
    print("=" * 70)
    print("SIMILARITY MATRIX")
    print("=" * 70)

    print(
        f"{'LOCAL SPEAKER':<20}",
        end="",
    )

    for speaker in manager._speakers:
        print(
            f"{'GLOBAL ' + str(speaker.speaker_id):>14}",
            end="",
        )

    print()

    print(
        "-" * (
            20
            + 14 * len(manager._speakers)
        )
    )

    for index, local_speaker in enumerate(speakers):

        embedding = embeddings[index]

        print(
            f"{local_speaker:<20}",
            end="",
        )

        for global_speaker in manager._speakers:

            similarity = cosine_similarity(
                embedding,
                global_speaker.embedding,
            )

            print(
                f"{similarity:>14.4f}",
                end="",
            )

        print()


def print_best_matches(
    manager: SpeakerManager,
    speakers,
    embeddings,
):
    print()
    print("=" * 70)
    print("BEST MATCHES")
    print("=" * 70)

    for index, local_speaker in enumerate(speakers):

        embedding = embeddings[index]

        similarities = [
            cosine_similarity(
                embedding,
                global_speaker.embedding,
            )
            for global_speaker in manager._speakers
        ]

        best_index = int(
            np.argmax(similarities)
        )

        best_similarity = similarities[
            best_index
        ]

        best_global = manager._speakers[
            best_index
        ]

        matched = (
            best_similarity
            >= manager.similarity_threshold
        )

        status = (
            "MATCH"
            if matched
            else "NEW"
        )

        print(
            f"{local_speaker:<15}"
            f" -> GLOBAL_{best_global.speaker_id:<3}"
            f" similarity={best_similarity:.4f}"
            f"  {status}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading diarization pipeline...")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1"
    )

    print("✓ Pipeline loaded")

    manager = SpeakerManager(
        similarity_threshold=0.7,
    )

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

        diarization = pipeline(
            str(audio_path)
        )

        print("✓ Diarization completed")

        speakers, embeddings = (
            extract_speaker_embeddings(
                diarization
            )
        )

        print()
        print("Local speakers:")
        print(speakers)

        print()
        print("Embedding shape:")
        print(embeddings.shape)

        # -------------------------------------------------
        # First chunk
        #
        # Establish the meeting-level speaker identities.
        # -------------------------------------------------

        if chunk_index == 1:

            print()
            print("=" * 70)
            print("ESTABLISHING GLOBAL SPEAKERS")
            print("=" * 70)

            for index, local_speaker in enumerate(
                speakers
            ):

                speaker_id, similarity, is_new = (
                    manager.match(
                        embeddings[index]
                    )
                )

                print(
                    f"{local_speaker:<15}"
                    f" -> GLOBAL_{speaker_id:<3}"
                    f" similarity={similarity:.4f}"
                    f" new={is_new}"
                )

        # -------------------------------------------------
        # Following chunks
        #
        # Do NOT mutate the manager yet.
        # First inspect whether the local speakers
        # correspond to the existing global speakers.
        # -------------------------------------------------

        else:

            print_similarity_matrix(
                manager,
                speakers,
                embeddings,
            )

            print_best_matches(
                manager,
                speakers,
                embeddings,
            )

            print()
            print("=" * 70)
            print("ACTUAL SpeakerManager MATCHING")
            print("=" * 70)

            for index, local_speaker in enumerate(
                speakers
            ):

                speaker_id, similarity, is_new = (
                    manager.match(
                        embeddings[index]
                    )
                )

                print(
                    f"{local_speaker:<15}"
                    f" -> GLOBAL_{speaker_id:<3}"
                    f" similarity={similarity:.4f}"
                    f" new={is_new}"
                )


if __name__ == "__main__":
    main()