"""
test_speaker_manager_centroid.py

Tests:

    Whisper
        ↓
    Pyannote diarization
        ↓
    Pyannote speaker_embeddings
        ↓
    SpeakerManager
        ↓
    local speaker -> global speaker
        ↓
    WordAligner
        ↓
    word-level speaker attribution

The important test is CROSS-CHUNK identity tracking.

Chunk 1 establishes the initial global speakers.

Chunk 2 gets completely new local speaker labels from pyannote.
SpeakerManager must determine which local speakers correspond
to existing global speakers using centroid embeddings.
"""

from pathlib import Path
import sys

import numpy as np


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from lma.transcription.speaker_manager import SpeakerManager
from lma.transcription.word_aligner import WordAligner


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_DIR = PROJECT_ROOT / "test_chunks"

CHUNK_FILES = [
    CHUNKS_DIR / "chunk_01.wav",
    CHUNKS_DIR / "chunk_02.wav",
]

SIMILARITY_THRESHOLD = 0.70


# ============================================================
# HELPERS
# ============================================================

def cosine_similarity(a, b):
    """
    Cosine similarity between two embeddings.
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / (a_norm * b_norm)
    )


def normalize_embedding(embedding):
    """
    L2 normalize an embedding.
    """

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


def print_separator(title=None):
    print()
    print("-" * 70)

    if title:
        print(title)

    print("-" * 70)


# ============================================================
# EXTRACT SPEAKER EMBEDDINGS
# ============================================================

def extract_speaker_embeddings(
    diarization_output,
):
    """
    Extract local speaker IDs and their corresponding
    pyannote speaker embeddings.

    Expected:

        speakers = [
            "SPEAKER_00",
            "SPEAKER_01",
            ...
        ]

        embeddings.shape == (
            number_of_speakers,
            embedding_dimension
        )
    """

    speakers = list(
        diarization_output.speaker_diarization.labels()
    )

    embeddings = np.asarray(
        diarization_output.speaker_embeddings
    )

    if embeddings.ndim != 2:
        raise RuntimeError(
            "Expected speaker_embeddings to be 2D, "
            f"got shape={embeddings.shape}"
        )

    if len(speakers) != len(embeddings):
        raise RuntimeError(
            "Speaker / embedding mismatch:\n"
            f"Speakers: {speakers}\n"
            f"Speaker count: {len(speakers)}\n"
            f"Embedding shape: {embeddings.shape}"
        )

    return speakers, embeddings


# ============================================================
# PRINT PYANNOTE EMBEDDINGS
# ============================================================

def print_embeddings(
    speakers,
    embeddings,
):
    print_separator(
        "PYANNOTE SPEAKER EMBEDDINGS"
    )

    print(
        f"Speaker count: {len(speakers)}"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    for index, speaker in enumerate(speakers):

        embedding = embeddings[index]

        print(
            f"{index:2d}. "
            f"{speaker:<12} "
            f"shape={embedding.shape} "
            f"norm={np.linalg.norm(embedding):.4f}"
        )


# ============================================================
# PRINT LOCAL DIARIZATION
# ============================================================

def print_local_diarization(
    diarization,
    chunk_name,
):
    print_separator(
        f"LOCAL DIARIZATION: {chunk_name}"
    )

    for index, (
        turn,
        _,
        speaker,
    ) in enumerate(
        diarization.itertracks(
            yield_label=True
        )
    ):

        start_ms = round(
            turn.start * 1000
        )

        end_ms = round(
            turn.end * 1000
        )

        duration_ms = (
            end_ms - start_ms
        )

        print(
            f"{index:2d}. "
            f"{start_ms:6d} → "
            f"{end_ms:6d} "
            f"[{speaker}] "
            f"duration={duration_ms:4d}ms"
        )


# ============================================================
# PRINT GLOBAL DIARIZATION
# ============================================================

def print_global_diarization(
    diarization,
    local_to_global,
    chunk_name,
):
    print_separator(
        f"GLOBAL DIARIZATION: {chunk_name}"
    )

    for index, (
        turn,
        _,
        local_speaker,
    ) in enumerate(
        diarization.itertracks(
            yield_label=True
        )
    ):

        global_speaker = (
            local_to_global[
                local_speaker
            ]
        )

        start_ms = round(
            turn.start * 1000
        )

        end_ms = round(
            turn.end * 1000
        )

        duration_ms = (
            end_ms - start_ms
        )

        print(
            f"{index:2d}. "
            f"{start_ms:6d} → "
            f"{end_ms:6d} "
            f"[{global_speaker}] "
            f"duration={duration_ms:4d}ms"
        )


# ============================================================
# PRINT GLOBAL SPEAKER STATE
# ============================================================

def print_global_speaker_state(
    speaker_manager,
):
    print_separator(
        "CURRENT GLOBAL SPEAKER STATE"
    )

    # This assumes the refactored SpeakerManager exposes
    # a global_speakers structure.
    #
    # If your implementation uses a different attribute,
    # change this section only.

    global_speakers = (
        speaker_manager.global_speakers
    )

    if not global_speakers:
        print("No global speakers.")
        return

    for global_id, state in (
        global_speakers.items()
    ):

        centroid = state["centroid"]

        count = state.get(
            "embedding_count",
            state.get(
                "count",
                1,
            ),
        )

        print(
            f"{global_id:<20} "
            f"embeddings={count:<4} "
            f"centroid_shape={centroid.shape} "
            f"centroid_norm="
            f"{np.linalg.norm(centroid):.4f}"
        )


# ============================================================
# LOAD WHISPER
# ============================================================

def load_whisper():
    """
    Replace this with your existing Whisper loader.

    This function is intentionally isolated so the test
    doesn't depend on the rest of the application.
    """

    from whisper_model import load_whisper_model

    print(
        "Loading Whisper model..."
    )

    model = load_whisper_model()

    print(
        "✓ Whisper model loaded"
    )

    return model


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_chunk(
    whisper_model,
    audio_path,
):
    print_separator(
        f"WHISPER: {audio_path.name}"
    )

    result = whisper_model.transcribe(
        str(audio_path),
        word_timestamps=True,
    )

    word_count = 0

    for segment in result.get(
        "segments",
        [],
    ):
        word_count += len(
            segment.get(
                "words",
                [],
            )
        )

    print(
        f"Whisper words: {word_count}"
    )

    return result


# ============================================================
# LOAD PYANNOTE
# ============================================================

def load_pyannote():
    """
    Replace the model name/path here with the exact pipeline
    loader used by your project.
    """

    from pyannote.audio import Pipeline

    print(
        "Loading pyannote pipeline..."
    )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
    )

    print(
        "✓ Pyannote pipeline loaded"
    )

    return pipeline


# ============================================================
# RUN DIARIZATION
# ============================================================

def diarize(
    pipeline,
    audio_path,
):
    print_separator(
        f"DIARIZATION: {audio_path.name}"
    )

    output = pipeline(
        str(audio_path)
    )

    print(
        f"Diarization output type: "
        f"{type(output).__name__}"
    )

    return output


# ============================================================
# BUILD GLOBAL DIARIZATION SEGMENTS
# ============================================================

def build_global_segments(
    diarization,
    local_to_global,
):
    """
    Convert pyannote's local labels into integer/global
    speaker IDs suitable for WordAligner.

    Returns:

        [
            (
                start_ms,
                end_ms,
                global_speaker_id
            ),
            ...
        ]
    """

    segments = []

    for (
        turn,
        _,
        local_speaker,
    ) in diarization.itertracks(
        yield_label=True
    ):

        global_name = (
            local_to_global[
                local_speaker
            ]
        )

        # Extract numeric part from:
        #
        # GLOBAL_SPEAKER_3
        #
        global_id = int(
            global_name.split("_")[-1]
        )

        segments.append(
            (
                round(
                    turn.start * 1000
                ),
                round(
                    turn.end * 1000
                ),
                global_id,
            )
        )

    segments.sort(
        key=lambda x: x[0]
    )

    return segments


# ============================================================
# PRINT WORD ALIGNMENT
# ============================================================

def print_word_alignment(
    aligned_words,
):
    print_separator(
        "FINAL WORD-LEVEL ALIGNMENT"
    )

    for word in aligned_words:

        print(
            f"{word.start_ms:6d} → "
            f"{word.end_ms:6d} "
            f"{word.word:<25} "
            f"[GLOBAL_SPEAKER_"
            f"{word.speaker_id}]"
        )


# ============================================================
# PRINT SPEAKER TEXT
# ============================================================

def print_speaker_text(
    aligned_words,
):
    print_separator(
        "SPEAKER TEXT"
    )

    grouped = {}

    for word in aligned_words:

        speaker = (
            word.speaker_id
        )

        grouped.setdefault(
            speaker,
            [],
        )

        grouped[speaker].append(
            word.word
        )

    for speaker in sorted(
        grouped
    ):

        text = " ".join(
            grouped[speaker]
        )

        print(
            f"[GLOBAL_SPEAKER_{speaker}] "
            f"{text}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "#" * 70
    )

    print(
        "# SPEAKER MANAGER + "
        "CENTROID EMBEDDING TEST"
    )

    print(
        "#" * 70
    )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    whisper_model = (
        load_whisper()
    )

    pipeline = (
        load_pyannote()
    )

    # --------------------------------------------------------
    # SPEAKER MANAGER
    # --------------------------------------------------------

    speaker_manager = (
        SpeakerManager(
            similarity_threshold=
                SIMILARITY_THRESHOLD
        )
    )

    print()
    print(
        "SpeakerManager initialized"
    )

    print(
        f"Similarity threshold: "
        f"{SIMILARITY_THRESHOLD}"
    )

    # --------------------------------------------------------
    # WORD ALIGNER
    # --------------------------------------------------------

    word_aligner = (
        WordAligner()
    )

    # --------------------------------------------------------
    # PROCESS CHUNKS
    # --------------------------------------------------------

    for chunk_index, audio_path in enumerate(
        CHUNK_FILES,
        start=1,
    ):

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: "
                f"{audio_path}"
            )

        print()
        print(
            "#" * 70
        )

        print(
            f"# PROCESSING CHUNK "
            f"{chunk_index}: "
            f"{audio_path.name}"
        )

        print(
            "#" * 70
        )

        # ====================================================
        # WHISPER
        # ====================================================

        whisper_result = (
            transcribe_chunk(
                whisper_model,
                audio_path,
            )
        )

        # ====================================================
        # DIARIZATION
        # ====================================================

        diarization_output = (
            diarize(
                pipeline,
                audio_path,
            )
        )

        diarization = (
            diarization_output
            .speaker_diarization
        )

        print_local_diarization(
            diarization,
            audio_path.name,
        )

        # ====================================================
        # LOCAL SPEAKERS
        # ====================================================

        local_speakers = list(
            diarization.labels()
        )

        print_separator(
            "LOCAL SPEAKERS"
        )

        for speaker in local_speakers:
            print(
                f"  {speaker}"
            )

        # ====================================================
        # EXTRACT EMBEDDINGS
        # ====================================================

        print_separator(
            "EXTRACTING SPEAKER EMBEDDINGS"
        )

        speakers, embeddings = (
            extract_speaker_embeddings(
                diarization_output
            )
        )

        print_embeddings(
            speakers,
            embeddings,
        )

        # ====================================================
        # LOCAL → GLOBAL
        # ====================================================

        print_separator(
            "LOCAL → GLOBAL SPEAKER MAPPING"
        )

        local_to_global = {}

        for local_index, local_speaker in enumerate(
            speakers
        ):

            embedding = (
                embeddings[
                    local_index
                ]
            )

            embedding = (
                normalize_embedding(
                    embedding
                )
            )

            # ------------------------------------------------
            # THIS IS THE CORE TEST
            #
            # SpeakerManager should:
            #
            #   1. compare embedding to all centroids
            #   2. choose highest similarity
            #   3. match if >= threshold
            #   4. otherwise create new speaker
            #   5. update centroid on match
            # ------------------------------------------------

            result = (
                speaker_manager
                .identify_speaker(
                    embedding
                )
            )

            # ------------------------------------------------
            # Expected result format:
            #
            # {
            #     "speaker_id":
            #         "GLOBAL_SPEAKER_0",
            #
            #     "similarity":
            #         0.91,
            #
            #     "is_new":
            #         False
            # }
            #
            # If your refactored SpeakerManager returns
            # a tuple instead, adapt these three lines.
            # ------------------------------------------------

            global_speaker = (
                result["speaker_id"]
            )

            similarity = (
                result.get(
                    "similarity",
                    0.0,
                )
            )

            is_new = (
                result.get(
                    "is_new",
                    False,
                )
            )

            local_to_global[
                local_speaker
            ] = global_speaker

            status = (
                "NEW"
                if is_new
                else "MATCH"
            )

            print(
                f"{local_speaker:<14} "
                f"→ "
                f"{global_speaker:<20} "
                f"similarity="
                f"{similarity:.4f} "
                f"[{status}]"
            )

        # ====================================================
        # PRINT MAPPING
        # ====================================================

        print_separator(
            "LOCAL → GLOBAL MAPPING"
        )

        for local_speaker in speakers:

            print(
                f"{local_speaker:<14} "
                f"→ "
                f"{local_to_global[local_speaker]}"
            )

        # ====================================================
        # GLOBAL DIARIZATION
        # ====================================================

        print_global_diarization(
            diarization,
            local_to_global,
            audio_path.name,
        )

        # ====================================================
        # WORD ALIGNMENT
        # ====================================================

        print_separator(
            "WORD ALIGNMENT"
        )

        global_segments = (
            build_global_segments(
                diarization,
                local_to_global,
            )
        )

        print(
            "WordAligner integration "
            "will use:"
        )

        for (
            start_ms,
            end_ms,
            speaker_id,
        ) in global_segments:

            print(
                f"{start_ms:6d} → "
                f"{end_ms:6d} "
                f"GLOBAL_SPEAKER_"
                f"{speaker_id}"
            )

        # ====================================================
        # ALIGN
        # ====================================================

        aligned_words = (
            word_aligner.align(
                whisper_result,
                global_segments,
            )
        )

        print_word_alignment(
            aligned_words
        )

        print_speaker_text(
            aligned_words
        )

        # ====================================================
        # SHOW CENTROIDS AFTER THIS CHUNK
        # ====================================================

        print_global_speaker_state(
            speaker_manager
        )

    # ========================================================
    # FINAL STATE
    # ========================================================

    print()
    print(
        "#" * 70
    )

    print(
        "# FINAL GLOBAL SPEAKER STATE"
    )

    print(
        "#" * 70
    )

    print_global_speaker_state(
        speaker_manager
    )

    print()
    print(
        "#" * 70
    )

    print(
        "# TEST COMPLETE"
    )

    print(
        "#" * 70
    )


if __name__ == "__main__":
    main()