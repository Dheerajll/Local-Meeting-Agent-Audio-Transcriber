from pathlib import Path

import mlx_whisper
import numpy as np
from pyannote.audio import Pipeline

from lma.transcription.speaker_manager import SpeakerManager
from lma.transcription.cache import get_model_snapshot_path

# ============================================================
# CONFIG
# ============================================================

AUDIO_FILE = Path("test_chunks/chunk_01.wav")

WHISPER_MODEL = ("mlx-community/whisper-large-v3-mlx")

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

snapshot_path = get_model_snapshot_path(WHISPER_MODEL)


# ============================================================
# HELPERS
# ============================================================

def overlap_ms(
    start_ms: int,
    end_ms: int,
    speaker_start_ms: int,
    speaker_end_ms: int,
) -> int:

    start = max(start_ms, speaker_start_ms)
    end = min(end_ms, speaker_end_ms)

    return max(0, end - start)


def get_speaker_segments(diarization):
    """
    Convert pyannote diarization output into simple
    timestamped speaker segments.
    """

    annotation = diarization.exclusive_speaker_diarization

    segments = []

    for turn, _, speaker in annotation.itertracks(
        yield_label=True
    ):
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)

        segments.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker": speaker,
            }
        )

    return segments


def get_speaker_for_word(
    start_ms: int,
    end_ms: int,
    speaker_segments,
    speaker_mapping,
):
    """
    Assign a word to the speaker having the greatest
    temporal overlap with that word.
    """

    best_speaker = None
    best_overlap = 0

    for segment in speaker_segments:

        overlap = overlap_ms(
            start_ms,
            end_ms,
            segment["start_ms"],
            segment["end_ms"],
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = segment["speaker"]

    if best_speaker is None:
        return None, 0

    global_speaker = speaker_mapping.get(
        best_speaker
    )

    return global_speaker, best_overlap


def reconstruct_transcript(words):
    """
    Reconstruct:

        [speaker 0] text [speaker 1] text

    from word-level speaker assignments.
    """

    output = []

    current_speaker = None
    current_words = []

    for item in words:

        speaker = item["speaker"]
        word = item["word"]

        if speaker != current_speaker:

            if current_words:
                text = "".join(
                    current_words
                ).strip()

                if text:
                    output.append(
                        f"[speaker {current_speaker}] {text}"
                    )

            current_speaker = speaker
            current_words = []

        current_words.append(word)

    if current_words:

        text = "".join(
            current_words
        ).strip()

        if text:
            output.append(
                f"[speaker {current_speaker}] {text}"
            )

    return "\n".join(output)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("WORD-LEVEL SPEAKER ALIGNMENT TEST")
    print("=" * 70)

    print()
    print(f"Audio: {AUDIO_FILE}")

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_FILE}"
        )

    # --------------------------------------------------------
    # WHISPER
    # --------------------------------------------------------

    print()
    print("Loading Whisper...")

    result = mlx_whisper.transcribe(
        str(AUDIO_FILE),
        path_or_hf_repo=snapshot_path,
        word_timestamps=True,
    )

    print("✓ Whisper completed")

    print()
    print("=" * 70)
    print("WHISPER WORD TIMESTAMPS")
    print("=" * 70)

    all_words = []

    for segment in result["segments"]:

        print()
        print(
            f"{segment['start'] * 1000:7.0f}"
            f" → "
            f"{segment['end'] * 1000:7.0f}"
        )

        print(segment["text"])

        for word in segment.get("words", []):

            start_ms = int(
                word["start"] * 1000
            )

            end_ms = int(
                word["end"] * 1000
            )

            text = word["word"]

            print(
                f"    "
                f"{start_ms:7d}"
                f" → "
                f"{end_ms:7d}"
                f"  "
                f"{text!r}"
            )

            all_words.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "word": text,
                }
            )

    # --------------------------------------------------------
    # DIARIZATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Loading diarization pipeline...")
    print("=" * 70)

    pipeline = Pipeline.from_pretrained(
        DIARIZATION_MODEL
    )

    print("✓ Pipeline loaded")

    print()
    print("Running diarization...")

    diarization = pipeline(
        str(AUDIO_FILE),
        num_speaker = 3,

    )

    print("✓ Diarization completed")

    speaker_segments = get_speaker_segments(
        diarization
    )

    # --------------------------------------------------------
    # LOCAL SPEAKERS
    # --------------------------------------------------------

    local_speakers = sorted(
        {
            segment["speaker"]
            for segment in speaker_segments
        }
    )

    print()
    print("=" * 70)
    print("DIARIZATION")
    print("=" * 70)

    for segment in speaker_segments:

        print(
            f"{segment['start_ms']:7d}"
            f" → "
            f"{segment['end_ms']:7d}"
            f"   "
            f"{segment['speaker']}"
        )

    print()
    print("Local speakers:")
    print(local_speakers)

    # --------------------------------------------------------
    # GLOBAL SPEAKER MAPPING
    # --------------------------------------------------------

    manager = SpeakerManager()

    speaker_mapping = {}

    # We need embeddings for the actual SpeakerManager.
    #
    # pyannote's output contains speaker_embeddings,
    # ordered according to the local speaker identities.
    #
    # Build the local speaker → embedding mapping.
    # --------------------------------------------------------

    embeddings = np.asarray(
        diarization.speaker_embeddings
    )

    print()
    print("=" * 70)
    print("GLOBAL SPEAKER ASSIGNMENTS")
    print("=" * 70)

    for index, local_speaker in enumerate(
        local_speakers
    ):

        embedding = embeddings[index]

        global_id, similarity, is_new = (
            manager.match(embedding)
        )

        speaker_mapping[
            local_speaker
        ] = global_id

        print(
            f"{local_speaker:15}"
            f" → GLOBAL_{global_id}"
            f"   similarity={similarity:.4f}"
            f"   new={is_new}"
        )

    # --------------------------------------------------------
    # WORD → SPEAKER
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("WORD-LEVEL SPEAKER ALIGNMENT")
    print("=" * 70)

    aligned_words = []

    for item in all_words:

        speaker, overlap = (
            get_speaker_for_word(
                item["start_ms"],
                item["end_ms"],
                speaker_segments,
                speaker_mapping,
            )
        )

        item = {
            **item,
            "speaker": speaker,
            "overlap_ms": overlap,
        }

        aligned_words.append(item)

        print(
            f"{item['start_ms']:7d}"
            f" → "
            f"{item['end_ms']:7d}"
            f"  "
            f"[speaker {speaker}]"
            f"  "
            f"overlap={overlap:4d}ms"
            f"  "
            f"{item['word']!r}"
        )

    # --------------------------------------------------------
    # FINAL TRANSCRIPT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL ANNOTATED TRANSCRIPT")
    print("=" * 70)

    transcript = reconstruct_transcript(
        aligned_words
    )

    print()
    print(transcript)


if __name__ == "__main__":
    main()