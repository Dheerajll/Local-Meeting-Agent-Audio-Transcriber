import numpy as np

from lma.transcription.speaker_manager import SpeakerManager


def make_embedding(seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)

    embedding = rng.normal(
        size=256
    )

    embedding /= np.linalg.norm(
        embedding
    )

    return embedding.astype(
        np.float32
    )


manager = SpeakerManager(
    similarity_threshold=0.7
)


# --------------------------------------------------
# Simulated chunk 1
# --------------------------------------------------

speaker_a = make_embedding(1)
speaker_b = make_embedding(2)

global_a = manager.assign_speaker(
    speaker_a
)

global_b = manager.assign_speaker(
    speaker_b
)

print("Chunk 1")
print(
    "local SPEAKER_00 ->",
    global_a
)
print(
    "local SPEAKER_01 ->",
    global_b
)


# --------------------------------------------------
# Simulated chunk 2
#
# Pyannote's local labels are reversed.
# --------------------------------------------------

speaker_b_chunk_2 = (
    speaker_b
    + np.random.normal(
        0,
        0.01,
        size=speaker_b.shape,
    )
)

speaker_a_chunk_2 = (
    speaker_a
    + np.random.normal(
        0,
        0.01,
        size=speaker_a.shape,
    )
)


global_b_2 = manager.assign_speaker(
    speaker_b_chunk_2
)

global_a_2 = manager.assign_speaker(
    speaker_a_chunk_2
)


print()
print("Chunk 2")
print(
    "local SPEAKER_00 ->",
    global_b_2
)
print(
    "local SPEAKER_01 ->",
    global_a_2
)


# --------------------------------------------------
# Verify
# --------------------------------------------------

assert global_a == 0
assert global_b == 1

assert global_b_2 == 1
assert global_a_2 == 0

print()
print("✓ Speaker identity continuity test passed")