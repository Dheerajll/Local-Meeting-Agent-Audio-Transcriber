from dataclasses import dataclass

import numpy as np
from lma.schemas import GlobalSpeaker

class SpeakerManager:
    """
    Maintains speaker identities across multiple audio chunks.

    Pyannote speaker labels are local to each diarization run.
    SpeakerManager converts them into meeting-level speaker IDs.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self._speakers: list[GlobalSpeaker] = []

    @staticmethod
    def _cosine_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(a, b) / denominator
        )

    def match(
        self,
        embedding: np.ndarray,
    ) -> tuple[int, float, bool]:

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if not self._speakers:
            speaker_id = 0

            self._speakers.append(
                GlobalSpeaker(
                    speaker_id=speaker_id,
                    embedding=embedding,
                )
            )

            return speaker_id, 1.0, True

        best_speaker = None
        best_similarity = -1.0

        for speaker in self._speakers:

            similarity = self._cosine_similarity(
                embedding,
                speaker.embedding,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker

        if (
            best_speaker is not None
            and best_similarity >= self.similarity_threshold
        ):
            return (
                best_speaker.speaker_id,
                best_similarity,
                False,
            )

        speaker_id = len(self._speakers)

        self._speakers.append(
            GlobalSpeaker(
                speaker_id=speaker_id,
                embedding=embedding,
            )
        )

        return (
            speaker_id,
            best_similarity,
            True,
        )