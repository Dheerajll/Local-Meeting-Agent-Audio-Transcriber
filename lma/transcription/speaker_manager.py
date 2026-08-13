from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GlobalSpeaker:
    """
    Persistent global speaker identity.

    embedding:
        Running centroid embedding.

    observation_count:
        Number of local speaker embeddings assigned
        to this global speaker.
    """

    speaker_id: int
    embedding: np.ndarray
    observation_count: int = 1


class SpeakerManager:
    """
    Maintains global speaker identities across audio chunks.

    Each local speaker embedding is compared against
    the centroid embedding of every known global speaker.

    If the best similarity is >= threshold:
        -> match existing global speaker
        -> update its centroid

    Otherwise:
        -> create a new global speaker
    """

    def __init__(
        self,
        similarity_threshold: float = 0.70,
    ):
        self.similarity_threshold = (
            similarity_threshold
        )

        self._speakers: dict[
            int, GlobalSpeaker
        ] = {}

        self._next_speaker_id = 0

    # ========================================================
    # ASSIGN SINGLE SPEAKER
    # ========================================================

    def assign_speaker(
        self,
        embedding: np.ndarray,
    ) -> tuple[int, float, bool]:
        """
        Assign one local speaker embedding to a global speaker.

        Returns:

            global_speaker_id
            similarity
            is_new
        """

        embedding = self._prepare_embedding(
            embedding
        )

        # First speaker in the system.
        if not self._speakers:

            speaker_id = self._create_speaker(
                embedding
            )

            return (
                speaker_id,
                1.0,
                True,
            )

        # ----------------------------------------------------
        # Find closest global speaker centroid.
        # ----------------------------------------------------

        best_speaker_id = None
        best_similarity = -1.0

        for (
            speaker_id,
            speaker,
        ) in self._speakers.items():

            similarity = self._cosine_similarity(
                embedding,
                speaker.embedding,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker_id = speaker_id

        # ----------------------------------------------------
        # Existing speaker.
        # ----------------------------------------------------

        if (
            best_speaker_id is not None
            and best_similarity
            >= self.similarity_threshold
        ):
            self._update_centroid(
                best_speaker_id,
                embedding,
            )

            return (
                best_speaker_id,
                best_similarity,
                False,
            )

        # ----------------------------------------------------
        # New speaker.
        # ----------------------------------------------------

        speaker_id = self._create_speaker(
            embedding
        )

        return (
            speaker_id,
            best_similarity,
            True,
        )

    # ========================================================
    # ASSIGN MULTIPLE LOCAL SPEAKERS
    # ========================================================

    def assign_speakers(
        self,
        speakers: list[str],
        embeddings: np.ndarray,
    ) -> dict[str, int]:
        """
        Map local diarization speaker labels to global IDs.

        Example:

            speakers = [
                "SPEAKER_00",
                "SPEAKER_01",
                "SPEAKER_02",
            ]

            returns:

            {
                "SPEAKER_00": 0,
                "SPEAKER_01": 1,
                "SPEAKER_02": 2,
            }
        """

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must have shape "
                "(num_speakers, embedding_dimension). "
                f"Got {embeddings.shape}."
            )

        if len(speakers) != len(embeddings):
            raise ValueError(
                "Number of speakers must match "
                "number of embeddings. "
                f"Speakers={len(speakers)}, "
                f"embeddings={len(embeddings)}."
            )

        mapping = {}

        for speaker, embedding in zip(
            speakers,
            embeddings,
        ):
            (
                global_speaker_id,
                _similarity,
                _is_new,
            ) = self.assign_speaker(
                embedding
            )

            mapping[speaker] = (
                global_speaker_id
            )

        return mapping

    # ========================================================
    # CENTROID UPDATE
    # ========================================================

    def _update_centroid(
        self,
        speaker_id: int,
        embedding: np.ndarray,
    ) -> None:
        """
        Update the running centroid for a global speaker.
        """

        speaker = self._speakers[
            speaker_id
        ]

        count = speaker.observation_count

        new_centroid = (
            speaker.embedding * count
            + embedding
        ) / (count + 1)

        speaker.embedding = (
            self._normalize(
                new_centroid
            )
        )

        speaker.observation_count += 1

    # ========================================================
    # CREATE GLOBAL SPEAKER
    # ========================================================

    def _create_speaker(
        self,
        embedding: np.ndarray,
    ) -> int:
        """
        Create a new global speaker.
        """

        speaker_id = (
            self._next_speaker_id
        )

        self._next_speaker_id += 1

        self._speakers[
            speaker_id
        ] = GlobalSpeaker(
            speaker_id=speaker_id,
            embedding=embedding.copy(),
            observation_count=1,
        )

        return speaker_id

    # ========================================================
    # EMBEDDING PREPARATION
    # ========================================================

    @staticmethod
    def _prepare_embedding(
        embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Convert embedding to a normalized 1D float32 vector.
        """

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if embedding.ndim == 1:
            pass

        elif (
            embedding.ndim == 2
            and embedding.shape[0] == 1
        ):
            embedding = embedding[0]

        else:
            raise ValueError(
                "Speaker embedding must have shape "
                "(embedding_dimension,) or "
                "(1, embedding_dimension). "
                f"Got {embedding.shape}."
            )

        return SpeakerManager._normalize(
            embedding
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize(
        embedding: np.ndarray,
    ) -> np.ndarray:
        """
        L2-normalize an embedding.
        """

        norm = np.linalg.norm(
            embedding
        )

        if norm < 1e-12:
            raise ValueError(
                "Cannot normalize a zero "
                "speaker embedding."
            )

        return (
            embedding / norm
        ).astype(np.float32)

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    @staticmethod
    def _cosine_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Cosine similarity between normalized embeddings.
        """

        return float(
            np.dot(a, b)
        )