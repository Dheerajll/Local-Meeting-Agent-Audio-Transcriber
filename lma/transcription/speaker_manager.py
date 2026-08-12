from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class GlobalSpeaker:
    """
    Represents one persistent/global speaker identity.

    embedding:
        Running centroid embedding for this speaker.

    observation_count:
        Number of local speaker embeddings that have been
        assigned to this global speaker.
    """

    speaker_id: int
    embedding: np.ndarray
    observation_count: int = 1


class SpeakerManager:
    """
    Maintains global speaker identities across audio chunks.

    Each local speaker embedding is compared against the
    centroid embedding of every known global speaker.

    If the best similarity is >= threshold:
        -> assign to that global speaker
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

        self._speakers: Dict[
            int, GlobalSpeaker
        ] = {}

        self._next_speaker_id = 0

    # ========================================================
    # PUBLIC API
    # ========================================================

    def assign_speaker(
        self,
        embedding: np.ndarray,
    ) -> Tuple[int, float, bool]:
        """
        Assign an embedding to a global speaker.

        Returns:

            global_speaker_id
            similarity
            is_new

        Example:

            speaker_id, similarity, is_new = (
                manager.assign_speaker(embedding)
            )
        """

        embedding = self._prepare_embedding(
            embedding
        )

        # ----------------------------------------------------
        # No global speakers yet.
        # ----------------------------------------------------

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
        # Find the closest global speaker.
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

            if (
                similarity
                > best_similarity
            ):
                best_similarity = similarity
                best_speaker_id = speaker_id

        # ----------------------------------------------------
        # Match existing speaker.
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
        # No sufficiently similar speaker.
        # ----------------------------------------------------

        speaker_id = self._create_speaker(
            embedding
        )

        return (
            speaker_id,
            best_similarity,
            True,
        )

    def assign_speakers(
        self,
        speakers: list[str],
        embeddings: np.ndarray,
    ) -> dict[str, int]:
        """
        Assign multiple local speakers to global speakers.

        Parameters
        ----------
        speakers:
            Local speaker labels, e.g.

                [
                    "SPEAKER_00",
                    "SPEAKER_01",
                    "SPEAKER_02",
                ]

        embeddings:
            Array shaped:

                (num_speakers, embedding_dimension)

        Returns
        -------
        dict

            {
                "SPEAKER_00": 0,
                "SPEAKER_01": 1,
                "SPEAKER_02": 2,
            }
        """

        embeddings = np.asarray(
            embeddings
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array "
                "with shape "
                "(num_speakers, embedding_dimension). "
                f"Got shape: {embeddings.shape}"
            )

        if len(speakers) != len(embeddings):
            raise ValueError(
                "Number of speakers does not match "
                "number of embeddings. "
                f"Speakers: {len(speakers)}, "
                f"Embeddings: {len(embeddings)}"
            )

        mapping = {}

        for speaker, embedding in zip(
            speakers,
            embeddings,
        ):

            (
                global_speaker_id,
                similarity,
                is_new,
            ) = self.assign_speaker(
                embedding
            )

            mapping[speaker] = (
                global_speaker_id
            )

            print(
                f"{speaker:<15}"
                f"→ GLOBAL_SPEAKER_"
                f"{global_speaker_id:<3}"
                f" similarity={similarity:.4f}"
                f" "
                f"[{'NEW' if is_new else 'MATCH'}]"
            )

        return mapping

    # ========================================================
    # CENTROID MANAGEMENT
    # ========================================================

    def _update_centroid(
        self,
        speaker_id: int,
        embedding: np.ndarray,
    ) -> None:
        """
        Update the running centroid of a global speaker.

        Uses:

            new_centroid =
                (
                    old_centroid * count
                    + new_embedding
                )
                / (count + 1)

        The result is normalized again.
        """

        speaker = self._speakers[
            speaker_id
        ]

        count = speaker.observation_count

        old_centroid = speaker.embedding

        new_centroid = (
            old_centroid * count
            + embedding
        ) / (count + 1)

        new_centroid = (
            self._normalize(
                new_centroid
            )
        )

        speaker.embedding = (
            new_centroid
        )

        speaker.observation_count += 1

    # ========================================================
    # CREATE SPEAKER
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
        Convert embedding to float32 1D normalized vector.
        """

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        # Accept (256,)
        if embedding.ndim == 1:
            pass

        # Also accept (1, 256)
        elif (
            embedding.ndim == 2
            and embedding.shape[0] == 1
        ):
            embedding = embedding[0]

        else:
            raise ValueError(
                "Speaker embedding must have "
                "shape (embedding_dimension,) "
                "or (1, embedding_dimension). "
                f"Got shape: {embedding.shape}"
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
        Cosine similarity between two
        normalized embeddings.
        """

        return float(
            np.dot(a, b)
        )

    # ========================================================
    # INSPECTION
    # ========================================================

    def get_speakers(
        self,
    ) -> dict[int, GlobalSpeaker]:
        """
        Return all known global speakers.
        """

        return self._speakers.copy()

    def get_embedding(
        self,
        speaker_id: int,
    ) -> np.ndarray:
        """
        Return the centroid embedding
        for a global speaker.
        """

        if speaker_id not in self._speakers:
            raise KeyError(
                f"Unknown global speaker: "
                f"{speaker_id}"
            )

        return self._speakers[
            speaker_id
        ].embedding.copy()

    def get_speaker_count(
        self,
    ) -> int:
        """
        Number of known global speakers.
        """

        return len(self._speakers)

    def print_state(self) -> None:
        """
        Print the current global speaker state.
        """

        print()
        print(
            "=" * 70
        )
        print(
            "GLOBAL SPEAKER STATE"
        )
        print(
            "=" * 70
        )

        if not self._speakers:
            print(
                "No global speakers."
            )
            return

        for (
            speaker_id,
            speaker,
        ) in self._speakers.items():

            print(
                f"GLOBAL_SPEAKER_{speaker_id}"
            )

            print(
                f"  observations: "
                f"{speaker.observation_count}"
            )

            print(
                f"  embedding shape: "
                f"{speaker.embedding.shape}"
            )

            print(
                f"  embedding norm: "
                f"{np.linalg.norm(speaker.embedding):.6f}"
            )

            print()