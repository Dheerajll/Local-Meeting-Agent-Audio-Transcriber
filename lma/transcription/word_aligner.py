from dataclasses import dataclass


@dataclass(slots=True)
class AlignedWord:
    word: str
    start_ms: int
    end_ms: int
    speaker_id: int


class WordAligner:

    # Maximum distance between a diarization transition
    # and a Whisper word boundary that we will accept.
    MAX_BOUNDARY_DISTANCE_MS = 500

    def align(
        self,
        whisper_result: dict,
        diarization_segments: list[tuple[int, int, int]],
    ) -> list[AlignedWord]:

        words = self._extract_words(
            whisper_result
        )

        if not words:
            return []

        # ----------------------------------------------------
        # STEP 1
        #
        # Baseline assignment using word midpoint.
        # ----------------------------------------------------

        self._assign_words_from_diarization(
            words,
            diarization_segments,
        )

        # ----------------------------------------------------
        # STEP 2
        #
        # Find speaker transitions.
        # ----------------------------------------------------

        transitions = self._find_transitions(
            diarization_segments
        )

        # ----------------------------------------------------
        # STEP 3
        #
        # Refine ownership around every transition.
        # ----------------------------------------------------

        refined_transitions = (
            self._analyze_transitions(
                words,
                transitions,
            )
        )

        # ----------------------------------------------------
        # STEP 4
        #
        # Apply transition ownership.
        # ----------------------------------------------------

        self._apply_transition_ownership(
            refined_transitions
        )

        # ----------------------------------------------------
        # STEP 5
        #
        # Apply the first-word correction.
        #
        # This handles the case where:
        #
        # speaker 1:
        #     ...
        #
        # speaker 2:
        #     I thought ...
        #
        # but diarization places the boundary at the
        # end of "I".
        # ----------------------------------------------------

        self._correct_first_word_ownership(
            refined_transitions
        )

        # ----------------------------------------------------
        # STEP 6
        #
        # Convert internal words into AlignedWord objects.
        # ----------------------------------------------------

        aligned = []

        for word in words:

            if word["speaker"] is None:
                continue

            aligned.append(
                AlignedWord(
                    word=word["text"],
                    start_ms=word["start_ms"],
                    end_ms=word["end_ms"],
                    speaker_id=word["speaker"],
                )
            )

        return aligned

    # ========================================================
    # WHISPER WORD EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_words(
        whisper_result: dict,
    ) -> list[dict]:

        words = []

        for segment in whisper_result.get(
            "segments",
            [],
        ):

            for word_data in segment.get(
                "words",
                [],
            ):

                start = word_data.get(
                    "start"
                )

                end = word_data.get(
                    "end"
                )

                text = word_data.get(
                    "word",
                    "",
                ).strip()

                if start is None:
                    continue

                if end is None:
                    continue

                if not text:
                    continue

                words.append(
                    {
                        "start_ms": round(
                            float(start) * 1000
                        ),
                        "end_ms": round(
                            float(end) * 1000
                        ),
                        "text": text,
                        "speaker": None,
                    }
                )

        return words

    # ========================================================
    # INITIAL MIDPOINT ASSIGNMENT
    # ========================================================

    @staticmethod
    def _assign_words_from_diarization(
        words: list[dict],
        diarization_segments: list[
            tuple[int, int, int]
        ],
    ) -> None:

        for word in words:

            midpoint = (
                word["start_ms"]
                + word["end_ms"]
            ) / 2

            word["speaker"] = None

            for (
                speaker_start_ms,
                speaker_end_ms,
                speaker_id,
            ) in diarization_segments:

                if (
                    speaker_start_ms
                    <= midpoint
                    < speaker_end_ms
                ):

                    word["speaker"] = (
                        speaker_id
                    )

                    break

            # ----------------------------------------------------
            # FALLBACK
            #
            # The word's midpoint didn't land inside any
            # diarization segment (usually because it's a
            # pause/gap between turns, not real silence).
            # Instead of dropping the word, assign it to
            # whichever segment is closest in time.
            # ----------------------------------------------------

            if word["speaker"] is not None:
                continue

            best_speaker = None
            best_distance = None

            for (
                speaker_start_ms,
                speaker_end_ms,
                speaker_id,
            ) in diarization_segments:

                # distance from midpoint to this segment
                # (0 if midpoint is inside it, which
                # shouldn't happen here since we already
                # checked that above)
                if midpoint < speaker_start_ms:
                    distance = speaker_start_ms - midpoint
                elif midpoint >= speaker_end_ms:
                    distance = midpoint - speaker_end_ms
                else:
                    distance = 0

                if (
                    best_distance is None
                    or distance < best_distance
                ):
                    best_distance = distance
                    best_speaker = speaker_id

            word["speaker"] = best_speaker
    # ========================================================
    # FIND SPEAKER TRANSITIONS
    # ========================================================

    @staticmethod
    def _find_transitions(
        diarization_segments: list[
            tuple[int, int, int]
        ],
    ) -> list[dict]:

        if not diarization_segments:
            return []

        segments = sorted(
            diarization_segments,
            key=lambda segment: segment[0],
        )

        transitions = []

        for previous, current in zip(
            segments,
            segments[1:],
        ):

            previous_start_ms = previous[0]
            previous_end_ms = previous[1]
            previous_speaker = previous[2]

            current_start_ms = current[0]
            current_end_ms = current[1]
            current_speaker = current[2]

            # Silence/gap between same-speaker segments
            # is not a speaker transition.
            if (
                previous_speaker
                == current_speaker
            ):
                continue

            transitions.append(
                {
                    "time_ms": current_start_ms,
                    "from": previous_speaker,
                    "to": current_speaker,
                }
            )

        return transitions

    # ========================================================
    # FIND CLOSEST WHISPER WORD BOUNDARY
    # ========================================================

    @classmethod
    def _find_best_word_boundary(
        cls,
        words: list[dict],
        transition_ms: int,
    ) -> dict | None:

        if len(words) < 2:
            return None

        best = None

        for left, right in zip(
            words,
            words[1:],
        ):

            boundary_ms = left["end_ms"]

            distance_ms = abs(
                boundary_ms
                - transition_ms
            )

            if (
                best is None
                or distance_ms
                < best["distance_ms"]
            ):

                best = {
                    "boundary_ms": boundary_ms,
                    "left_word": left,
                    "right_word": right,
                    "distance_ms": distance_ms,
                }

        if best is None:
            return None

        if (
            best["distance_ms"]
            > cls.MAX_BOUNDARY_DISTANCE_MS
        ):
            return None

        return best

    # ========================================================
    # FIND WORDS AROUND TRANSITION
    # ========================================================

    @staticmethod
    def _words_around_transition(
        words: list[dict],
        transition_ms: int,
    ) -> list[dict]:

        if not words:
            return []

        closest_index = min(
            range(len(words)),
            key=lambda index: min(
                abs(
                    words[index]["start_ms"]
                    - transition_ms
                ),
                abs(
                    words[index]["end_ms"]
                    - transition_ms
                ),
            ),
        )

        # Same idea as the test script, but we don't
        # actually need to restrict this too aggressively.
        start = max(
            0,
            closest_index - 6,
        )

        end = min(
            len(words),
            closest_index + 7,
        )

        return words[start:end]

    # ========================================================
    # ANALYZE TRANSITIONS
    # ========================================================

    @classmethod
    def _analyze_transitions(
        cls,
        words: list[dict],
        transitions: list[dict],
    ) -> list[dict]:

        refined = []

        for transition in transitions:

            transition_ms = (
                transition["time_ms"]
            )

            context = (
                cls._words_around_transition(
                    words,
                    transition_ms,
                )
            )

            boundary = (
                cls._find_best_word_boundary(
                    context,
                    transition_ms,
                )
            )

            if boundary is None:
                continue

            refined.append(
                {
                    "transition_ms": transition_ms,
                    "boundary_ms": boundary[
                        "boundary_ms"
                    ],
                    "from": transition["from"],
                    "to": transition["to"],
                    "left_word": boundary[
                        "left_word"
                    ],
                    "right_word": boundary[
                        "right_word"
                    ],
                    "distance_ms": boundary[
                        "distance_ms"
                    ],
                }
            )

        return refined

    # ========================================================
    # APPLY TRANSITION OWNERSHIP
    # ========================================================

    @staticmethod
    def _apply_transition_ownership(
        transitions: list[dict],
    ) -> None:

        for transition in transitions:

            new_speaker = transition[
                "to"
            ]

            right_word = transition[
                "right_word"
            ]

            # The first Whisper word after the refined
            # boundary belongs to the new speaker.
            right_word["speaker"] = (
                new_speaker
            )

    # ========================================================
    # FIRST-WORD OWNERSHIP CORRECTION
    # ========================================================

    @staticmethod
    def _correct_first_word_ownership(
        transitions: list[dict],
    ) -> None:

        for transition in transitions:

            boundary_ms = transition[
                "boundary_ms"
            ]

            new_speaker = transition[
                "to"
            ]

            left_word = transition[
                "left_word"
            ]

            right_word = transition[
                "right_word"
            ]

            # If the diarization boundary falls exactly
            # at the end of the left Whisper word, then
            # the experimental rule says that this left
            # word may actually be the first word of the
            # new speaker.
            #
            # Example:
            #
            # speaker 1:
            #     ... yet
            #
            # speaker 2:
            #     I thought ...
            #
            # Whisper:
            #
            #     I       10540 -> 11540
            #     thought 11540 -> 11760
            #
            # Diarization:
            #
            #     speaker transition = 11540
            #
            # Therefore:
            #
            #     I       -> speaker 2
            #     thought -> speaker 2

            if (
                left_word["end_ms"]
                == boundary_ms
            ):

                left_word["speaker"] = (
                    new_speaker
                )

                right_word["speaker"] = (
                    new_speaker
                )

    # ========================================================
    # LEGACY BEST-OVERLAP HELPER
    # ========================================================

    @staticmethod
    def _find_best_speaker(
        word_start_ms: int,
        word_end_ms: int,
        diarization_segments: list[
            tuple[int, int, int]
        ],
    ) -> int | None:

        best_speaker = None
        best_overlap = 0

        for (
            speaker_start_ms,
            speaker_end_ms,
            speaker_id,
        ) in diarization_segments:

            overlap_start = max(
                word_start_ms,
                speaker_start_ms,
            )

            overlap_end = min(
                word_end_ms,
                speaker_end_ms,
            )

            overlap = max(
                0,
                overlap_end - overlap_start,
            )

            if overlap > best_overlap:

                best_overlap = overlap
                best_speaker = speaker_id

        return best_speaker