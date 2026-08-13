from dataclasses import dataclass


@dataclass(slots=True)
class AlignedWord:
    word: str
    start_ms: int
    end_ms: int
    speaker_id: int


class WordAligner:

    MAX_BOUNDARY_DISTANCE_MS = 500

    def align(
        self,
        whisper_result: dict,
        diarization_segments: list[tuple[int, int, int]],
    ) -> list[AlignedWord]:

        words = self._extract_words(whisper_result)

        if not words:
            return []

        # Initial speaker assignment using word midpoint.
        self._assign_words_from_diarization(
            words,
            diarization_segments,
        )

        # Find diarization speaker transitions.
        transitions = self._find_transitions(
            diarization_segments
        )

        # Match each transition to the closest
        # Whisper word boundary.
        refined_transitions = self._analyze_transitions(
            words,
            transitions,
        )

        # Assign the first word after each transition
        # to the new speaker.
        self._apply_transition_ownership(
            refined_transitions
        )

        # Handle the case where the transition occurs
        # exactly at the end of the first word spoken
        # by the new speaker.
        self._correct_first_word_ownership(
            refined_transitions
        )

        return [
            AlignedWord(
                word=word["text"],
                start_ms=word["start_ms"],
                end_ms=word["end_ms"],
                speaker_id=word["speaker"],
            )
            for word in words
            if word["speaker"] is not None
        ]

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

                start = word_data.get("start")
                end = word_data.get("end")
                text = word_data.get(
                    "word",
                    "",
                ).strip()

                if (
                    start is None
                    or end is None
                    or not text
                ):
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
    # INITIAL SPEAKER ASSIGNMENT
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
                    word["speaker"] = speaker_id
                    break

            if word["speaker"] is not None:
                continue

            # If the midpoint falls outside every
            # diarization segment, use the closest segment.
            best_speaker = None
            best_distance = None

            for (
                speaker_start_ms,
                speaker_end_ms,
                speaker_id,
            ) in diarization_segments:

                if midpoint < speaker_start_ms:
                    distance = (
                        speaker_start_ms
                        - midpoint
                    )
                else:
                    distance = (
                        midpoint
                        - speaker_end_ms
                    )

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

            previous_speaker = previous[2]
            current_start_ms = current[0]
            current_speaker = current[2]

            if previous_speaker == current_speaker:
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
                or distance_ms < best["distance_ms"]
            ):
                best = {
                    "boundary_ms": boundary_ms,
                    "left_word": left,
                    "right_word": right,
                    "distance_ms": distance_ms,
                }

        if (
            best is None
            or best["distance_ms"]
            > cls.MAX_BOUNDARY_DISTANCE_MS
        ):
            return None

        return best

    # ========================================================
    # WORDS AROUND TRANSITION
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

            transition_ms = transition["time_ms"]

            context = cls._words_around_transition(
                words,
                transition_ms,
            )

            boundary = cls._find_best_word_boundary(
                context,
                transition_ms,
            )

            if boundary is None:
                continue

            refined.append(
                {
                    "transition_ms": transition_ms,
                    "boundary_ms": boundary["boundary_ms"],
                    "from": transition["from"],
                    "to": transition["to"],
                    "left_word": boundary["left_word"],
                    "right_word": boundary["right_word"],
                    "distance_ms": boundary["distance_ms"],
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

            transition["right_word"]["speaker"] = (
                transition["to"]
            )

    # ========================================================
    # FIRST-WORD OWNERSHIP CORRECTION
    # ========================================================

    @staticmethod
    def _correct_first_word_ownership(
        transitions: list[dict],
    ) -> None:

        for transition in transitions:

            left_word = transition["left_word"]
            right_word = transition["right_word"]
            boundary_ms = transition["boundary_ms"]
            new_speaker = transition["to"]

            if left_word["end_ms"] == boundary_ms:

                left_word["speaker"] = new_speaker
                right_word["speaker"] = new_speaker