from __future__ import annotations

from math import exp
from pathlib import Path

import mlx_whisper

from lma.core.schemas import AudioChunk, TranscriptChunk
from lma.transcription.diarizer import Diarizer
from lma.transcription.speaker_manager import SpeakerManager
from lma.transcription.word_aligner import WordAligner


class WhisperTranscriber:
    """
    Transcribes an AudioChunk and produces speaker-labelled text.

    Pipeline:

        Whisper
            ↓
        Diarizer
            ↓
        SpeakerManager
            ↓
        WordAligner
            ↓
        speaker-labelled transcript

    Responsibilities:
        - Run Whisper transcription
        - Run speaker diarization
        - Assign persistent/global speaker identities
        - Align Whisper words with speakers
        - Format speaker-labelled transcript
        - Convert the result into TranscriptChunk

    Does not:
        - write files
        - publish results
        - manage queues
        - manage sessions
    """

    def __init__(
        self,
        model_path: Path,
        diarizer: Diarizer,
        speaker_manager: SpeakerManager,
        word_aligner: WordAligner,
        language: str | None = None,
    ) -> None:

        self.model_path = model_path
        self.diarizer = diarizer
        self.speaker_manager = speaker_manager
        self.word_aligner = word_aligner
        self.language = language

    # ========================================================
    # TRANSCRIPTION
    # ========================================================

    def transcribe(
        self,
        audio_path: Path,
        chunk: AudioChunk,
    ) -> TranscriptChunk:

        # ----------------------------------------------------
        # STEP 1
        #
        # Run Whisper.
        # ----------------------------------------------------

        if self.language == "ne":
            temperature=0.1
            compression_ratio_threshold=1.8
            logprob_threshold=-0.8
        else:
            temperature=0.0
            compression_ratio_threshold=0
            logprob_threshold=0
        
        whisper_result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=str(self.model_path),
            task="transcribe", 
            language=self.language,
            
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            condition_on_previous_text=False,
            word_timestamps=True,
        )

        # ----------------------------------------------------
        # STEP 2
        #
        # Run diarization.
        # ----------------------------------------------------

        diarization = self.diarizer.diarize(
            audio_path
        )

        # ----------------------------------------------------
        # STEP 3
        #
        # Assign local speakers to persistent
        # global speaker identities.
        # ----------------------------------------------------

        local_to_global = {}

        for index, local_speaker in enumerate(
            diarization.speakers
        ):

            (
                global_speaker_id,
                _similarity,
                _is_new,
            ) = self.speaker_manager.assign_speaker(
                diarization.embeddings[index]
            )

            local_to_global[
                local_speaker
            ] = global_speaker_id

        # ----------------------------------------------------
        # STEP 4
        #
        # Convert diarization segments into the
        # representation expected by WordAligner.
        # ----------------------------------------------------

        diarization_segments = [
            (
                segment.start_ms,
                segment.end_ms,
                local_to_global[
                    segment.speaker_id
                ],
            )
            for segment in diarization.segments
        ]

        # ----------------------------------------------------
        # STEP 5
        #
        # Align Whisper words with global speakers.
        # ----------------------------------------------------

        aligned_words = self.word_aligner.align(
            whisper_result,
            diarization_segments,
        )

        # ----------------------------------------------------
        # STEP 6
        #
        # Convert aligned words into:
        #
        # [speaker 0] hello this is...
        # [speaker 1] yeah I agree...
        #
        # Consecutive words belonging to the same
        # speaker are kept in the same dialogue block.
        # ----------------------------------------------------

        text = self._format_speaker_text(
            aligned_words
        )

        # ----------------------------------------------------
        # STEP 7
        #
        # Calculate transcription confidence.
        # ----------------------------------------------------

        confidence = self._calculate_confidence(
            whisper_result.get(
                "segments",
                [],
            )
        )

        language = whisper_result.get(
            "language",
            self.language,
        )

        return TranscriptChunk(
            chunk_id=chunk.chunk_id,
            raw_text=text,
            confidence=confidence,
            language=language,
            start_ms=chunk.start_ms,
            end_ms=chunk.end_ms,
            reason=chunk.reason,
            forced=chunk.forced,
        )

    # ========================================================
    # SPEAKER TEXT FORMATTING
    # ========================================================

    @staticmethod
    def _format_speaker_text(
        aligned_words,
    ) -> str:

        if not aligned_words:
            return ""

        lines = []

        current_speaker = None
        current_words = []

        for word in aligned_words:

            if (
                current_speaker is None
            ):
                current_speaker = (
                    word.speaker_id
                )

            # ------------------------------------------------
            # Speaker changed.
            # ------------------------------------------------

            if (
                word.speaker_id
                != current_speaker
            ):

                if current_words:

                    lines.append(
                        f"[speaker {current_speaker}] "
                        + " ".join(
                            current_words
                        )
                    )

                current_speaker = (
                    word.speaker_id
                )

                current_words = []

            current_words.append(
                word.word
            )

        # ----------------------------------------------------
        # Flush final speaker.
        # ----------------------------------------------------

        if current_words:

            lines.append(
                f"[speaker {current_speaker}] "
                + " ".join(current_words)
            )

        return " ".join(lines)

    # ========================================================
    # CONFIDENCE
    # ========================================================

    @staticmethod
    def _calculate_confidence(
        segments: list[dict],
    ) -> float:

        if not segments:
            return 0.0

        total_weight = 0.0
        weighted_confidence = 0.0

        for segment in segments:

            start = float(
                segment["start"]
            )

            end = float(
                segment["end"]
            )

            duration = max(
                end - start,
                0.0,
            )

            if duration <= 0:
                continue

            avg_logprob = float(
                segment["avg_logprob"]
            )

            no_speech_prob = float(
                segment["no_speech_prob"]
            )

            segment_confidence = (
                exp(avg_logprob)
                * (1.0 - no_speech_prob)
            )

            weighted_confidence += (
                segment_confidence
                * duration
            )

            total_weight += duration

        if total_weight == 0:
            return 0.0

        confidence = (
            weighted_confidence
            / total_weight
        )

        return max(
            0.0,
            min(1.0, confidence),
        )