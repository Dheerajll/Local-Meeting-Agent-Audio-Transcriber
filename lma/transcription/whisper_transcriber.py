from pathlib import Path
from math import exp

import mlx_whisper

from lma.constants import ChunkReason
from lma.schemas import AudioChunk, TranscriptChunk


class WhisperTranscriber:
    """
    Transcribes an AudioChunk using a locally cached MLX Whisper model.

    Responsibilities:
        - Run Whisper transcription
        - Convert Whisper output into TranscriptChunk

    Does not:
        - write files
        - publish results
        - manage queues
        - manage sessions
    """

    def __init__(self,model_path: Path,language: str | None = None) -> None:

        self.model_path = model_path
        self.language = language

    def transcribe(self,audio_path: Path,chunk: AudioChunk) -> TranscriptChunk:

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=str(self.model_path),
            language=self.language,
            temperature=0.0,
            condition_on_previous_text=True,
        )

        text = result["text"].strip()
        language = result["language"]

        confidence = self._calculate_confidence(
            result["segments"]
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

    @staticmethod
    def _calculate_confidence(segments: list[dict]) -> float:

        if not segments:
            return 0.0

        total_weight = 0.0
        weighted_confidence = 0.0

        for segment in segments:

            start = float(segment["start"])
            end = float(segment["end"])

            duration = max(end - start,0.0,)

            if duration <= 0:
                continue

            avg_logprob = float(segment["avg_logprob"])

            no_speech_prob = float(segment["no_speech_prob"])

            segment_confidence = (exp(avg_logprob)* (1.0 - no_speech_prob))

            weighted_confidence += (segment_confidence * duration)

            total_weight += duration

        if total_weight == 0:
            return 0.0

        confidence = (weighted_confidence/ total_weight)

        return max(
            0.0,
            min(1.0, confidence),
        )