from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyannote.audio import Pipeline


@dataclass(slots=True)
class DiarizationSegment:
    start_ms: int
    end_ms: int
    speaker_id: str


@dataclass(slots=True)
class DiarizationResult:
    segments: list[DiarizationSegment]
    speakers: list[str]
    embeddings: np.ndarray


class Diarizer:
    """
    Runs pyannote speaker diarization.

    Responsibilities:
        - Run speaker diarization
        - Extract local speaker segments
        - Extract local speaker embeddings

    Does not:
        - assign persistent/global speaker identities
        - align words
        - transcribe audio
        - create TranscriptChunk
        - manage sessions
        - write files
        - publish results
    """

    def __init__(
        self,
        model: str = (
            "pyannote/speaker-diarization-community-1"
        ),
    ) -> None:

        self.pipeline = Pipeline.from_pretrained(
            model
        )

    def diarize(
        self,
        audio_path: Path,
    ) -> DiarizationResult:

        result = self.pipeline(
            str(audio_path)
        )

        diarization = (
            result.speaker_diarization
        )

        speakers = list(
            diarization.labels()
        )

        segments = []

        for (
            segment,
            _,
            speaker,
        ) in diarization.itertracks(
            yield_label=True,
        ):

            segments.append(
                DiarizationSegment(
                    start_ms=round(
                        segment.start * 1000
                    ),
                    end_ms=round(
                        segment.end * 1000
                    ),
                    speaker_id=speaker,
                )
            )

        embeddings = np.asarray(
            result.speaker_embeddings,
            dtype=np.float32,
        )

        return DiarizationResult(
            segments=segments,
            speakers=speakers,
            embeddings=embeddings,
        )