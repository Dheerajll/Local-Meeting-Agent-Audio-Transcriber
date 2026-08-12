from dataclasses import dataclass


@dataclass(slots=True)
class SpeakerSegment:
    start_ms: int
    end_ms: int
    speaker_id: int


@dataclass(slots=True)
class AlignedSegment:
    start_ms: int
    end_ms: int
    text: str
    speaker_id: int


class SpeakerAligner:
    """
    Aligns Whisper transcript segments with
    meeting-level speaker segments.

    Alignment is based on maximum temporal overlap.
    """

    def align(self,transcript_segments,speaker_segments: list[SpeakerSegment],) -> list[AlignedSegment]:

        aligned = []

        for transcript in transcript_segments:

            transcript_start = transcript["start"]
            transcript_end = transcript["end"]

            best_speaker = None
            best_overlap = 0.0

            for speaker in speaker_segments:

                overlap_start = max(
                    transcript_start,
                    speaker.start_ms / 1000.0,
                )

                overlap_end = min(
                    transcript_end,
                    speaker.end_ms / 1000.0,
                )

                overlap = max(0.0,overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker.speaker_id

            if best_speaker is None:
                continue

            aligned.append(
                AlignedSegment(
                    start_ms=int(
                        transcript_start * 1000
                    ),
                    end_ms=int(
                        transcript_end * 1000
                    ),
                    text=transcript["text"].strip(),
                    speaker_id=best_speaker,
                )
            )

        return aligned