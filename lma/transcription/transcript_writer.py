import json
from pathlib import Path
from dataclasses import asdict

from lma.core.schemas import TranscriptChunk


class TranscriptWriter:
    """
    Persists TranscriptChunk objects as JSON files.

    This class knows nothing about:
        - Whisper
        - audio
        - queues
        - backend delivery
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def write(self, transcript: TranscriptChunk) -> Path:
        path = (
            self.output_dir
            / f"chunk_{transcript.chunk_id:06d}.json"
        )

        data = asdict(transcript)

        # ChunkReason is an Enum, so convert it to its
        # serialized value before writing JSON.
        data["reason"] = transcript.reason.value

        with path.open("w",encoding="utf-8") as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return path