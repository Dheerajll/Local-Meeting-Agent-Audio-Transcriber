from pathlib import Path

import mlx_whisper

from lma.transcription.cache import get_model_snapshot_path


class WhisperTranscriber:

    def __init__(self,model_name: str,language: str | None = None):
        self.model_name = model_name
        self.language = language

        self.model_path = get_model_snapshot_path(model_name)

        print(f"Using Whisper snapshot: {self.model_path}")

    def transcribe(self,audio_path: Path,) -> str:

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=str(self.model_path),
            language=self.language,
            temperature=0.0,
        )

        return result["text"].strip()