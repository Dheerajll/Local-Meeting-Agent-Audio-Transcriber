from pathlib import Path
import wave

from lma.core.schemas import AudioChunk
from datetime import datetime

class WavWriter:
    """
    Writes AudioChunk PCM bytes to a temporary WAV file.

    This class is intentionally unaware of Whisper or the
    transcription pipeline.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, chunk: AudioChunk) -> Path:
        timestamp = datetime.now().strftime("%H%M%S")
        filename = self.output_dir / f"chunk_{timestamp}_{chunk.chunk_id:02}.wav"

        with wave.open(str(filename), "wb") as wav:
            wav.setnchannels(chunk.channels)
            wav.setsampwidth(2)          # 16-bit PCM
            wav.setframerate(chunk.sample_rate)
            wav.writeframes(chunk.pcm_bytes)

        return filename
    
    def delete(self, path: Path) -> None:
        if path.exists():
            path.unlink()