import numpy as np #type:ignore
import torch #type:ignore
from silero_vad import load_silero_vad, VADIterator #type:ignore

from lma.constants import (
    SAMPLE_RATE,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    VAD_THRESHOLD,
    SpeechEvent,
)
from lma.schemas import PCMFrame

from lma.audio.detector.base import SpeechDetector

torch.set_num_threads(1)


class SileroDetector(SpeechDetector):
    """
    Silero VAD implementation of SpeechDetector.
    """

    def __init__(self):

        model = load_silero_vad()

        self._vad = VADIterator(
            model,
            threshold=VAD_THRESHOLD,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        )

    def process(self,frame: PCMFrame,) -> SpeechEvent | None:

        audio = (np.frombuffer(frame.data,dtype=np.int16).astype(np.float32)/ 32768.0)

        result = self._vad(
            torch.from_numpy(audio),
            return_seconds=False,
        )

        if not result:
            return None

        if "start" in result:
            return SpeechEvent.START

        if "end" in result:
            return SpeechEvent.END

        return None