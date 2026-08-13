from abc import ABC, abstractmethod

from lma.core.constants import SpeechEvent
from lma.core.schemas import PCMFrame

class SpeechDetector(ABC):
    """
    Abstract speech detector.

    Implementations receive one fixed-size PCM frame and determine whether
    speech has just started, just ended, or nothing changed.
    """

    @abstractmethod
    def process(self, frame: PCMFrame) -> SpeechEvent | None:
        """
        Process one PCM frame.

        Returns:
            SpeechEvent.START
            SpeechEvent.END
            None
        """
        raise NotImplementedError