from abc import ABC, abstractmethod

from lma.constants import SpeechEvent


class SpeechDetector(ABC):
    """
    Abstract speech detector.

    Implementations receive one fixed-size PCM frame and determine whether
    speech has just started, just ended, or nothing changed.
    """

    @abstractmethod
    def process(self, pcm: bytes) -> SpeechEvent | None:
        """
        Process one PCM frame.

        Returns:
            SpeechEvent.START
            SpeechEvent.END
            None
        """
        raise NotImplementedError