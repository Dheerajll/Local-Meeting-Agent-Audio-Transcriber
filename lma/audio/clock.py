from lma.constants import FRAME_DURATION_MS

class AudioClock:
    """
    Tracks the current audio position.

    Every processed frame advances the clock by one frame duration.
    """

    def __init__(self) -> None:
        self._position_ms = 0

    @property
    def position_ms(self) -> int:
        return self._position_ms

    def tick(self) -> int:
        """
        Advance the clock by one frame.

        Returns:
            The timestamp of the frame being processed.
        """

        timestamp = self._position_ms
        self._position_ms += FRAME_DURATION_MS
        return timestamp

    def reset(self) -> None:
        self._position_ms = 0