from lma.constants import FRAME_DURATION_MS

class AudioClock:

    def __init__(self):

        self.position_ms = 0


    def advance(self):

        current = self.position_ms

        self.position_ms += FRAME_DURATION_MS

        return current