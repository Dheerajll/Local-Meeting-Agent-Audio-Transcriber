from lma.audio.capture import FFmpegCapture
from lma.audio.clock import AudioClock
from lma.core.schemas import PCMFrame


class AudioSource:
    """
    Combines raw capture with timestamps.
    """

    def __init__(self,capture: FFmpegCapture,clock: AudioClock):

        self.capture = capture
        self.clock = clock


    def start(self):

        self.capture.start()


    def stop(self):

        self.capture.stop()


    def frames(self):

        for pcm in self.capture.frames():

            yield PCMFrame(
                data=pcm,
                timestamp_ms=self.clock.tick()
            )