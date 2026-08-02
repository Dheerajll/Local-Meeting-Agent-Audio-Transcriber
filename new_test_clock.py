from lma.audio_p.capture import FFmpegCapture
from lma.audio_p.clock import AudioClock
from lma.audio_p.source import AudioSource
from lma.audio_p.detector import SileroDetector


source = AudioSource(
    FFmpegCapture(":0"),
    AudioClock()
)

detector = SileroDetector()

source.start()

try:

    for frame in source.frames():

        event = detector.process(frame)

        if event:
            print(
                frame.timestamp_ms,
                event
            )

finally:
    source.stop()