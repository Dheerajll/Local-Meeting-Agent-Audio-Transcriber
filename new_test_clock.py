from lma.audio_p.capture import FFmpegCapture
from lma.audio_p.clock import AudioClock
from lma.audio_p.source import AudioSource


source = AudioSource(
    capture=FFmpegCapture(":0"),
    clock=AudioClock()
)


source.start()


count = 0


try:

    for frame in source.frames():

        print(
            frame.timestamp_ms,
            len(frame.data)
        )

        count += 1

        if count == 10:
            break


finally:

    source.stop()