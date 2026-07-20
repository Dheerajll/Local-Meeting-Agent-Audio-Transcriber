from lma.audio.capture import FFmpegCapture


capture = FFmpegCapture(
    device=":0"
)


capture.start()


count = 0


try:

    for frame in capture.frames():

        count += 1

        print(
            count,
            len(frame)
        )


        if count == 100:
            break


finally:

    capture.stop()