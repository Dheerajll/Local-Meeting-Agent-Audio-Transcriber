import subprocess

from lma.constants import (
    CHANNELS,
    SAMPLE_RATE,
    FRAME_BYTES,
    READ_SIZE,
)

from ..exceptions import FFmpegError


class FFmpegCapture:


    def __init__(
        self,
        device=":0"
    ):

        self.device = device

        self.process = None

        self.leftover = bytearray()



    def start(self):

        command = [

            "ffmpeg",

            "-f",
            "avfoundation",

            "-i",
            self.device,

            "-ac",
            str(CHANNELS),

            "-ar",
            str(SAMPLE_RATE),

            "-f",
            "s16le",

            "pipe:1"
        ]


        try:

            self.process = subprocess.Popen(

                command,

                stdout=subprocess.PIPE,

                stderr=subprocess.DEVNULL,

                bufsize=4096
            )


        except Exception as exc:

            raise FFmpegError(
                f"Could not start FFmpeg: {exc}"
            )
        
    def frames(self):
                
        if self.process is None:

            raise FFmpegError(
                "Capture not started"
            )


        while True:

            data = (
                self.process.stdout.read(READ_SIZE)
            )


            if not data:

                break


            self.leftover.extend(data)


            while len(self.leftover) >= FRAME_BYTES:


                frame = bytes(
                    self.leftover[:FRAME_BYTES]
                )


                del self.leftover[:FRAME_BYTES]


                yield frame
    
    def stop(self):

        if self.process:

            self.process.terminate()

            self.process.wait(
                timeout=2
            )

            self.process = None
        