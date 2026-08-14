import subprocess
import threading

# Adjust these imports if you kept them at the root lma/ level instead of lma/core/
from lma.core.constants import CHANNELS, SAMPLE_RATE, FRAME_BYTES, READ_SIZE
from lma.core.exceptions import FFmpegError

class FFmpegCapture:
    def __init__(self, device=":0"):
        self.device = device
        self.process = None
        self.leftover = bytearray()

    def start(self):
        # AVFoundation expects "[video]:[audio]". 
        # If the device string doesn't start with ":", prepend it 
        # to tell FFmpeg we only want audio.
        if self.device.startswith(":"):
            input_arg = self.device
        else:
            input_arg = f":{self.device}"

        command = [
            "ffmpeg",
            "-f", "avfoundation",
            "-i", input_arg,
            "-ac", str(CHANNELS),
            "-ar", str(SAMPLE_RATE),
            "-f", "s16le",
            "pipe:1"
        ]

        try:
            # Capture stderr so we can see FFmpeg errors instead of hiding them
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=4096
            )
        except Exception as exc:
            raise FFmpegError(f"Could not start FFmpeg: {exc}")

        # Read stderr in a background thread so it doesn't block the main pipe
        def _print_stderr():
            for line in self.process.stderr:
                decoded = line.decode('utf-8', errors='replace').strip()
                # Ignore the startup banner and stream info.
                # Only print if it's an actual error or failure.
                if "error" in decoded.lower() or "fail" in decoded.lower():
                    print(f"❌ [FFmpeg] {decoded}")
                    
        self._stderr_thread = threading.Thread(target=_print_stderr, daemon=True)
        self._stderr_thread.start()

    def frames(self):
        if self.process is None:
            raise FFmpegError("Capture not started")
            
        while True:
            data = self.process.stdout.read(READ_SIZE)
            if not data:
                ret = self.process.poll()
                if ret is not None:
                    print(f"\n⚠️ FFmpeg process exited unexpectedly with code {ret}.")
                else:
                    print("\n⚠️ FFmpeg stream ended.")
                break

            self.leftover.extend(data)
            while len(self.leftover) >= FRAME_BYTES:
                frame = bytes(self.leftover[:FRAME_BYTES])
                del self.leftover[:FRAME_BYTES]
                yield frame

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None