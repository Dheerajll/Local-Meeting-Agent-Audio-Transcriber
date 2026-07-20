"""Rolling PCM byte buffer used by AudioRecorder to build up a chunk without
repeated bytearray slicing in the hot loop.
"""


class RollingAudioBuffer:

    def __init__(self, initial: bytes = b""):
        self._data = bytearray(initial)

    def append(self, pcm: bytes) -> None:
        self._data.extend(pcm)

    def tail(self, n_bytes: int) -> bytes:
        if n_bytes <= 0:
            return b""
        return bytes(self._data[-n_bytes:])

    def read(self) -> bytes:
        return bytes(self._data)

    def clear(self) -> None:
        self._data = bytearray()

    def __len__(self) -> int:
        return len(self._data)