"""
Generic background worker with a clean shutdown path.

Both the recorder's thread and this transcription pipeline block on I/O
(ffmpeg reads, queue.get()) and need a way to stop without just being
killed as a daemon mid-chunk. This wraps that pattern once instead of
reimplementing it per component — subclasses implement run() and check
self.stopped in their loop.
"""

import threading


class Worker(threading.Thread):

    def __init__(self, *, name: str | None = None):
        super().__init__(name=name, daemon=False)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        raise NotImplementedError("Subclasses must implement run()")