import queue
import os
import threading

from lma.audio.source import AudioSource
from lma.audio.recorder import AudioRecorder
from lma.audio.detector.silero import SileroDetector

from lma.audio.chunk_buffer import ChunkBuffer
from lma.audio.chunk_builder import ChunkBuilder
from lma.test_consumer import chunk_writer_worker
from lma.workers.publisher import ChunkPublisher
from lma.audio.capture import FFmpegCapture
from lma.audio.clock import AudioClock


os.makedirs(
    "test_chunks",
    exist_ok=True
)


chunk_queue = queue.Queue()


publisher = ChunkPublisher(
    chunk_queue
)


recorder = AudioRecorder(
    detector=SileroDetector(),

    chunk_buffer=ChunkBuffer(),

    chunk_builder=ChunkBuilder(),

    publisher=publisher,
)


writer = threading.Thread(
    target=chunk_writer_worker,
    args=(chunk_queue,),
    daemon=True,
)

writer.start()


capture = FFmpegCapture()
clock = AudioClock()
source = AudioSource(capture,clock)

print("🎙 Starting audio pipeline")
source.start()


try:

    for frame in source.frames():

        recorder.process(frame)


except KeyboardInterrupt:

    print(
        "Stopping..."
    )


finally:

    chunk_queue.put(None)

    writer.join(timeout=2)

    source.stop()