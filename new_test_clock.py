from lma.audio.source import AudioSource
from lma.audio.capture import FFmpegCapture
from lma.audio.clock import AudioClock
from lma.audio.recorder import AudioRecorder
from lma.audio.detector.silero import SileroDetector
from lma.audio.chunk_buffer import ChunkBuffer
from lma.audio.chunk_builder import ChunkBuilder
from lma.workers.publisher import ChunkPublisher
from lma.queues import transcription_queue
source = AudioSource(
    FFmpegCapture(":0"),
    AudioClock(),
)

recorder = AudioRecorder(
    detector=SileroDetector(),
    chunk_buffer=ChunkBuffer(),
    chunk_builder=ChunkBuilder(),
    publisher=ChunkPublisher(transcription_queue),
)

source.start()

try:
    for frame in source.frames():
        recorder.process(frame)
 
finally:
    source.stop()