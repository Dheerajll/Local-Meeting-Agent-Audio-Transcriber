from lma.audio_p.source import AudioSource
from lma.audio_p.capture import FFmpegCapture
from lma.audio_p.clock import AudioClock
from lma.audio_p.recorder import AudioRecorder
from lma.audio_p.detector import SileroDetector
from lma.audio_p.chunk_buffer import ChunkBuffer
from lma.audio_p.chunk_builder import ChunkBuilder
from lma.publisher import ChunkPublisher
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