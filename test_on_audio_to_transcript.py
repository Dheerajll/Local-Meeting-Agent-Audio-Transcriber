import queue
import threading
import os
from pathlib import Path

from lma.audio.source import AudioSource
from lma.audio.recorder import AudioRecorder
from lma.audio.detector.silero import SileroDetector

from lma.audio.chunk_buffer import ChunkBuffer
from lma.audio.chunk_builder import ChunkBuilder
from lma.workers.publisher import ChunkPublisher

from lma.audio.capture import FFmpegCapture
from lma.audio.clock import AudioClock

from lma.audio.wav_writer import WavWriter
from lma.transcription.whisper_transcriber import WhisperTranscriber
from lma.transcription.transcript_writer import TranscriptWriter
from lma.transcription.worker import TranscriptionWorker
from lma.transcription.diarizer import Diarizer
from lma.transcription.word_aligner import WordAligner
from lma.transcription.speaker_manager import SpeakerManager
from lma.transcription.cache import get_model_snapshot_path
from lma.core.paths import create_session_dir


# --------------------------------------------------
# Session
# --------------------------------------------------

session_paths = create_session_dir(
    session_id="test"
)

print(
    f"Session: {session_paths.root}"
)


# --------------------------------------------------
# Queue
# --------------------------------------------------

chunk_queue = queue.Queue()


# --------------------------------------------------
# Publisher
# --------------------------------------------------

publisher = ChunkPublisher(
    chunk_queue
)


# --------------------------------------------------
# Audio recorder
# --------------------------------------------------

recorder = AudioRecorder(
    detector=SileroDetector(),

    chunk_buffer=ChunkBuffer(),

    chunk_builder=ChunkBuilder(),

    publisher=publisher,
)


# --------------------------------------------------
# Whisper
# --------------------------------------------------

MODEL_PATH = get_model_snapshot_path("mlx-community--whisper-large-v3-mlx")

transcriber = WhisperTranscriber(
    model_path=MODEL_PATH,
    language="en",
    diarizer=Diarizer(),
    speaker_manager=SpeakerManager(),
    word_aligner=WordAligner()
)


# --------------------------------------------------
# Writers
# --------------------------------------------------

wav_writer = WavWriter(
    session_paths.audio
)

transcript_writer = TranscriptWriter(
    session_paths.transcripts
)


# --------------------------------------------------
# Transcription worker
# --------------------------------------------------

worker = TranscriptionWorker(
    input_queue=chunk_queue,

    wav_writer=wav_writer,

    transcriber=transcriber,

    transcript_writer=transcript_writer,
)


worker.start()


# --------------------------------------------------
# Audio source
# --------------------------------------------------

capture = FFmpegCapture()

clock = AudioClock()

source = AudioSource(
    capture,
    clock,
)


print(
    "🎙 Starting audio pipeline"
)

source.start()


try:

    for frame in source.frames():

        recorder.process(frame)


except KeyboardInterrupt:

    print(
        "\nStopping..."
    )


finally:

    print(
        "🛑 Stopping audio source..."
    )

    source.stop()


    # --------------------------------------------------
    # Wait for all published chunks to be processed
    # --------------------------------------------------

    print(
        "⏳ Waiting for transcription queue..."
    )

    chunk_queue.join()


    # --------------------------------------------------
    # Stop transcription worker
    # --------------------------------------------------

    print(
        "🛑 Stopping transcription worker..."
    )

    worker.stop()


    print(
        "✓ Test finished"
    )