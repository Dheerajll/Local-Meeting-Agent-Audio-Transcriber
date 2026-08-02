from lma.audio_p.base import SpeechDetector
from lma.audio_p.chunk_buffer import ChunkBuffer
from lma.audio_p.chunk_builder import ChunkBuilder
from lma.publisher import ChunkPublisher

from lma.schemas import (AudioChunk,PCMFrame,)

from lma.constants import RecorderState


class AudioRecorder:
    """
    Coordinates the audio pipeline.

    Responsibilities:
        - Receive PCM frames
        - Ask the detector for speech events
        - Feed audio into the ChunkBuffer
        - Manage recorder state
        - Publish completed AudioChunks

    It intentionally knows nothing about:
        - FFmpeg
        - NumPy
        - Torch
        - Silero internals
    """

    def __init__(self,detector: SpeechDetector,chunk_buffer: ChunkBuffer,chunk_builder: ChunkBuilder,publisher: ChunkPublisher[AudioChunk]):

        self.detector = detector
        self.chunk_buffer = chunk_buffer
        self.chunk_builder = chunk_builder
        self.publisher = publisher

        self.state = RecorderState.IDLE

        self.is_speaking = False

        self.chunk_start_ms = 0

        self.last_speech_timestamp = 0

        self.over_soft_limit = False

    def process(self,frame: PCMFrame) -> None:
        """
        Process a single timestamped PCM frame.
        """

        event = self.detector.process(frame)

        self._handle_event(event,frame)

        if self.state != RecorderState.IDLE:
            self.chunk_buffer.add(frame.data)

        self._update_state(frame)

    def _handle_event(self,event,frame: PCMFrame) -> None:
        """
        Handle START / END events.

        Implemented in the next commit.
        """
        pass

    def _update_state(self,frame: PCMFrame) -> None:
        """
        Update recorder state.

        Implemented in the next commit.
        """
        pass