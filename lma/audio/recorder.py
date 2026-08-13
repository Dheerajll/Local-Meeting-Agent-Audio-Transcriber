from lma.audio.detector.base import SpeechDetector
from lma.audio.chunk_buffer import ChunkBuffer
from lma.audio.chunk_builder import ChunkBuilder
from lma.workers.publisher import ChunkPublisher

from lma.core.schemas import (AudioChunk,PCMFrame,)

from lma.core.constants import (RecorderState,SpeechEvent,
SOFT_LIMIT_MS,HARD_LIMIT_MS,FINALIZE_SILENCE_MS,
CHANNELS,SAMPLE_RATE,ChunkReason,FRAME_DURATION_MS
)


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



    def _handle_event(self,event: SpeechEvent | None,frame: PCMFrame) -> None:

        if event is None:
            return
        if event == SpeechEvent.START:

            self.is_speaking = True
            self.last_speech_timestamp = frame.timestamp_ms

            if self.state == RecorderState.IDLE:
                self.chunk_buffer.start()
                self.chunk_start_ms = max(0,frame.timestamp_ms - FRAME_DURATION_MS)
                self.over_soft_limit = False

                self.state = RecorderState.RECORDING

                print(f"🗣️ Chunk started @ {frame.timestamp_ms}ms")

            elif self.state == RecorderState.WAITING_FOR_RESUME:

                self.state = RecorderState.RECORDING
                print("▶️ Speech resumed")
            return

        if event == SpeechEvent.END:

            self.is_speaking = False

            self.last_speech_timestamp = frame.timestamp_ms

            if self.state == RecorderState.RECORDING:

                self.state = RecorderState.WAITING_FOR_RESUME

                print(f"⏳ Waiting for resume...")

    def _finalize(self,*,reason: ChunkReason,forced: bool) -> None:

        finalized = self.chunk_buffer.finalize()

        chunk = self.chunk_builder.build(
            chunk=finalized,
            start_ms=self.chunk_start_ms,
            reason=reason,
            forced=forced,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
        )

        self.publisher.publish(chunk)

        print(f"💾 Published chunk {chunk.chunk_id}")

        #
        # Reset recorder state
        #

        self.state = RecorderState.IDLE

        self.over_soft_limit = False

        self.last_speech_timestamp = 0

        #
        # If we hit the hard limit while
        # the speaker is still talking,
        # immediately continue recording.
        #

        if forced and self.is_speaking:

            self.chunk_buffer.start()

            self.chunk_start_ms = chunk.end_ms

            self.state = RecorderState.RECORDING

            print("▶️ Continuing after hard limit")

    def _update_state(self,frame: PCMFrame,):

        if self.state == RecorderState.IDLE:
            return

        duration = self.chunk_buffer.duration_ms()

        if (duration >= SOFT_LIMIT_MS and not self.over_soft_limit):
            self.over_soft_limit = True
            print("⏱ Soft limit reached")





        if duration >= HARD_LIMIT_MS:
            print("⚠ Hard limit reached")
            self._finalize(reason=ChunkReason.HARD_LIMIT,forced=True)
            return

        if self.state == RecorderState.WAITING_FOR_RESUME:

            silence = (frame.timestamp_ms-self.last_speech_timestamp)

            if silence >= FINALIZE_SILENCE_MS:

                print("🔇 Finalize on silence")

                reason = (ChunkReason.SOFT_LIMIT if self.over_soft_limit else ChunkReason.NATURAL_SILENCE)

                self._finalize(reason=reason,forced=False)
                return